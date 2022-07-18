from model.util import DecoderEmbed, ExtractInfo, DecoderDecode, OPERATOR_DICT
from model.util import get_float_info, emb_matrix_tensor, weight_to_index, get_noun_info, compute_accuracy, build_masks
from transformers import T5Model
from model.loss import SmoothedCrossEntropyLoss
from torch import nn
import numpy as np
import torch, math

MODEL_SIZE = {"small": 512, "base": 768, "large": 1024}


class OperandAttention(nn.Module):
    def __init__(self, d_model=None):
        super().__init__()
        self.d_model = d_model
        self.opnd_layer_norm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.opt_linear1 = nn.Linear(self.d_model, self.d_model)
        self.opt_linear2 = nn.Linear(self.d_model, self.d_model)
        self.opnd_linear1 = nn.Linear(self.d_model, self.d_model)
        self.opnd_linear2 = nn.Linear(self.d_model, self.d_model)

    def forward(self, output, opnd_embeddings):
        opnd_embeddings = self.opnd_layer_norm(opnd_embeddings)
        left_query = self.opt_linear1(output) / math.sqrt(self.d_model)
        left_key = self.opnd_linear1(opnd_embeddings)
        left_operand= torch.bmm(left_query, left_key.transpose(1, 2))
        right_query = self.opt_linear2(output) / math.sqrt(self.d_model)
        right_key = self.opnd_linear2(opnd_embeddings)
        right_operand = torch.bmm(right_query, right_key.transpose(1, 2))
        return left_operand, right_operand


class NounAttention(nn.Module):
    def __init__(self, d_model=None):
        super().__init__()
        self.d_model = d_model
        self.query_linear = nn.Linear(self.d_model, self.d_model)
        self.key_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, output):
        query = self.query_linear(output.last_hidden_state) / math.sqrt(self.d_model)
        key = self.key_linear(output.encoder_last_hidden_state)
        noun_weight = torch.bmm(query, key.transpose(1, 2))
        return noun_weight


class Doctor(nn.Module):
    def __init__(self, tokenizer=None, model_name=None, model_size=None,
                 dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.d_model = MODEL_SIZE[model_size]
        self.tokenizer = tokenizer
        self.extract_info = ExtractInfo(tokenizer)
        self.decoder_embedding = DecoderEmbed(device=self.device)
        self.decode = DecoderDecode(device=self.device)
        self.T5 = T5Model.from_pretrained(f"{model_name}-{model_size}", dropout_rate=dropout)
        for para in self.T5.get_input_embeddings().parameters():
            para.requires_grad = False
        self.lm_head = nn.Linear(self.d_model, len(OPERATOR_DICT), bias=False)
        self.operand_attention = OperandAttention(d_model=self.d_model)
        self.noun_attention = NounAttention(d_model=self.d_model)
        self.loss_func = SmoothedCrossEntropyLoss(ignore_index=-100)

    def forward(self, input_str: list, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, generate=False,
                label_tokens=None):
        float_dict, float_emb = get_float_info(model=self.T5, extractor=self.extract_info,
                                               input_str=input_str, input_ids=input_ids, device=self.device)

        # 모델에 먹여주는 부분
        label_idx, decoder_input_embedding, operand_embeddings = \
            self.preprocess(label_tokens=label_tokens, float_dict=float_dict, float_emb=float_emb)
        operand_embeddings, pad_mask = emb_matrix_tensor(operand_embeddings, device=self.device)
        decoder_mask = build_masks(label_idx=label_idx, device=self.device, generate=generate)
        output = self.T5(input_ids=input_ids, attention_mask=attention_mask,
                         decoder_inputs_embeds=decoder_input_embedding,
                         decoder_attention_mask=decoder_mask)
        lm_output = self.lm_head(output.last_hidden_state)
        operand_left, operand_right = self.operand_attention(output=output.last_hidden_state,
                                                             opnd_embeddings=operand_embeddings)
        noun_weight = self.noun_attention(output=output)
        

        if generate:
            result = {"predicted": list()}
            index_list = weight_to_index(lm_output=lm_output, operand_left=operand_left,
                                         operand_right=operand_right, pad_mask=pad_mask)
            for index, floats in zip(index_list, float_dict):
                predicted = self.decode(indexes=index, float_idx_dict=floats)
                result["predicted"].append(predicted)
            noun_pred = torch.argmax(noun_weight, dim=2)
            result["predicted"] = self.postprocess(predicted=result["predicted"], input_str=input_str,
                                                   input_ids=input_ids, noun_label=noun_pred)
            return result
        else:
            result = dict()
            metrics = dict()
            lm_output = lm_output[:, :-1, :]
            operand_left = operand_left[:, :-1, :]
            operand_right = operand_right[:, :-1, :]
            noun_label = get_noun_info(extractor=self.extract_info, input_str=input_str,
                                       label=label_tokens, device=self.device)
            noun_weight = noun_weight[:, :-1, :]

            metrics['noun_loss'] = float(self.loss_func(noun_weight.transpose(2, 1), target=noun_label))
            metrics['operator_loss'] = float(self.loss_func(lm_output.transpose(2, 1), target=label_idx[:, :, 0]))
            metrics['operand_left_loss'] = float(
                self.loss_func(operand_left.transpose(2, 1), target=label_idx[:, :, 1]))
            metrics['operand_right_loss'] = float(
                self.loss_func(operand_right.transpose(2, 1), target=label_idx[:, :, 2]))

            noun_correct, noun_total = compute_accuracy(noun_label, noun_weight.transpose(2, 1), ignore_index=-100)
            operator_correct, operator_total = compute_accuracy(label_idx[:, :, 0],
                                                                lm_output.transpose(2, 1),
                                                                ignore_index=-100)
            operand_left_correct, operand_left_total = compute_accuracy(label_idx[:, :, 1],
                                                                        operand_left.transpose(2, 1),
                                                                        ignore_index=-100)
            operand_right_correct, operand_right_total = compute_accuracy(label_idx[:, :, 2],
                                                                          operand_right.transpose(2, 1),
                                                                          ignore_index=-100)
                                                                          
            metrics['operator_acc'] = np.array([operator_correct, operator_total])
            metrics['noun_acc'] = np.array([noun_correct, noun_total])
            metrics['operand_left_acc'] = np.array([operand_left_correct, operand_left_total])
            metrics['operand_right_acc'] = np.array([operand_right_correct, operand_right_total])

            loss = self.loss_func(lm_output.transpose(2, 1), target=label_idx[:, :, 0])
            loss += self.loss_func(operand_left.transpose(2, 1), target=label_idx[:, :, 1])
            loss += self.loss_func(operand_right.transpose(2, 1), target=label_idx[:, :, 2])
            loss += self.loss_func(noun_weight.transpose(2, 1), target=noun_label)
            metrics['loss'] = float(loss)
            result["loss"] = loss
            return result, metrics

    def generate(self, input_str, input_ids, attention_mask, max_len=16):
        output = self.forward(input_str=input_str, input_ids=input_ids, attention_mask=attention_mask, generate=True,
                              label_tokens=None)["predicted"]
        for _ in range(max_len):
            output = self.forward(input_str=input_str, input_ids=input_ids, attention_mask=attention_mask,
                                  generate=True, label_tokens=output)["predicted"]
            counter = 0
            for i in output:
                if "END" in str(i):
                    counter += 1
            if counter == len(output):
                break
        return output

    def preprocess(self, label_tokens, float_dict, float_emb):
        label_idx, decoder_input_embedding, operand_embeddings = list(), list(), list()
        if label_tokens:
            max_label_len = max([len(i) for i in label_tokens])
            for label, dic, emb in zip(label_tokens, float_dict, float_emb):
                label_idx.append(self.decode.return_index(tokens=label, float_idx_dict=dic, max_len=max_label_len))
                if max_label_len > len(label):
                    label = [['BEGIN','','']] + label
                    line1 = torch.cat([self.decoder_embedding(label, float_idx_dict=dic, float_operand_emb=emb),
                                       torch.zeros(max_label_len - len(label) + 1, self.d_model).to(self.device)])
                else:
                    label = [['BEGIN','','']] + label
                    line1 = torch.cat([self.decoder_embedding(label, float_idx_dict=dic, float_operand_emb=emb)])
                decoder_input_embedding.append(line1)
                line2 = self.decoder_embedding.return_operand_embedding(float_operand_emb=emb)
                operand_embeddings.append(line2)
            label_idx = torch.stack(label_idx)
            decoder_input_embedding = torch.stack(decoder_input_embedding)
            return label_idx, decoder_input_embedding, operand_embeddings
        else:
            for emb in float_emb:
                label = [['BEGIN', '', '']]
                line1 = torch.cat([self.decoder_embedding(label, float_idx_dict=None, float_operand_emb=emb)])
                decoder_input_embedding.append(line1.to(self.device))
                line2 = self.decoder_embedding.return_operand_embedding(float_operand_emb=emb)
                operand_embeddings.append(line2)
            decoder_input_embedding = torch.stack(decoder_input_embedding)
            return None, decoder_input_embedding, operand_embeddings

    def postprocess(self, predicted, input_str, input_ids, noun_label):
        result = list()
        for n1, (tuples, text, ids) in enumerate(zip(predicted, input_str, input_ids)):
            line = [list(i) for i in tuples]
            for n2, token in enumerate(tuples):
                if token[0] == 'TAR':
                    tgt = noun_label[n1][n2]
                    target_noun = self.tokenizer.decode(ids[tgt:tgt + 4]).split(' ')[0]
                    line[n2][1] = noun_process(noun=target_noun)
            result.append(line)
        return result


def noun_process(noun):
    if noun[-3:] in ["까지는", "까지의", "입니다"]:
        noun = noun[:-3]
    if len(noun) >= 3 and noun[-2:] in ['이네', '이의', '이와', '이는', '이를', '에게', "까지", "께서"]:
        noun = noun[:-2]
    if len(noun) >= 2 and noun[-1] in ['은', '는', '이', '가', '와', '과', '을', '를', '에', '게', '의', '네', ',']:
        noun = noun[:-1]
    return noun
