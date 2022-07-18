from pathlib import Path
from torch import nn
from typing import Union
from transformers import T5TokenizerFast
import torch, json, csv, re, copy, math


def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def load_csv(path):
    with open(path, 'r') as f:
        file = [x for x in csv.reader(f.readlines(), quoting=csv.QUOTE_NONNUMERIC)]
    return file


OPERATOR_DICT = {
    "begin_0": "BEGIN",
    "target_0": "TAR",
    "argument_0": "arg",
    "end_1": "END",
    "add_2": "+",
    "subtract_2": "-",
    "multiply_2": "*",
    "divide_2": "/",
    "equal_2": "=",
    "find_2": "find",
    "order_2": "ord"
}

OPERATOR_DICT_REV = {v: k for k, v in OPERATOR_DICT.items()}


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        div_term = (torch.arange(0, embedding_dim) // 2) * 2
        div_term = torch.exp(div_term.float() * (-math.log(10000.0) / embedding_dim))
        multiplier = torch.zeros(2, embedding_dim, dtype=torch.float)
        multiplier[0, 1::2] = 1.0
        multiplier[1, 0::2] = 1.0

        self.register_buffer('_div_term', div_term)
        self.register_buffer('multiplier', multiplier)

    @property
    def device(self) -> torch.device:
        return self._div_term.device

    def before_trigonometric(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.float()
        return indices * self._div_term

    def forward(self, index_or_range: Union[torch.Tensor, int, range], ignored_index=-100) -> torch.Tensor:
        with torch.no_grad():
            return self._forward(index_or_range, ignored_index)

    def _forward(self, index_or_range: Union[torch.Tensor, int, range], ignored_index=-100) -> torch.Tensor:
        if type(index_or_range) is int:
            indices = torch.tensor(index_or_range)
        elif type(index_or_range) is range:
            indices = torch.as_tensor(list(index_or_range))
        else:
            indices = index_or_range
        indices = indices.unsqueeze(-1)
        indices = indices.to(self.device)
        phase = self.before_trigonometric(indices)

        cos_value = phase.cos()
        sin_value = phase.sin()
        cos_multiplier = self.multiplier[0]
        sin_multiplier = self.multiplier[1]
        result_shape = [1] * (phase.dim() - 1) + [-1]
        cos_multiplier = cos_multiplier.view(*result_shape)
        sin_multiplier = sin_multiplier.view(*result_shape)

        result = cos_value * cos_multiplier + sin_value * sin_multiplier
        ignored_indices = (indices == ignored_index)
        if ignored_indices.any():
            result.masked_fill_(ignored_indices, 0.0)
        return result.contiguous()


class DecoderEmbed(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.util_path = Path(__file__).parents[0] / "utils"
        self.fixed_idx_dict = load_json(self.util_path / "fixed_idx_dict.json")
        self.fixed_operator_emb = torch.Tensor(load_csv(self.util_path / "fixed_operator_emb.csv"))
        self.fixed_operand_emb = torch.Tensor(load_csv(self.util_path / "fixed_operand_emb.csv"))
        self.fixed_operator_emb = nn.Embedding.from_pretrained(self.fixed_operator_emb, freeze=True)
        self.fixed_operand_emb = nn.Embedding.from_pretrained(self.fixed_operand_emb, freeze=True)
        self.d_model = self.fixed_operator_emb.weight.shape[-1]
        self.operand_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.d_model)
        self.memory_embedding = PositionalEncoding(self.d_model)
        self.degrade_factor = 1.0
        self.operator_param = torch.nn.Parameter(torch.tensor(self.degrade_factor), requires_grad=True)
        self.operand_param = torch.nn.Parameter(torch.tensor(self.degrade_factor), requires_grad=True)
        self.opt_layer_norm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.opnd_layer_norm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.linear = nn.Linear(self.d_model * 3, self.d_model)

    def forward(self, tokens: list, float_operand_emb: list, float_idx_dict: dict):
        return self.get_embedding(tokens, float_operand_emb, float_idx_dict)

    def compute_input(self, res_token: list):
        output_embed = torch.cat([self.opt_layer_norm(res_token[0]),
                                  self.opnd_layer_norm(res_token[1]),
                                  self.opnd_layer_norm(res_token[2])], dim=1)
        output_embed = self.linear(output_embed)
        return output_embed

    def return_operand_embedding(self, float_operand_emb: list):
        operand_embedding = list()
        for k in self.fixed_idx_dict["constant"].keys():
            operand_embedding.append(self.get_opnd_fixed_embedding(k))
        for k in range(16):
            operand_embedding.append(self.get_opnd_memory_embedding(element=None, index=k))
        for k in range(len(float_operand_emb)):
            operand_embedding.append(self.get_opnd_float_embedding(float_operand_emb, index=k))
        return torch.cat(operand_embedding)

    def get_embedding(self, tokens: list, float_operand_emb: list, float_idx_dict: dict):
        result = list()
        for n1, tuples in enumerate(tokens):
            res_token = list()
            arity = 1
            for n2, element in enumerate(tuples):
                if int(arity) >= n2:
                    if element in OPERATOR_DICT_REV.keys():
                        res_token.append(self.get_opt_embedding(element, n1))
                        arity = OPERATOR_DICT_REV[element].split("_")[1]
                    elif element in float_idx_dict.keys():
                        index = float_idx_dict[element]
                        res_token.append(self.get_opnd_float_embedding(float_operand_emb, index))
                    elif 'R' in element:
                        res_token.append(self.get_opnd_memory_embedding(element=element, index=None))
                    elif element in self.fixed_idx_dict["constant"].keys():
                        res_token.append(self.get_opnd_fixed_embedding(element))
                    else:
                        raise NotImplementedError(f"{element}: Token is not in target dictionary.")
                else:
                    res_token.append(torch.Tensor([0] * self.d_model).unsqueeze(0).to(self.device))
            result.append(self.compute_input(res_token))
        return torch.cat(result)

    def get_opt_embedding(self, element, n):
        c_f = self.operator_param
        e_f = self.fixed_operator_emb(
            torch.LongTensor([self.fixed_idx_dict["operator"][OPERATOR_DICT_REV[element]]]).to(
                self.device))
        pe = self.memory_embedding(n+1)
        return c_f * e_f + pe

    def get_opnd_float_embedding(self, float_operand_emb, index):
        c_a = self.operand_param
        u_num = self.operand_embedding(torch.LongTensor([0]).to(self.device))
        e_aij = float_operand_emb[index]
        return c_a * u_num + e_aij

    def get_opnd_memory_embedding(self, element=None, index=None):
        c_a = self.operand_param
        u_expr = self.operand_embedding(torch.LongTensor([1]).to(self.device))
        if element and not index:
            index = int(element.replace('R', ''))
        pe = self.memory_embedding(index+1)
        return c_a * u_expr + pe

    def get_opnd_fixed_embedding(self, element):
        c_a = self.operand_param
        u_const = self.operand_embedding(torch.LongTensor([2]).to(self.device))
        e_c = self.fixed_operand_emb(
            torch.LongTensor([self.fixed_idx_dict["constant"][element]]).to(self.device))
        return c_a * u_const + e_c


class DecoderDecode:
    def __init__(self, device):
        self.device = device
        self.util_path = Path(__file__).parents[0] / "utils"
        self.fixed_idx_dict = load_json(self.util_path / "fixed_idx_dict.json")
        self.operator_dict = {v: k for k, v in self.fixed_idx_dict["operator"].items()}
        self.operand_dict = {v: k for k, v in self.fixed_idx_dict["constant"].items()}
        self.operand_dict.update({v + len(self.operand_dict): k for v, k in enumerate([f'R{i}' for i in range(16)])})
        self.operand_embedding = load_csv(Path(__file__).parents[0] / "utils" / "fixed_operand_emb.csv")

    def __call__(self, indexes: list, float_idx_dict: dict):
        float_idx_dict = {v + len(self.operand_dict): k for k, v in float_idx_dict.items()}
        operand_dict_temp = copy.deepcopy(self.operand_dict)
        operand_dict_temp.update(float_idx_dict)
        return self.get_decoded(indexes, operand_dict_temp)

    def get_decoded(self, indexes: list, operand_dict: dict):
        result = list()
        for index in indexes:
            arity = int(self.operator_dict[index[0]].split("_")[1])
            if arity == 2:
                res = [OPERATOR_DICT[self.operator_dict[index[0]]],
                       operand_dict[index[1]].split("_")[0],
                       operand_dict[index[2]].split("_")[0]]
            elif arity == 1:
                res = [OPERATOR_DICT[self.operator_dict[index[0]]],
                       operand_dict[index[1]].split("_")[0],
                       '']
            else:
                res = [OPERATOR_DICT[self.operator_dict[index[0]]],
                       '',
                       '']
            result.append(res)
        return result

    def return_index(self, tokens: list, float_idx_dict: dict, max_len: int):
        float_idx_dict = {k: v + len(self.operand_dict) for k, v in float_idx_dict.items()}
        operand_dict_temp = {k: v for v, k in self.operand_dict.items()}
        operand_dict_temp.update(float_idx_dict)
        result = list()
        for tuples in tokens:
            arity = int(OPERATOR_DICT_REV[tuples[0]].split('_')[1])
            if arity == 2:
                res = [self.fixed_idx_dict["operator"][OPERATOR_DICT_REV[tuples[0]]],
                       operand_dict_temp[tuples[1]],
                       operand_dict_temp[tuples[2]]]
            else:
                res = [self.fixed_idx_dict["operator"][OPERATOR_DICT_REV[tuples[0]]],
                       -100,
                       -100]
            result.append(res)
        result = result + ([[-100, -100, -100]] * (max_len - len(result)))
        return torch.LongTensor(result).to(self.device)


class ExtractInfo:
    def __init__(self, tokenizer: T5TokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, text, encoder_last_hidden_state):
        num_index = self.extract_num_index(text)
        embedding_matrix = self.get_embedding_matrix(encoder_last_hidden_state, num_index)
        index_dict = {k: n for n, k in enumerate(num_index.keys())}
        return index_dict, embedding_matrix

    def get_embedding_matrix(self, encoder_last_hidden_state, num_index):
        embedding_matrix = list()
        for k, v in num_index.items():
            embedding_matrix.append(encoder_last_hidden_state[v[0]:v[1] + 1].mean(dim=0).unsqueeze(0))
        return embedding_matrix

    def regx_hunter_with_index(self, regx, text):
        found = re.finditer(regx, text)
        result = [(i.group(), i.start(), i.end()) for i in found]
        return result

    def extract_num_index(self, text):
        num_found = self.regx_hunter_with_index('[0-9\-]*[./]?[0-9]+', text)
        token_mapping = self.tokenizer([text], is_split_into_words=True, return_offsets_mapping=True).offset_mapping
        num_index = dict()
        for num in num_found:
            flag = False
            for n, j in enumerate(token_mapping):
                if num[1] == j[0]+1:
                    idx1 = copy.copy(n)
                    flag = True
                if num[2] <= j[1] and flag:
                    idx2 = copy.copy(n)
                    key = num[0]
                    if key in num_index.keys():
                        i = 2
                        while f"{key}_{i}" in num_index.keys():
                            i += 1
                        key = f"{key}_{i}"
                    num_index[key] = (idx1, idx2)
                    break
        return num_index

    def make_noun_label(self, text, tokens):
        noun = make_noun_regx(tokens)
        noun_idx = self.extract_noun(text, noun)
        return [noun_idx[i[1]] for i in tokens if (i[0] == 'TAR')]

    def extract_noun(self, text, noun):
        noun_found = self.regx_hunter_with_index(noun, text)
        token_mapping = self.tokenizer([text], is_split_into_words=True, return_offsets_mapping=True).offset_mapping
        noun_index = dict()
        for noun in noun_found:
            for n, j in enumerate(token_mapping):
                if noun[1] < j[1]:
                    key = copy.copy(noun[0])
                    if key in noun_index.keys():
                        break
                    noun_index[key] = n
                    break
        return noun_index


def regx_hunter(regx, text):
    reg = re.compile(regx)
    return reg.findall(text)


def make_noun_regx(token):
    noun_list = [i[1] for i in token if i[0] == 'TAR']
    return '|'.join(noun_list)


def emb_matrix_tensor(operand_embeddings, device):
    batch_size = len(operand_embeddings)
    max_len = max([i.shape[0] for i in operand_embeddings])
    embedding_size = operand_embeddings[0].shape[1]
    emb = torch.zeros(batch_size, max_len, embedding_size).to(device)
    mask = torch.zeros(batch_size, max_len).to(device)
    for n, i in enumerate(operand_embeddings):
        emb[n][:i.shape[0]] = i
        mask[n][i.shape[0]:] = 1
    mask = mask.bool()
    return emb, mask


def weight_to_index(lm_output, operand_left, operand_right, pad_mask):
    operand_left = torch.masked_fill(operand_left, pad_mask.unsqueeze(1), float('-inf'))
    operand_right = torch.masked_fill(operand_right, pad_mask.unsqueeze(1), float('-inf'))
    operator_argmax = lm_output.argmax(dim=2)
    operand_left_argmax = operand_left.argmax(dim=2)
    operand_right_argmax = operand_right.argmax(dim=2)
    index_list = torch.cat([operator_argmax.unsqueeze(2), operand_left_argmax.unsqueeze(2),
                            operand_right_argmax.unsqueeze(2)], dim=2).tolist()
    return index_list


def get_float_info(model, extractor, input_str, input_ids, device):
    float_dict, float_emb = list(), list()
    dummy_input = torch.zeros(len(input_ids), 1, dtype=torch.int).to(device)
    encoder_last_hidden_state = model(input_ids=input_ids, decoder_input_ids=dummy_input).encoder_last_hidden_state
    for text, hidden in zip(input_str, encoder_last_hidden_state):
        num_dict, num_emb = extractor(text, hidden)
        float_dict.append(num_dict)
        float_emb.append(num_emb)
    return float_dict, float_emb


def get_noun_info(extractor, input_str, label, device):
    noun_label = list()
    max_token_len = max([len(i) for i in label])
    for text, tokens in zip(input_str, label):
        noun_info = extractor.make_noun_label(text, tokens)
        noun_label.append(noun_info + ([-100] * (max_token_len - len(noun_info))))
    return torch.LongTensor(noun_label).to(device)


def compute_accuracy(target, outputs, ignore_index=None):
    arg = outputs.argmax(dim=1)
    if ignore_index is None:
        n_total = target.shape[0] * target.shape[1]
        n_correct = torch.sum(torch.tensor(arg == target))
    else:
        ignore_mask = torch.tensor(target != ignore_index)
        n_total = torch.sum(ignore_mask)
        n_correct = torch.sum(torch.logical_and(torch.tensor(arg == target), ignore_mask))
    return int(n_correct), int(n_total)


def build_masks(label_idx, device, generate):
    if generate:
        return None
    else:
        label_idx = label_idx.select(dim=2, index=0)
        label = torch.cat([torch.zeros(label_idx.shape[0], 1).to(device), label_idx], dim=1)
        masks = (label != -100).to(device)
    return masks