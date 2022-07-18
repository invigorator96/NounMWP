from transformers import T5Tokenizer, T5Model
from pathlib import Path
import torch, json, csv


def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


def write_csv(result, path):
    with open(path, 'w', newline='', encoding='UTF-8') as f:
        wr = csv.writer(f)
        for i in result:
            wr.writerow(i)
        f.close()


def get_embedding(input_id, model, tokenizer):
    emb = model.get_input_embeddings().forward(input_id).mean(dim=0)
    return emb


def get_fixed(model_size='base'):
    file_path = Path(__file__).parents[0]
    model_path = f"KETI-AIR/ke-t5-{model_size}"
    model = T5Model.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    fixed_dict = load_json(file_path / 'fixed_dict.json')
    result_json = dict()
    result_emb_operator = list()
    result_emb_operand = list()
    for key, values in fixed_dict.items():
        result_json[key] = dict()
        for n, value in enumerate(values):
            input_id = [i for i in tokenizer(value.split('_')[0]).input_ids if i not in [0, 1, 7]]
            embedding = get_embedding(torch.LongTensor(input_id), model, tokenizer).tolist()
            result_json[key][value] = n
            if key == "operator":
                result_emb_operator.append(embedding)
            else:
                result_emb_operand.append(embedding)
    write_json(result_json, file_path / "fixed_idx_dict.json")
    write_csv(result_emb_operator, file_path / "fixed_operator_emb.csv")
    write_csv(result_emb_operand, file_path / "fixed_operand_emb.csv")


if __name__ == '__main__':
    get_fixed()
