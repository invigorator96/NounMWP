from transformers import T5TokenizerFast
from pathlib import Path
import torch, json, os, multiprocessing, random, math

from solver.solver import solve
from model.doctor import Doctor
from model.utils.get_fixed_embedding import get_fixed

ARITYDICT = {'TAR': 2, 'arg': 0, 'END': 1, '+': 2, '-': 2, '*': 2, '/': 2, '=': 2, 'find': 2, 'ord': 2}


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8-sig') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


def parallelize(exp):
    for n, i in enumerate(exp):
        while len(exp[n]) < 3:
            exp[n].append('')
    return exp


def get_data(train_data, test_data, prefix):
    train_x = [prefix + i["question"] for i in train_data.values()]
    train_y = [parallelize(eval(i["expression"])) for i in train_data.values()]
    eval_x = [prefix + i["question"] for i in list(test_data.values())]
    eval_y = [parallelize(eval(i["expression"])) for i in test_data.values()]
    test_x = eval_x
    test_y = [[parallelize(eval(i["expression"])), i["answer"]] for i in test_data.values()]
    return [train_x, train_y], [eval_x, eval_y], [test_x, test_y]


def output_process(expression):
    var_num = 0
    for n1, i1 in enumerate(expression):
        if i1[0] == 'TAR':
            expression[n1][2] = f"X{var_num}"
            var_num += 1
        elif i1[0] in ['ord', 'find']:
            expression[n1][1] = f"R{n1 - 1}"
        elif i1[0] == 'END':
            expression = expression[:n1 + 1]
            expression[n1] = ['END', f"R{n1 - 1}"]
            break
    for n1, i1 in enumerate(expression):
        for n2, i2 in enumerate(i1):
            if i2.replace('-', '').replace('/', '').replace('.', '').isnumeric():
                expression[n1][n2] = eval(i2)
    return expression


def solve_singleprocess(expr_seq):
    try:
        y = solve(expr_seq)
        if type(y) is list:
            ans = list()
            for x in y:
                ans_tmp, _ = x.writeAnswer()
                ans.append(ans_tmp)
        else:
            ans, _ = y.writeAnswer()
        return ans
    except:
        return None


def run_test(model, test_data, batch_size):
    score = 0
    test_num_iter = math.ceil(len(test_data[0]) / batch_size)
    pool = multiprocessing.Pool(4)
    result = {"correct": dict(), "incorrect": dict()}
    key = 1
    for idx in range(test_num_iter):
        print(f"processing rate: {round(idx / test_num_iter, 4) * 100}%")
        input_str = test_data[0][batch_size * idx:batch_size * (idx + 1)]
        answers = test_data[1][batch_size * idx:batch_size * (idx + 1)]
        inputs = tokenizer(input_str, return_tensors='pt', padding=True).to(device)
        gen = model.generate(input_str=input_str, input_ids=inputs['input_ids'],
                             attention_mask=inputs['attention_mask'], max_len=16)
        for q, a, g in zip(input_str, answers, gen):
            res = output_process(g)
            ans = pool.apply(solve_singleprocess, [res])
            line = {"question": q, "answer": str(a[1]), "expr": str(a[0]), "y_hat": str(ans), "expr_hat": str(res)}
            if ans == a[1]:
                score += 1
                result["correct"][key] = line
            else:
                result["incorrect"][key] = line
            key += 1
            pool.close()
            pool.join()
            pool = multiprocessing.Pool(4)
    accuracy = round(score / len(test_data[0]) * 100, 4)
    line = f'(Eval) - Answer Accuracy: {accuracy}%'
    print(line)
    return result



model_name = "KETI-AIR/ke-t5"
model_size = "base"
tokenizer = T5TokenizerFast.from_pretrained(f"{model_name}-{model_size}")
device = 'cuda'
seed = 607


def test(lr=3e-4, dropout=0.1, batch_size=64, fold=0):
    # get_fixed(model_size=model_size)
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    dataset = f"noun_fold{fold}"
    data_path = Path(__file__).parents[1] / 'dataset' / 'noun' / 'fold'
    model_path = Path(__file__).parents[0] / f'model_save/{model_size}_model_saved/{dataset}/dr{dropout}_lr{lr}'
    data = load_json(data_path / f'{dataset}.json')
    train_data = data["train"]
    test_data = data["test"]
    torch.manual_seed(seed)
    prefix = "Solve the math word problem: "
    _, _, test_data = get_data(train_data, test_data, prefix)
    doctor = Doctor(tokenizer=tokenizer, device=device, model_name=model_name,
                    model_size=model_size, dropout=dropout).to(device)
    state_dict = torch.load(model_path / f"max.pt")
    doctor.load_state_dict(state_dict)

    with torch.no_grad():
        result = run_test(model=doctor, test_data=test_data, batch_size=batch_size)
    write_json(result, f'../result/res_fold/result_fold{fold}.json')


if __name__ == '__main__':
    for i in range(5):
        test(fold=i)
