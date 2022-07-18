from transformers import T5TokenizerFast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import torch, json, os, multiprocessing, math
from torch.utils.tensorboard import SummaryWriter

from solver.solver import solve
from model.doctor import Doctor
from model.utils.get_fixed_embedding import get_fixed


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def parallelize(exp):
    for n, i in enumerate(exp):
        while len(exp[n]) < 3:
            exp[n].append('')
    return exp


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


def get_data(train_data, test_data, prefix):
    train_x = [prefix + i["question"] for i in train_data.values()]
    train_y = [parallelize(eval(i["expression"])) for i in train_data.values()]
    eval_x = [prefix + i["question"] for i in list(test_data.values())]
    eval_y = [parallelize(eval(i["expression"])) for i in test_data.values()]
    test_x = eval_x
    test_y = [i["answer"] for i in test_data.values()]
    return [train_x, train_y], [eval_x, eval_y], [test_x, test_y]


def run_train(model, train_data, optimizer, scheduler, clip, log, batch_size, iter, TB_writer=None):
    train_num_iter = math.ceil(len(train_data[0]) / batch_size)
    cum_loss = 0
    metric_losses = {key: 0 for key in
                     ['loss', 'noun_loss', 'operator_loss', 'operand_left_loss', 'operand_right_loss',
                      'noun_acc', 'operator_acc', 'operand_left_acc', 'operand_right_acc']}
    model.train()
    model.zero_grad()
    for idx in range(train_num_iter):
        input_str = train_data[0][batch_size * idx:batch_size * (idx + 1)]
        label_tokens = train_data[1][batch_size * idx:batch_size * (idx + 1)]
        inputs = tokenizer(input_str, return_tensors='pt', padding=True).to(device)
        result, metrics = model(input_str=input_str, input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                generate=False, label_tokens=label_tokens)
        for key in metric_losses.keys():
            metric_losses[key] += metrics[key]
        loss = result["loss"]
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        model.zero_grad()
        #if (idx + 1) % 2 == 0 or idx == train_num_iter:  # Wait for several backward steps
        #    clip_grad_norm_(model.parameters(), clip)
        #    optimizer.step()
        #    model.zero_grad()
        cum_loss += float(result["loss"])
        if (idx + 1) % math.ceil(train_num_iter / 3) == 0 or (idx + 1) == train_num_iter:
            line = f'(Epoch {iter + 1}) {idx + 1} / {train_num_iter} loss: {round(cum_loss / round(train_num_iter / 3), 4)}'
            print(line)
            log.write(line + "\n")
            cum_loss = 0
    for key in metric_losses.keys():
        if 'loss' in key:
            TB_writer.add_scalar(key + '/train', metric_losses[key], iter)
        elif 'acc' in key:
            acc = metric_losses[key]
            TB_writer.add_scalar(key + '/train', acc[0] / acc[1], iter)
    scheduler.step()
    torch.cuda.empty_cache()
    return model, optimizer


def run_eval(model, eval_data, log, batch_size, iter, TB_writer=None):
    eval_num_iter = math.ceil(len(eval_data[0]) / batch_size)
    model.eval()
    cum_loss = 0
    metric_losses = {key: 0 for key in
                     ['loss', 'operator_loss', 'operand_left_loss', 'operand_right_loss', 'operator_acc',
                      'operand_left_acc', 'operand_right_acc']}
    for idx in range(eval_num_iter):
        input_str = eval_data[0][batch_size * idx:batch_size * (idx + 1)]
        label_tokens = eval_data[1][batch_size * idx:batch_size * (idx + 1)]
        inputs = tokenizer(input_str, return_tensors='pt', padding=True).to(device)
        result, metrics = model(input_str=input_str, input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                generate=False, label_tokens=label_tokens)
        for key in metric_losses.keys():
            metric_losses[key] += metrics[key]

        cum_loss += float(result["loss"])
        if (idx + 1) % math.ceil(eval_num_iter / 3) == 0 or (idx + 1) == eval_num_iter:
            line = f'((EVAL) (Epoch {iter + 1}) {idx + 1} / {eval_num_iter} loss: {round(cum_loss / round(eval_num_iter / 3), 4)}'
            print(line)
            log.write(line + "\n")
            cum_loss = 0
    for key in metric_losses.keys():
        if 'loss' in key:
            TB_writer.add_scalar(key + '/eval', metric_losses[key], iter)
        elif 'acc' in key:
            acc = metric_losses[key]
            TB_writer.add_scalar(key + '/eval', acc[0] / acc[1], iter)
    torch.cuda.empty_cache()
    return metric_losses


def run_test(model, test_data, log, batch_size):
    score = 0
    test_num_iter = math.ceil(len(test_data[0]) / batch_size)
    pool = multiprocessing.Pool(4)
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
            if ans == a:
                score += 1
            pool.close()
            pool.join()
            pool = multiprocessing.Pool(4)

    accuracy = round(score / len(test_data[0]) * 100, 4)
    line = f'(Eval) - Answer Accuracy: {accuracy}%'
    print(line)
    log.write(line + "\n")
    torch.cuda.empty_cache()
    return accuracy


model_name = "KETI-AIR/ke-t5"
model_size = "base"
tokenizer = T5TokenizerFast.from_pretrained(f"{model_name}-{model_size}")
device = 'cuda'
seed = 607


def train_eval_loop(lr=3e-4, dropout=0.1, batch_size=64, fold=0):
    get_fixed(model_size=model_size)
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    dataset = f"noun_fold{fold}"
    data_path = Path(__file__).parents[1] / 'dataset' / 'noun' / 'fold'
    model_path = Path(__file__).parents[0] / f'model_save/{model_size}/{dataset}/dr{dropout}_lr{lr}'
    os.makedirs(model_path, exist_ok=True)
    data = load_json(data_path / f'{dataset}.json')
    train_data = data["train"]
    test_data = data["test"]
    torch.manual_seed(seed)
    prefix = "Solve the math word problem: "
    train_data, eval_data, test_data = get_data(train_data, test_data, prefix)
    num_train = 500
    num_print_test = 20

    log_path = model_path / 'log.txt'
    tensorboard_path = model_path / 'tensorboard'
    TB_writer = SummaryWriter(tensorboard_path)
    log = open(log_path, 'w+')

    doctor = Doctor(tokenizer=tokenizer, device=device, model_name=model_name,
                    model_size=model_size, dropout=dropout).to(device)
    optimizer = AdamW([param for param in doctor.parameters() if param.requires_grad==True], lr=lr)
    scheduler = WarmupConstantSchedule(optimizer, warmup_steps=num_train * 0.01)

    for iter in range(num_train):
        doctor, optimizer = run_train(model=doctor, train_data=train_data, optimizer=optimizer, scheduler=scheduler,
                                      clip=10, log=log, batch_size=batch_size, iter=iter, TB_writer=TB_writer)
        with torch.no_grad():
            run_eval(model=doctor, eval_data=eval_data, log=log, batch_size=batch_size, iter=iter, TB_writer=TB_writer)
            if (iter + 1) % num_print_test == 0:
                accuracy = run_test(model=doctor, test_data=test_data, log=log, batch_size=batch_size)
                TB_writer.add_scalar('answer accuracy/test', accuracy, iter)
    # torch.save(doctor.state_dict(), model_path / f"epoch_{num_train}.pt")
    # torch.save(optimizer.state_dict(), model_path / f"epoch_{num_train}_opt.pt")
    log.close()


if __name__ == '__main__':
    for lr in [3e-4]:
        for fold in range(5):
            train_eval_loop(lr=lr, dropout=0.1, fold=fold)
