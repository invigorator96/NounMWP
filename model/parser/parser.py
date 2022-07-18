from itertools import permutations
from pathlib import Path
import json, re, copy


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def write_json(res, path):
    with open(path, 'w', encoding='utf-8-sig') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()


def regx_hunter(regx, text, lower=True):
    if lower:
        reg = re.compile(regx, re.I)
    else:
        reg = re.compile(regx)
    return reg.findall(text)


def multiple_100(match):
    value = float(match.group()) * 100
    if int(value) == value:
        return str(int(value))
    else:
        return str(value)


def multiple_1000(match):
    value = float(match.group()) * 1000
    if int(value) == value:
        return str(int(value))
    else:
        return str(value)


def div_100(match):
    value = float(match.group()) / 100
    if int(value) == value:
        return str(int(value))
    else:
        return str(value)


def div_1000(match):
    value = float(match.group()) / 1000
    if int(value) == value:
        return str(int(value))
    else:
        return str(value)


def unit_parser(text, unit_dict):
    for k, v in unit_dict.items():
        regx = '[몇0-9\.]+ ?(' + '|'.join(v) + ')[^a-zA-Z]'
        regx_hunted = regx_hunter(regx, text)
        if regx_hunted and len(set(regx_hunted)) > 1:  # 등장하는 단위가 두 개 이상일 경우만
            ## 일단 바꿈의 대상이 되는 단위를 찾는 작업
            regx1 = regx.replace('0-9\.]+', ']+')
            regx1_hunted = regx_hunter(regx1, text)
            if regx1_hunted:
                target = regx1_hunted[-1].lower()  # 몇 ml입니까? 같은 문구가 있는 경우 ml가 target
            else:
                target = regx_hunted[-1].lower()  # 그런 부분이 없다면 맨 마지막으로 등장하 단위가 target

            ## 단일 단위의 숫자로 바꿔주는 작업
            unique = list(set([i.lower() for i in regx_hunted]))
            unique.remove(target)
            if len(unique) > 0:
                to_change = unique[0].lower()  # 두 개를 초과하는 단위는 안 등장한다고 가정
                regx2 = f'[0-9\.]+(?= ?{to_change})'
                regx2_comp = re.compile(regx2, re.I)
                if v.index(to_change) > v.index(target):  # 변환 대상이 되는 단위가 더 큰 경우 (e.g. 1l -> 1000ml)
                    if target in ['cm', 'mm', '센티미터', '밀리미터']:  # 100을 곱해줘야 하는 경우
                        text = regx2_comp.sub(multiple_100, text)
                    else:
                        text = regx2_comp.sub(multiple_1000, text)
                else:
                    if to_change in ['cm', 'mm', '센티미터', '밀리미터']:  # 100을 곱해줘야 하는 경우
                        text = regx2_comp.sub(div_100, text)
                    else:
                        text = regx2_comp.sub(div_1000, text)

                ## 단위 보정
                regx4 = '(?<=[0-9]) ?' + to_change + '(?=[^a-zA-Z])'
                regx4_comp = re.compile(regx4, re.I)
                text = regx4_comp.sub(target, text)

                ## 연달아 쓰인 경우를 보정하는 작업
                regx3 = f'[0-9\.]+ ?{target}[^a-zA-Z][0-9\.]+ ?{target}'
                regx3_hunted = regx_hunter(regx3, text)
                for j in regx3_hunted:
                    num_change = regx_hunter('[0-9\.]+', j)
                    num_res = sum([float(n) for n in num_change])
                    if int(num_res) == num_res:
                        text = text.replace(j, str(int(num_res)) + target)
                    else:
                        text = text.replace(j, str(num_res) + target)
            else:
                pass
        else:  # 한 가지 단위만 등장하는 경우 or 해당하는 단위가 등장하지 않는 경우
            pass
    return text


def geo_parser(text, obj):
    text_ori = copy.copy(text)
    for k, v in obj.items():
        if k in text_ori:
            for k2, v2 in v.items():
                if (k2 in text_ori) and (v2 not in text):
                    text = text + ' ' + v2
        else:
            pass
    return text


def animal_parser(text, obj):
    text_ori = copy.copy(text)
    for k, v in obj.items():
        if (' ' + k in text_ori) or (text_ori[:len(k)] == k):
            if ('다리' in text_ori) and ('마리' in text_ori) and ('몇 ' + k not in text_ori):
                text = text + ' ' + v
        else:
            pass
    return text


def time_parser(text):
    time_list = [['일', '시간', 24], ['시간', '분', 60], ['시', '분', 60], ['분', '초', 60]]
    for time in time_list:
        regx = f'[0-9]+{time[0]} ?[0-9]+{time[1]}'  # 두 가지 이상이 연달아 쓰인 경우
        regx_hunted = regx_hunter(regx, text)
        if regx_hunted:
            if (f'몇 {time[0]}' in text) and (
                    f'몇 {time[0]} 몇 {time[1]}' not in text):  # 더 큰 단위를 묻는 경우(몇 시간 몇 분 같이 복수 정답을 바라는 경우는 일단 낮은 단위로. 문제 수정 필요.)
                for i in regx_hunted:
                    num_hunted = regx_hunter('[0-9]+', i)
                    res = float(num_hunted[0]) + round(float(num_hunted[1]) / time[2], 2)
                    if res == int(res):
                        res = int(res)
                    text = text.replace(i, f'{str(res)}{time[0]}')
            else:  # 더 작은 단위를 묻거나 언급되지 않은 경우 그냥 작은 단위(
                for i in regx_hunted:
                    num_hunted = regx_hunter('[0-9]+', i)
                    res = float(num_hunted[1]) + round(float(num_hunted[0]) * time[2], 2)
                    if res == int(res):
                        res = int(res)
                    text = text.replace(i, f'{str(res)}{time[1]}')
                if time == ['시', '분', 60]:
                    num_hunted2 = regx_hunter('[0-9]+시', text)
                    for j in num_hunted2:
                        res2 = float(j[:-1]) * time[2]
                        if res2 == int(res2):
                            res2 = int(res2)
                        text = text.replace(j, f'{str(res2)}{time[1]}')

    return text


def time2_parser(text, obj):
    for k, v in obj.items():
        if k in text:
            text += ' ' + v
    return text


def work_parser(text, obj):
    delpattern = "[0-9|만](\s)?원"  # 일관련 문제중 몇만원이 관련된 문제는 전체 일의 양을 1로 놓을 필요가 없는 경우가 대부분입니다
    delpattern2 = "물의(\s)?양은"
    text_ori = copy.copy(text)
    for k, v in obj.items():
        if k in text_ori:
            if (re.findall(delpattern, string=text_ori) != []) or (re.findall(delpattern2, string=text_ori) != []):
                pass
            else:
                text = text + ' ' + v
        else:
            pass
    return text


def round_parser(text, obj):
    checkpattern = "빈틈없이"
    for k, v in obj.items():
        if k in text:
            if checkpattern in text:
                text += ' ' + v
            else:
                pass
    return text


def eq_solver(abc_eq):
    candidate = list()
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    num_alphabet = len(set(regx_hunter('[A-G]', str(abc_eq))))
    combinations = list(permutations(range(10), num_alphabet))
    for comb in combinations:
        question = abc_eq
        for n, alpha in enumerate(alphabet[:num_alphabet]):
            question = str(question).replace(alpha, str(comb[n]))
        question = eval(question)
        answer = list()
        for i in question:
            try:
                answer.append(eval(i))
            except SyntaxError:
                answer.append(False)
        if all(answer):
            candidate.append(comb)
    return f" 이 때 가능한 {', '.join(alphabet[:num_alphabet])} 쌍은 {str(candidate)[1:-1]} 이다."


def non_eq_parser(text, abc_non):
    result = text
    for abc in abc_non:
        res = abc
        found = regx_hunter('[A-Z]*[0-9]*[A-G]+[0-9]*', abc)
        for i in found:
            len_i = len(i) - 1
            res_temp = [f'{j}*{(10 ** (len_i - n))}' if n != len_i else j for n, j in enumerate(i)]
            res_temp = '(' + '+'.join(res_temp) + ')'
            res = res.replace(i, res_temp)
        result = result.replace(abc, res)
    return result


def abc_parser(text):
    found = regx_hunter('[0-9A-Z+\-*/=×÷]*[A-G]+[0-9A-Z+\-*/=×÷]*', text, lower=False)
    if found:
        found = eval(str(found).replace('×', '*').replace('÷', '/').replace('=', '=='))
        abc_eq = [i for i in found if ("=" in i) and (len(i) > 1)]
        if abc_eq:
            eq_solved = eq_solver(abc_eq)
        else:
            eq_solved = ''
        abc_non = [i for i in found if ("=" not in i) and (len(i) > 1)]
        if abc_non:
            text = non_eq_parser(text, abc_non)
        else:
            pass
        return text + eq_solved
    else:
        return text

def replacer(text):
    text = text.replace('몇 m 몇 cm', '몇 cm').replace('몇 l 몇 ml', '몇 ml').replace('몇 L 몇 mL', '몇 ml').replace(
        '몇 kg 몇 g', '몇 g').replace('몇 L 몇 ml', '몇 ml')
    text = text.replace('넷','4').replace('다섯','5').replace('여섯','6').replace('일곱','7').replace('여덟','8')\
        .replace('아홉','9')
    return text


def num_parse(text):
    while regx_hunter('[\d]+,[\d]+원', text):
        res = regx_hunter('[\d]+,[\d]+원', text)[0]
        text = text.replace(res,res.replace(',',''))
    return text


def parse(text, obj_path=Path(__file__).parents[0] / "obj.json"):
    obj_dict = load_json(obj_path)
    result = unit_parser(text, obj_dict['unit'])
    result = geo_parser(result, obj_dict['geometry'])
    result = replacer(result)
    result = num_parse(result)
    result = abc_parser(result)
    result = time_parser(result)
    result = time2_parser(result, obj_dict['time'])
    result = work_parser(result, obj_dict['work'])
    result = round_parser(result, obj_dict['round'])
    return result


if __name__ == '__main__':
    question = {
        "22": {
            "question": "50의 몇 퍼센트가 6인가?",
        }
    }
    print({k: parse(i["question"]) for k, i in question.items()})
    # texts = load_json(Path(__file__).parents[0] / 'data' / 'new.json')
    # result = dict()
    # for k, v in texts.items():
    #     text = v['문제'].replace('{', '').replace('}', '')
    #     result[k] = {'origin': text, 'parsed': parse(text)}
    # write_json(result, Path(__file__).parents[0] / 'data' / 'parsed.json')
