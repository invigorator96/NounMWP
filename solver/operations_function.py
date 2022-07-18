import math
import itertools
from collections.abc import Iterable
def cond_oper(x,oper,conditions):
    if oper=='and':
        return (conditions[0](x) and conditions[1](x))
    elif oper=='or':
        return (conditions[0](x) or conditions[1](x))
    else:
        return (not conditions[0](x))

def lcm_sentence(input):
    ret_history="lcm_ret=1\nfor i in %s:\n    lcm_ret=abs(lcm_ret*i)//math.gcd(lcm_ret,int(i))" % input
    ret_sol="lcm_ret"
    return ret_sol, [ret_history]
def lcm_cal(input):
    num_list=input[0]
    ret=1
    for i in num_list:
        ret=abs(ret*int(i))//math.gcd(ret,int(i))
    return ret

def gcd_sentence(input):
    ret_history="gcd_ret=0\nfor i in %s:\n    gcd_ret=math.gcd(gcd_ret,int(i))" % input
    ret_sol="gcd_ret"
    return ret_sol, [ret_history]
def gcd_cal(input):
    num_list=input[0]
    ret=0
    for i in num_list:
        ret=math.gcd(ret,int(i))
    return ret
def avg_sentence(input):
    ret_sol="sum(%s)/len(%s)" % (input, input)
    return ret_sol, []

def comb_entire(inputs):
    sentence_input, cal_input = inputs
    target_sol, num_sol =sentence_input
    target_value, num_value=cal_input

    if isinstance(target_value, int):
        ret_sol='math.factorial(%s)//(math.factorial(%s)*math.factorial(%s-%s))' % (target_sol, num_sol, target_sol, num_sol)
        ret_ans=math.factorial(target_value)//(math.factorial(num_value)*math.factorial(target_value-num_value))
        return ret_ans, ret_sol, []
    elif isinstance(target_value, Iterable):
        ret_sol= "list(set(itertools.combinations(%s,%s)))" % (target_sol, num_sol)
        ret_ans= list(set(itertools.combinations(target_value,num_value)))
        return ret_ans, ret_sol, []
    else:
        ret_sol='math.factorial(%s)//(math.factorial(%s)*math.factorial(%s-%s))' % (target_sol, num_sol, target_sol, num_sol)
        try:            
            ret_ans=math.factorial(target_value)//(math.factorial(num_value)*math.factorial(target_value-num_value))
        except:
            ret_ans=ret_sol
        return ret_ans, ret_sol, []

def sum_entire(inputs):
    sentence_input, cal_input = inputs
    target_sol =sentence_input[0]
    target_value=cal_input[0]

    ret_ans=sum(target_value)

    flag=False
    try:
        ret_ans=float(ret_ans)
        flag=True
    except:
        pass
    if flag:
        ret_sol='sum(%s)' % target_sol
    else:
        ret_sol=str(ret_ans)
    return ret_ans, ret_sol, []

def avg_entire(inputs):
    sentence_input, cal_input = inputs
    target_sol =sentence_input[0]
    target_value=cal_input[0]
    ret_ans=sum(target_value)/len(target_value)

    flag=False
    try:
        ret_ans=float(ret_ans)
        flag=True
    except:
        pass

    if flag:
        ret_sol=('sum(%s)' % target_sol )+ '/' +('len(%s)' % target_sol)
    else:
        ret_sol=str(ret_ans)
    return ret_ans, ret_sol, []

def perm_entire(inputs):
    sentence_input, cal_input = inputs
    target_sol, num_sol =sentence_input
    target_value, num_value=cal_input

    if isinstance(target_value, int):
        ret_sol='math.factorial(%s)//math.factorial(%s-%s)' % (target_sol, target_sol, num_sol)
        ret_ans=math.factorial(target_value)//math.factorial(target_value-num_value)
        return ret_ans, ret_sol, []
    elif isinstance(target_value, Iterable):
        ret_sol= "list(set(itertools.permutations(%s,%s)))" % (target_sol, num_sol)
        ret_ans= list(set(itertools.permutations(target_value,num_value)))
        return ret_ans, ret_sol, []
    else:
        ret_sol='math.factorial(%s)//math.factorial(%s-%s)' % (target_sol, target_sol, num_sol)
        try:            
            ret_ans=math.factorial(target_value)//math.factorial(target_value-num_value)
        except:
            ret_ans=ret_sol
        return ret_ans, ret_sol, []

def filter_entire(inputs):
    sentence_input, cal_input = inputs
    list_sol, cond_sol =sentence_input
    list_value, cond_func=cal_input
    Flag=False
    try:
        if isinstance(list_value[0][1],str):
            Flag=True
    except:
        pass

    if Flag:
        
        new_cond_sol=cond_sol.replace('i', 'i[0]')
        ret_sol= '[i for i in %s if (%s)]' % (list_sol, new_cond_sol)
        try:
            ret_ans=[i for i in list_value if cond_func(i[0])]
        except:
            ret_ans=ret_sol
        return ret_ans, ret_sol, []
    else:
        
        ret_sol='[i for i in %s if (%s)]' % (list_sol, cond_sol)
        try:
            ret_ans=[i for i in list_value if cond_func(i)]
        except:
            ret_ans=ret_sol
        return ret_ans, ret_sol, []

def decimal_entire(inputs):
    sentence_input, cal_input = inputs
    target_sol =sentence_input[0]
    target_value=cal_input[0]

    flag=False
    try:
        ret_ans=[int(''.join([str(int(i)) for i in x])) for x in target_value]
        flag=True
    except:
        ret_ans=int(''.join([str(int(i)) for i in target_value]))
    if flag:
        ret_sol="[int(''.join([str(int(i)) for i in x])) for x in %s]" % target_sol
    else:
        ret_sol="int(''.join([str(int(i)) for i in %s]))" % target_sol
    return ret_ans, ret_sol, []


def len_entire(inputs):
    sentence_input, cal_input = inputs
    target_sol =sentence_input[0]
    target_value=cal_input[0]

    if isinstance(target_value, int):
        ret_sol="1+int(math.log10(%s))" % target_sol
        ret_ans= 1+int(math.log10(target_value))
        return ret_ans, ret_sol, []
    elif isinstance(target_value, Iterable):
        ret_sol="len(%s)" % target_sol
        ret_ans=len(target_value)
        return ret_ans, ret_sol, []
    else:
        ret_sol="1+int(math.log10(%s))" % target_sol
        try:
            ret_ans= 1+int(math.log10(target_value))
        except:
            ret_ans=ret_sol
        return ret_ans, ret_sol, []

