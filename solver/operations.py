from functools import partial
import sympy
import itertools
from solver.operations_function import *

OPERATIONS={
    '+':{'arity':2,'sep':True, 'sentence': '(%s)+(%s)','cal':(lambda x: x[0] + x[1])},
    '-':{'arity':2,'sep':True,'sentence':'(%s)-(%s)','cal':(lambda x: x[0]-x[1])},
    '*':{'arity':2,'sep':True,'sentence':'(%s)*(%s)','cal':(lambda x: x[0] * x[1])},
    '/':{'arity':2,'sep':True,'sentence':'(%s)/(%s)','cal':(lambda x: x[0]/x[1])},
    '**':{'arity':2,'sep':True,'sentence':'(%s)**(%s)','cal':(lambda x: x[0]**x[1])},
    '%':{'arity':2,'sep':True,'sentence':'(%s)%%(%s)','cal':(lambda x: x[0]%x[1])},
    '//':{'arity':2,'sep':True,'sentence':'(%s)//(%s)','cal':(lambda x: x[0]//x[1])},
    'map':{'arity':2,'sep':True,'sentence':'[(%s) for i in %s]','cal': lambda x: [x[1](i) for i in x[0]]},
    'filter':{'arity':2,'sep':False,'sentence':filter_entire},
    'arg':{'arity':0,'sep':True,'sentence':'','cal': None},
    'sum':{'arity':1,'sep':False,'sentence':sum_entire},
    'len':{'arity':1,'sep':False,'sentence':len_entire},
    'find':{'arity':2,'sep':True,'sentence':'(%s)[int(%s-0.5)]','cal':(lambda x:x[0][int(x[1]-0.5)])},
    'ord':{'arity':2,'sep':True,'sentence':'sorted(%s)[int(%s-0.5)]','cal':(lambda x:sorted(x[0])[int(x[1]-0.5)])},
    'comb':{'arity':2,'sep':False,'sentence':comb_entire},
    '=':{'arity':2,'sep':True,'sentence':'(%s)=(%s)','cal': (lambda x: sympy.Eq(x[0],x[1]))},
    '>':{'arity':2,'sep':True,'sentence':'(%s)>(%s)','cal':(lambda x: x[0] > x[1])},
    '>=':{'arity':2,'sep':True,'sentence':'(%s)>=(%s)','cal':(lambda x: x[0] >= x[1])},
    'perm':{'arity':2,'sep':False,'sentence':perm_entire},
    'round':{'arity':2,'sep':True,'sentence':'round(%s,%s)','cal':(lambda x: round(x[0],x[1]))},
    'range':{'arity':2,'sep':True,'sentence':'list(range(%s,%s+1))','cal': (lambda x: list(range(x[0],x[1]+1)))},
    'and':{'arity':2,'sep':True,'sentence':'(%s) and (%s)','cal':(lambda x: partial(cond_oper,oper='and',conditions=x))},
    'or':{'arity':2,'sep':True,'sentence':'(%s) or (%s)','cal':(lambda x: partial(cond_oper,oper='or',conditions=x))},
    'neg':{'arity':1,'sep':True,'sentence':'not (%s)','cal':(lambda x: partial(cond_oper,oper='neg',conditions=x))},
    'decimal':{'arity':1,'sep':False,'sentence':decimal_entire},
    'digit':{'arity':2,'sep':True, 'sentence': 'int(%s*(10**-%s))%%10', 'cal': (lambda x: int(x[0]*(10**-x[1]))%10)},
    'gcd':{'arity':1,'sep':True,'sentence': gcd_sentence, 'cal': gcd_cal},
    'lcm':{'arity':1,'sep':True,'sentence': lcm_sentence,'cal': lcm_cal},
    'avg':{'arity':1,'sep':False,'sentence': avg_entire},
    'LIST_eqn':{'arity':0, 'sep': True, 'sentence':'', 'cal': None}

}

COND_OP=['and','or','neg']
LIST_OP=['range','map','filter']
EQN_OP=['=','>=','>','LIST_eqn']
NUM_OP=['+','-','*','/','%','//']
UNK_OP=['sum','len','find','ord']


NEED_COND_OP=['filter']
NEED_LIST_OP=['comb','perm','sum','len','find','ord','gcd','lcm']


PRIMITIVES={
    'VAR':{},
    'TAR':{},
    'BEGIN':{},
    'END':{},
    'EQN':{},
    'COND':{},
    'NUM':{},
    'LIST':{},
}






__all__=['lcm_sentence','lcm_cal']