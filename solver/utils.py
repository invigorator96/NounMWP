import sympy

def find_var(formula):
    if isinstance(formula, str):
        ret = []
        if formula.find('CONDVAR') != -1:
            ret.append('CONDVAR')
        for i in range(10):
            if formula.find('X' + str(i)) != -1:
                ret.append('X' + str(i))
        return ret
    else:
        try:
            return formula.free_symbols
        except:
            return []

def extract_coeff2(eqns,vars):
    
    eqn_sols=[eqn.sol for eqn in eqns]
    assert len(eqn_sols) == 2
    coef_str = []
    cnst_str = []

    for i in range(2):
        meat = eqn_sols[i]
        lhs, rhs = meat.split("=")
        lhs = sympy.parsing.sympy_parser.parse_expr(lhs)
        rhs = sympy.parsing.sympy_parser.parse_expr(rhs)
        LC0, LC1, LD = lhs.coeff(vars[0]), lhs.coeff(vars[1]), lhs.coeff(vars[0], n=0).coeff(vars[1], n=0)
        RC0, RC1, RD = rhs.coeff(vars[0]), rhs.coeff(vars[1]), rhs.coeff(vars[0], n=0).coeff(vars[1], n=0)

        coef_str.extend(["("+str(LC0)+")-("+str(RC0)+")", "("+str(LC1)+")-("+str(RC1)+")"])
        cnst_str.append("("+str(RD)+")-("+str(LD)+")")
    return coef_str, cnst_str

def delete_repeated(history: list) -> list:
    ret=[]
    for hist in history:
        if not (hist in ret):
            ret.append(hist)
    return ret

def unpack_ans(value, sol):
    if type(value) in [list,tuple, range]:
        new_value=value[-1]
        new_sol=sol+'[-1]'
        return unpack_ans(new_value, new_sol)
    else:
        return value, sol

def LISTtoLIST_eqn(expressions):
    ret=[]
    for expr in expressions:
        if expr[0]=='LIST' and expr[1]== None:
            ret.append(('LIST_eqn', None))
        else:
            ret.append(expr)
    return ret
