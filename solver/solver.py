import sympy
from solver.operations import OPERATIONS, COND_OP, LIST_OP,EQN_OP,NUM_OP,UNK_OP, NEED_LIST_OP,NEED_COND_OP
from copy import deepcopy, copy
from solver.utils import find_var, extract_coeff2, delete_repeated, unpack_ans, LISTtoLIST_eqn
from functools import partial
import math, itertools

class ProblemTree():
    def __init__(self,operator, operands):
        self.operator=operator
        self.operands=operands
        self.type=self.output_type()
        return None
    def output_type(self):
        if self.operator in COND_OP:
            return 'cond'
        if self.operator in LIST_OP:
            return 'list'
        if self.operator in EQN_OP:
            return 'eqn'
        is_eqn=False
        for oper in self.operands:
            if type(oper) is Equation:
                is_eqn=True
                break
        if is_eqn:
            return 'eqn'
        else:
            return 'num'
        

class Object:
    def __init__(self,value,sol='',history=[],name=None):
        self.name=name
        self.value=value
        self.sol=sol
        self.history=history
        if len(sol)==0:
            self.sol=str(value)
        self.problemtree=None
        return None
    def get_sol(self):
        return self.sol

class Answer(Object):
    def writeAnswer(self):
        hist=delete_repeated(self.history)
        hist_str='\n'.join(hist)
        if 'math.' in hist_str+self.sol:
            hist_str='import math\n'+hist_str
        if 'itertools.' in hist_str+self.sol:
            hist_str='import itertools\n'+hist_str
        ans_value, sol_str=unpack_ans(self.value, self.sol)

        try:
            ans="{:.2f}".format(float(ans_value))

            solution="print('{:.2f}'.format("+sol_str+"))"
        except (TypeError, ValueError):
            ans=ans_value
            solution="print("+sol_str+")"
        return ans, hist_str+'\n'+solution
    def execAnswer(self):
        exec(self.writeAnswer()[1],globals())
        return None

class Equation(Object):
    def __init__(self,eqn_sympy,sol='',history=[],name=None):
        super(Equation,self).__init__(value=eqn_sympy,sol=sol,history=history,name=name)
        self.value=eqn_sympy

        self.is_EQ= isinstance(self.value,sympy.core.relational.Eq)
        self.idx=None
        if self.is_EQ:
            self.general_form=str(self.value.lhs-self.value.rhs)


        else:
            self.general_form=None


    def as_condition(self):
        return Condition(cond=self.value,sol=self.sol.replace('=','==').replace('>=','>'),history=self.history)

    def substitute(self,var_dict, prob_num=None):
        if isinstance(self.value, str):
            op=self.problemtree.operator
            sub_operands=[]
            if prob_num is None:
                prob_num=self.idx
            for oper in self.problemtree.operands:
                if isinstance(oper,Equation):
                    sub_operands.append(oper.substitute(var_dict, prob_num=prob_num-1))
                else:
                    sub_operands.append(oper)
            return tree_solve(ProblemTree(op, sub_operands),prob_num)
        else:

            new_value=self.value.subs({key: v.value for key,v in var_dict.items()})
            sol=self.sol; new_hist=[]+self.history
            for key, value in var_dict.items():
                sol=sol.replace(str(key),'(%s)' % value.sol)
                new_hist=new_hist+value.history
            if new_value.is_Number or new_value==True:
                return Answer(new_value,sol,history=new_hist)
            return Equation(new_value,sol,history=new_hist)

    def __call__(self,input):
        if not (type(input) in (list, tuple)):
            input=[input]
        elif len(find_var(self.value))==1:
            input=[input]
        try:
            ret=self.value.subs(dict(zip(self.value.free_symbols,input)))
            return ret
        except AttributeError:
            var = find_var(self.value)
            final = self.value
            for old, new in zip(var, input):
                final=final.replace(old, str(new))
            ret = eval('\n'.join(self.history)+'\n'+final)

            return ret

    def replace_var(self,newvar='i'):
        variables=find_var(self.sol)
        if len(variables)==1:
            var=variables[0]
            return self.sol.replace(var,newvar)
        else:
            ret=copy(self.sol)
            for idx, oldvar in enumerate(variables):
                ret=ret.replace(oldvar, "%s[%s]" % (newvar, idx))
            return ret


class Condition(Equation):
    def __init__(self,cond,sol='',history=[]):
        self.value=cond
        self.sol=sol
        self.history=history
        if isinstance(cond,str):
            self.value=self.get_cond(cond,False)
        elif isinstance(cond,sympy.core.relational.Relational):
            self.value=self.get_cond(cond,True)
        if len(sol)==0:
            self.sol=str(cond)
        return None
    def __call__(self, x):
        return self.value(x)
    def get_cond(self,formula,sympy):
        if not sympy:
            variables=find_var(formula)
            if len(variables)==1:
                variable=variables[0]
                return lambda x: eval(formula.replace(variable, str(x)).replace('=','==').replace('>=','>'))
            else:
                def cond_multiple(x, formula, variables):
                    cond_str=copy(formula)
                    for var, num in zip(variables, x):
                        cond_str=cond_str.replace(var,str(num))
                    return cond_str.replace('=','==').replace('>=','>')
                return lambda x: eval(partial(cond_multiple, formula=formula, variables=variables)(x))
        else:
            variables=list(formula.free_symbols)
            if len(variables)>1:
                return lambda x: formula.subs(zip(variables,x))
            else:
                return lambda x: formula.subs(zip(variables, [x]))
    def get_sol(self,var='i'):
        condvars=find_var(self.sol)
        if len(condvars)==1:
            condvar=condvars[0]
            return str(self.sol).replace(condvar,var)
        else:
            ret=copy(str(self.sol))
            for idx, condvar in enumerate(condvars):
                ret=ret.replace(condvar, "%s[%d]" % (var, idx))
            return ret

def solve(expressions: list, verbose=False):

    try:
        return _solve(expressions, verbose=verbose)
    except:
        return _solve(LISTtoLIST_eqn(expressions), verbose=verbose)

def _solve(expressions: list, verbose=False)-> Answer:
    parse_expr,solved_trees, var_list,target_vars, solved_VARs=None, [],[],[],[] 

    delayed={}
    eqn_list=[]
    target_name_dict={}
    for R_idx, expression in enumerate(expressions):

        delayed_idx=list(delayed.keys())
        for i in delayed_idx:
            if ("R%s" % i in expression[1:]) and isinstance(solved_trees[i],str):
                if len(target_vars)==0:
                    target_vars=var_list
                if len(solved_VARs)==0:
                    solved_VARs=multiEQN_solve(eqn_list,var_list, verbose=verbose)
                new_history=[]; arg_sol=[]
                TAR_vars=[]

                for var in target_vars:
                    if var in solved_VARs:
                        ans=solved_VARs[var]
                        target=deepcopy(ans)
                        if target.name is None:
                            target.name=target_name_dict[str(var)]
                        TAR_vars.append(target)

                for tar in TAR_vars:
                    new_history.extend(tar.history)
                if delayed[i]=='arg':
                    arg_sol=["(%s,'%s')" % (t.sol, t.name) for t in TAR_vars]
                    arg_sol='['+','.join(arg_sol) +']'
                    arg_ans=[(t.value,t.name) for t in TAR_vars]
                    solved_tree=Answer(arg_ans,sol=arg_sol,history=new_history,name='arg')
                    solved_trees[i]=solved_tree
                elif delayed[i]=='LIST':
                    arg_sol=[t.sol for t in TAR_vars]
                    arg_sol='['+','.join(arg_sol) +']'
                    arg_ans=[t.value for t in TAR_vars]
                    solved_tree=Answer(arg_ans,sol=arg_sol,history=new_history,name='TARlist')
                    solved_trees[i]=solved_tree
                else:
                    raise NameError
                del delayed[i]



        if verbose:
            print('%s th tree solve start with operation'% R_idx, expression[0])
        if expression[0] in ['VAR','TAR']:
            parse_expr=Equation(sympy.Symbol(expression[2]),name=expression[1]) 
            var_list.append(parse_expr.value)
            solved_trees.append(parse_expr)
            if expression[0]=='TAR':
                target_vars.append(var_list[-1])
                target_name_dict[str(parse_expr.value)]=expression[1]

        elif expression[0] == 'END':
            return_idx=int(expression[1][1:])
            if type(solved_trees[return_idx]) is Answer:
                return solved_trees[return_idx]

            if len(solved_VARs)==0:
                solved_VARs=multiEQN_solve(eqn_list,var_list,verbose=verbose)

            out_expr=solved_trees[return_idx]

            gen_ans=out_expr.substitute(solved_VARs, prob_num=return_idx)

            del solved_VARs
            return Answer(gen_ans.value, sol=gen_ans.sol,history=gen_ans.history)


        elif expression[0] in ['LIST']:
            if expression[1] is not None:
                parse_expr= Object(expression[1])
                solved_trees.append(parse_expr)
            else:
                solved_trees.append('LIST')
                delayed[R_idx]='LIST'

        elif expression[0] == 'arg':
            if verbose:
                print('arg parsed!!')
            delayed[R_idx]=expression[0]
            solved_trees.append(expression[0])
        elif expression[0] == 'LIST_eqn':
            if len(target_vars)==0:
                target_vars=var_list
            if expression[1] is None:
                solved_trees.append(Equation(target_vars, sol=str(target_vars), history=[], name='TARlist_eqns'))
            else:
                listed_eqn=sympy.sympify(expression[1])
                if verbose:
                    print("LIST_eqn got:",listed_eqn, type(listed_eqn))
                for i in range(len(listed_eqn)):
                    found_Ri=str(listed_eqn[i])
                    if found_Ri in ['R%d' % i for i in range(R_idx)]:
                        tmp=int(found_Ri[1:])
                        listed_eqn[i]=solved_trees[tmp].value

                solved_trees.append(Equation(listed_eqn, sol=str(listed_eqn), history=[], name= "List_eqn"))
        else:
            operation=expression[0]
            operands=[]
            for operand in expression[1:]:
                if operand in ['R%d' % i for i in range(R_idx)]:
                    operands.append(solved_trees[int(operand[1:])])
                else:
                    operands.append(Object(operand))
            parse_expr=ProblemTree(operation,operands)
            if verbose:
                print('%sth operands' % R_idx, [oper.sol for oper in operands])
                print('operand type:', [type(oper.value) for oper in operands],[type(oper) for oper in operands])
            solved_tree=tree_solve(parse_expr,R_idx, verbose=verbose)
            solved_trees.append(solved_tree)

            if operation == '=':
                eqn_list.append(solved_tree)
            if operation in NEED_COND_OP or type(solved_tree) is Condition:
                for operand in operands:
                    try:
                        eqn_list.remove(operand)
                    except:
                        continue
    return solved_trees[-1]


def multiEQN_solve(eqn_list,var_list,solved_dict={},verbose=False):
    try:
        return _multiEQN_solve(eqn_list,var_list,solved_dict=solved_dict,verbose=verbose)
    except:
        return sympy_throw(eqn_list,var_list,solved_dict=solved_dict,verbose=verbose)
def sympy_throw(eqn_list,var_list,solved_dict={},verbose=False):
    sympy_eqns=sympy.sympify([eqn.value for eqn in eqn_list])
    answer_list=list(sympy.solve(sympy_eqns,var_list))[0]
    answer_list=[var.subs(dict(zip(var.free_symbols, [0 for _ in var.free_symbols]))) for var in answer_list]
    solved_now={}
    for i, var in enumerate(var_list):
        res=Answer(answer_list[i],name=var.name)
        res.sol=res.sol.replace('sqrt','math.sqrt')
        solved_now[var]=res
    ret={**solved_dict, **solved_now}
    return ret
def _multiEQN_solve(eqn_list,var_list,solved_dict={},verbose=False):

    if len(eqn_list)==0 or len(var_list)==0:
        return solved_dict
    if verbose:
        print("MultiEQN solving start", [eqn.value for eqn in eqn_list], solved_dict)

    eqn_list=[eqn.substitute(solved_dict) for eqn in eqn_list]
    eqn_list=[x for x in eqn_list if isinstance(x,Equation)]
    
    used_vars={0}
    for eqn in eqn_list:
        if verbose:
            print("found_var in multiEQNsolve:",find_var(eqn.value))
        for var in find_var(eqn.value):
            used_vars.add(str(var))


    if verbose:
        print('multiEQNsolve used_vars:',used_vars)
        print('multiEQNsolve var in used_vars:',[str(var) in used_vars for var in var_list])
        print('multiEQNsolve var_list:',var_list)

    next_eqns=[]; next_vars=copy(var_list); reduced=False

    
    for var in var_list:
        if not (str(var) in used_vars):
            solved_dict[var]=Answer(0,sol='0',history=[])
            next_vars.remove(var)
            if verbose:
                print("var_list: %s, removing:" % next_vars, var)
            reduced=True

    for eqn in eqn_list:
        if len(find_var(eqn.value))==1:
            if verbose:
                print("univariate eqn detected: %s" % str(eqn.sol))
            solved_var=list(eqn.value.free_symbols)[0]
            ans=singleEqn_solve(eqn,target_var=solved_var,verbose=verbose)

            next_vars.remove(solved_var)

            solved_dict[solved_var]=ans
            reduced=True
        else:
            next_eqns.append(eqn)
    if not reduced:
        solved_now=linsystem_solve(next_eqns,next_vars,verbose=verbose)
        ret={**solved_now, **solved_dict}
        del solved_dict
        return ret
    else:
        return _multiEQN_solve(next_eqns,next_vars,solved_dict=solved_dict,verbose=verbose)
    
def iseval(eqn: Equation, var):
    lhs, rhs= eqn.sol.split('=')
    if lhs[1:-1] == str(var):
        try:
            hist='\n'.join(eqn.history)
            exec('outans=None\n'+hist+'\n'+'ans='+rhs+"\noutans=ans", globals())

            out_ans=float(outans)
            return Answer(out_ans, sol=str(var), history=eqn.history+[eqn.sol])
        except:
            return False
    elif rhs[1:-1]==str(var):
        try:
            hist='\n'.join(eqn.history)
            exec('outans=None\n'+hist+'\n'+'ans='+lhs+"\noutans=ans", globals())

            out_ans=float(outans)
            return Answer(out_ans, sol=str(var), history=eqn.history+[('%s=%s' % (rhs, lhs))])
        except:
            return False
    else:
        return False


def istrivial(eqn: Equation, var, verbose=False):
    if verbose:
        print(eqn.value,var,type(var), (var in eqn.value.free_symbols), eqn.sol)


    try:
        isLinear= (sympy.degree(eqn.value,var)==1)
    except:
        return False, 'nonpoly'
    isEval=iseval(eqn, var)

    if verbose:
        print("Iseval(%s)=" % eqn.sol, var, isEval)
    if isEval is not False:
        return True,isEval
    elif isLinear:
        eqn_func='eqn_%s=lambda %s:' % (str(var),str(var)) + eqn.general_form
        coeff='coeff_%s=eqn_%s(1)-eqn_%s(0)' %(str(var),str(var),str(var))
        const='const_%s=eqn_%s(0)' % (str(var),str(var))
        sol_putting='%s= -const_%s/coeff_%s' %(str(var),str(var),str(var))
        ans=sympy.solve(eqn.value)[0]
        hist=eqn.history+[eqn_func,coeff,const,sol_putting]
        return True, Answer(ans,sol=str(var),history=hist,name=eqn.name)
    else:
        return False, 'nonlinear'

def linsystem_solve(eqn_list,var_list, verbose=False):
    if verbose:
        print("linear system solver start with %s equations and %s variables" % (len(eqn_list), len(var_list)))
    if len(eqn_list)==0:
        return {}
    if verbose:
        print("linsystem_solve_eqn: %s" % [(eqn.value, eqn.sol, type(eqn)) for eqn in eqn_list])


    system=[sympy.sympify(eqn.value) for eqn in eqn_list]
    A,b=sympy.linear_eq_to_matrix(system,var_list)
    if A.shape[0]==A.shape[1] and A.det()!=0:
        A_inv=get_inverse_mat(A)
        answer_list=list(A_inv*b)
        history=[]
        if A.shape[0]==2:

            eqn_sols = [eqn_list[0], eqn_list[1]]
            coef_str, C_str = extract_coeff2(eqn_sols,var_list)

            history = eqn_list[0].history + eqn_list[1].history
            DET = "(%s)*(%s)-(%s)*(%s)" % (coef_str[0], coef_str[3], coef_str[1], coef_str[2])
            sol_list=[]
            sol_list.append("((%s)*(%s)-(%s)*(%s))/(%s)" % (C_str[0], coef_str[3], C_str[1], coef_str[1], DET))
            sol_list.append("((%s)*(%s)-(%s)*(%s))/(%s)" % (coef_str[0], C_str[1], coef_str[2], C_str[0], DET))
        else:
            sol_list=['+'.join(['(%s)*(%s)' % (A_inv[j,i],b[i]) for i in range(A.shape[0])]) for j in range(A.shape[1])]
        Ans_list=[Answer(x, sol=sol_list[i], history=history) for i, x in enumerate(answer_list)]
        return dict(zip(var_list,Ans_list))
    else:
        answer_list=list(sympy.linsolve(A,b))[0]
        answer_list=[var.subs(dict(zip(var.free_symbols, [0 for _ in var.free_symbols]))) for var in answer_list]
        answer_list=[Answer(x,sol=str(x),history=[]) for i, x in enumerate(answer_list)]
        return dict(zip(var_list,answer_list))

def get_inverse_mat(A):
    return A.inv()

    

def singleEqn_solve(equation:Equation,target_var=None, verbose=False):
    if verbose:
        print("single equation solving:", equation, equation.value)

    if target_var is None:
        target_var=list(equation.value.free_symbols)[0]

    isLinear, ans= istrivial(equation,target_var,verbose=verbose)
    if isLinear:
        return ans
    else:
        ans,sol,newton_history=Newton_solver(equation,target_var)

    return Answer(ans,sol,equation.history+[newton_history])

def Newton_solver(eqn,var,init=None): 
    eqn_func='eqn=lambda %s:' % str(var) + eqn.general_form
    derivative=str(sympy.diff(eqn.general_form))
    deriv_func='derivative=lambda %s:' % str(var) + derivative



    if init is None:
        init=0.01

    while init < 10**3:
        try:
            newton_method='ans=%s\nfor _ in range(100):\n    newans=ans-eqn(ans)/derivative(ans)\n    if abs(newans-ans)<1e-5:\n        break\n    ans=newans' % init
            history=eqn_func+'\n'+deriv_func+'\n'+newton_method
            exec_ans='out_ans=None\n'+history+'\nout_ans=(ans)'

            sol='ans'
            exec(exec_ans, globals())
            return out_ans, sol, history
        except:
            init=init * 10
        

    
def tree_solve(expression: ProblemTree, R_idx: int,verbose=False):
    operation=OPERATIONS[expression.operator]
    operands=expression.operands
    assert operation['arity']==len(expression.operands)


    if expression.operator in NEED_COND_OP:
        if operands[1].value is None:
            operands[1]=TRIVIAL_COND

        elif not isinstance(operands[1],Condition):
            operands=expression.operands[0], expression.operands[1].as_condition()
    elif expression.type=='cond':
        operands=[]
        for cond in expression.operands:
            if not isinstance(cond,Condition):
                operands.append(cond.as_condition())
            else:
                operands.append(cond)
    new_history=[]
    for oper in expression.operands:
        new_history.extend(oper.history)


    if operation['sep']:
        oper_sentence=operation['sentence']

        if type(oper_sentence) is str:

            oper_sen_func= (lambda x: (oper_sentence % tuple(x), []))
        else:
            oper_sen_func=oper_sentence

        if expression.operator=='map':
            new_ans=operation['cal']([operands[0].value, operands[1].__call__])
            new_sol, add_history = oper_sen_func((operands[1].replace_var(newvar='i'),operands[0].get_sol()))
        else:
            new_sol, add_history = oper_sen_func(tuple(oper.get_sol() for oper in operands))
            
            try:
                new_ans=operation['cal']([oper.value for oper in operands])

            except:
                new_ans=new_sol
    else:
        oper_func=operation['sentence']
        sentence_input=tuple(oper.get_sol() for oper in operands)
        cal_input=[oper.value for oper in operands]
        new_ans, new_sol, add_history=oper_func([sentence_input, cal_input])
    new_history.extend(add_history)
    if isinstance(new_ans, str):
        try:
            import_str="import math\nimport itertools"
            new_ans=eval(import_str+'\n'.join(new_history)+'\n'+new_ans)
        except:
            new_ans=str(new_ans)
    

    if expression.type=='cond':
        ret=Condition(new_ans,sol=new_sol,history=new_history)
        ret.problemtree=expression
        return ret
    elif expression.type=='eqn':



        ret=Equation(new_ans, sol=new_sol,history=new_history)
        ret.idx=R_idx
        ret.problemtree=expression
        return ret
    else:



        new_history.append('R%s='% R_idx +new_sol )
        ret=Answer(new_ans,'R%s' % R_idx,new_history)
        ret.problemtree=expression
        return ret
        




TRIVIAL_COND=Condition(lambda x: True,sol='True',history=[])
