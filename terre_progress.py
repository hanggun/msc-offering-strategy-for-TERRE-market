#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:42:24 2020

@author: s1925253
"""

from pyomo.environ import *
import pandas as pd
import numpy as np

def terreModel(pi_dt, price_DA_dt, price_BM_dt, q_DA_dt, M_up_dt, M_dw_dt, t, n,
               d_status, l_status, h, u_status):
    model = AbstractModel()
    #%%initialize index
    D_min = 0
    D_max = 70
    R_up = 30
    R_dw = 30
    L_max = 80
    L_min = 0
    P_up = 30
    P_dw = 30
    C0 = 0
    C = 10
    C_st = 200
    C_sh = 200
    eta = 1
    model.t = RangeSet(1, t) 
    model.k = RangeSet(0, t)
    model.n = RangeSet(1, n)
    model.zero = RangeSet(0,0)
    model.one = RangeSet(1,1)
    
    
    #%% initialize parameters
    model.pi = Param(model.t, model.n, initialize=pi_dt)
    model.price_DA = Param(model.t, initialize=price_DA_dt)
    model.price_BM = Param(model.t, model.n, initialize=price_BM_dt)
    model.q_DA = Param(model.t, initialize=q_DA_dt)
    model.M_up = Param(model.t, model.n, model.n, initialize=M_up_dt)
    model.M_dw = Param(model.t, model.n, model.n, initialize=M_dw_dt)
    
    #%% initialize variables
    model.q_up = Var(model.t, model.n, domain=NonNegativeReals)
    model.q_dw = Var(model.t, model.n, domain=NonNegativeReals)
    model.c = Var(model.t, model.n, domain=NonNegativeReals)
    model.d = Var(model.t, model.n, domain=NonNegativeReals, initialize=0)
    model.expected_d = Var(model.k, domain=NonNegativeReals, initialize=0)
    model.expected_l = Var(model.k, domain=NonNegativeReals, initialize=0)
    model.p_up = Var(model.t, model.n, domain=NonNegativeReals)
    model.p_dw = Var(model.t, model.n, domain=NonNegativeReals)
    model.l = Var(model.t, model.n, domain=NonNegativeReals, initialize=0)
    model.u = Var(model.k, model.n, domain=Boolean, initialize=0)
    model.y = Var(model.t, model.n, domain=Boolean)
    model.z = Var(model.t, model.n, domain=Boolean)
    model.o_up = Var(model.t, model.n, domain=NonNegativeReals)
    model.o_dw = Var(model.t, model.n, domain=NonNegativeReals)
    model.revenue = Var(model.t)
    
    #%%load data
    # data.load(filename='pi.csv', param=model.pi, index= (model.t, model.n))
    # data.load(filename='price_DA3.csv', param=model.price_DA, index= (model.t))
    # data.load(filename='price_BM3.csv', param=model.price_BM, index= (model.t, model.n))
    # data.load(filename='q_DA3.csv', param=model.q_DA, index=model.t)
    # data.load(filename='M_up3.csv', param=model.M_up, index=(model.t,model.n,model.n))
    # data.load(filename='M_dw3.csv', param=model.M_dw, index=(model.t,model.n,model.n))
    
    #%%constraints
    def ObjRule(model):
        return sum(model.pi[t,n] * (model.price_DA[t]*model.q_DA[t] + model.price_BM[t,n] * 
                                    (model.q_up[t,n]-model.q_dw[t,n]) - model.c[t,n]) 
                   for t in model.t for n in model. n)
    
    def con1(model, t, n):
        return model.q_DA[t] + model.q_up[t,n] - model.q_dw[t,n] == \
            model.d[t,n] - model.p_up[t,n] + model.p_dw[t,n]
            
    def con2(model, t, n):
        return model.c[t,n] == C0 * model.u[t,n] + C * model.d[t,n] + \
            C_st * model.y[t,n] + C_sh * model.z[t,n]
            
    
    #%% thermal unit constraint
    def con3(model, t, n):
        return model.d[t,n] >= model.u[t,n] * D_min
        
    def con4(model, t, n):
        return model.d[t,n] <= model.u[t,n] * D_max
        
    def con5(model, t, n):
        return model.d[t,n] - model.expected_d[t-1] <= R_up
        
    def con6(model, t, n):
        return model.expected_d[t-1] - model.d[t,n] <= R_dw
        
    def con7(model, t, n):
        return model.u[t, n] - model.u[t-1, n] <= model.y[t,n]
        
    def con8(model, t, n):
        return model.u[t-1, n] - model.u[t, n] <= model.z[t,n]
        
    #%% storage unit constraint
    def con9(model, t, n):
        return model.l[t,n] == model.expected_l[t-1] + eta*model.p_up[t,n] - model.p_dw[t,n]
        
    def con10(model, t, n):
        return model.l[t,n] <= L_max
        
    def con11(model, t, n):
        return model.l[t,n] >= L_min
        
    def con12(model, t, n):
        return model.p_up[t,n] <= P_up
        
    def con13(model, t, n):
        return model.p_dw[t,n] <= P_dw
    
    #%%initialization
    # def con14(model, t, n):
    #     return model.l[t,n] == 40
    
    # def con15(model, t, n):
    #     return model.d[t,n] == 0
        
    # def con16(model, t, n):
    #     return model.y[t,n] == 0
    
    # def con17(model, t, n):
    #     return model.z[t,n] == 0
    
    def con18(model, t, n):
        return model.u[t,n] == u_status[h-1]
    
    def con19(model, t, n):
        return model.q_up[t,n] == sum(model.M_up[t,n,n1] * model.o_up[t,n1] for n1 in model.n)
    
    def con20(model, t, n):
        return model.q_dw[t,n] == sum(model.M_dw[t,n,n1] * model.o_dw[t,n1] for n1 in model.n)
    
    def con21(model, t, n):
        if model.price_BM[t,n] <= model.price_DA[t]:
            return model.o_up[t,n] == 0
        else:
            return model.o_up[t,n] >= 0
    
    def con22(model, t, n):
        if model.price_BM[t,n] >= model.price_DA[t]:
            return model.o_dw[t,n] == 0
        else:
            return model.o_dw[t,n] >= 0
        
    def con23(model, t):
        return model.revenue[t] == sum(model.pi[t,n] * (model.price_DA[t]*model.q_DA[t] + model.price_BM[t,n] * 
                                    (model.q_up[t,n]-model.q_dw[t,n]) - model.c[t,n]) 
                    for n in model. n)
    
    #%%
    #use expected thermal generation level and storage level 
    def con24(model, t):
        return model.expected_d[t] == sum(model.pi[t,n] * model.d[t,n] for n in model.n)
    
    def con25(model, t):
        return model.expected_l[t] == sum(model.pi[t,n] * model.l[t,n] for n in model.n)
    
    def con26(model, t):
        return model.expected_d[t] == d_status[h-1]
    
    def con27(model, t):
        return model.expected_l[t] == l_status[h-1]
    
    # def con28(model, t, n):
    #     if model.q_DA[t] == 0:
    #         return model.q_up[t,n] == 0
    #     else:
    #         return model.q_up[t,n] >= 0
        
    # def con29(model, t, n):
    #     if model.q_DA[t] == 0:
    #         return model.q_dw[t,n] == 0
    #     else:
    #         return model.q_dw[t,n] >= 0
    
    #objective function
    model.profit = Objective(rule=ObjRule, sense=maximize)
    
    #constraint
    model.con1 = Constraint(model.t, model.n, rule=con1)
    model.con2 = Constraint(model.t, model.n, rule=con2)
    model.con3 = Constraint(model.t, model.n, rule=con3)
    model.con4 = Constraint(model.t, model.n, rule=con4)
    model.con5 = Constraint(model.t, model.n, rule=con5)
    model.con6 = Constraint(model.t, model.n, rule=con6)
    model.con7 = Constraint(model.t, model.n, rule=con7)
    model.con8 = Constraint(model.t, model.n, rule=con8)
    model.con9 = Constraint(model.t, model.n, rule=con9)
    model.con10 = Constraint(model.t, model.n, rule=con10)
    model.con11 = Constraint(model.t, model.n, rule=con11)
    model.con12 = Constraint(model.t, model.n, rule=con12)
    model.con13 = Constraint(model.t, model.n, rule=con13)
    # model.con14 = Constraint(model.zero, model.n, rule=con14)
    # model.con15 = Constraint(model.zero, model.n, rule=con15)
    # model.con16 = Constraint(model.zero, model.n, rule=con16)
    # model.con17 = Constraint(model.zero, model.n, rule=con17)
    model.con18 = Constraint(model.zero, model.n, rule=con18)
    model.con19 = Constraint(model.t, model.n, rule=con19)
    model.con20 = Constraint(model.t, model.n, rule=con20)
    model.con21 = Constraint(model.t, model.n, rule=con21)
    model.con22 = Constraint(model.t, model.n, rule=con22)
    model.con23 = Constraint(model.t, rule=con23)
    model.con24 = Constraint(model.t, rule=con24)
    model.con25 = Constraint(model.t, rule=con25)
    model.con26 = Constraint(model.zero, rule=con26)
    model.con27 = Constraint(model.zero, rule=con27)
    # model.con28 = Constraint(model.t, model.n, rule=con28)
    # model.con29 = Constraint(model.t, model.n, rule=con29)
    instance = model.create_instance()
    results = SolverFactory('glpk').solve(instance)
    
    # price = [instance.price_BM[1,1]*instance.pi[1,1], instance.price_BM[1,2] * instance.pi[1,2],
    #          instance.price_BM[1,3]*instance.pi[1,3]]
    price = [instance.pi[1,1], instance.pi[1,2],
             instance.pi[1,3]]
    
    idx = sorted(range(len(price)), key=price.__getitem__)[2]+1
    return instance.d.get_values()[1,idx], instance.l.get_values()[1,idx],\
        instance.q_up.get_values(), instance.q_dw.get_values(),\
            instance.revenue.get_values(), instance.u.get_values()[1,idx]