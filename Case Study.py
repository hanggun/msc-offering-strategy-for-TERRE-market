#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:39:54 2020

@author: s1925253
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from terre_progress import terreModel
import timeit
import random
from scipy.stats import norm
seed = 123
np.random.seed(seed)
random.seed(seed)
#%% create curve
def createCurve(temp_q_up, temp_q_dw, price_DA, price_BM, t, n, h):
        
    value_up_dt = pd.DataFrame.from_dict(temp_q_up, orient='index')
    value_dw_dt = pd.DataFrame.from_dict(temp_q_dw, orient='index')
    
    up = np.zeros([t,n])
    down = np.zeros([t,n])
    
    j = -1
    for i in range(t*n):
        if i % n == 0:
            j += 1
        
        k = i % n
        up[j,k] = value_up_dt.values[i,0]
        down[j,k] = value_dw_dt.values[i,0]
        
    up = pd.DataFrame(up)
    down = pd.DataFrame(down)
    
    up_curve = [[price_DA[i,0],100] for i in range(t)]
    down_curve = [[0,price_DA[i,0]] for i in range(t)]
    
    up_quant = [[0,100] for i in range(t)]
    down_quant = [[0,100] for i in range(t)]
    
    for i in range(t):
        for j in range(n):
            
            index = i * n + j
            #compare day ahead market price and BM price
            if price_BM[index, 2] > price_DA[i,0]:
                #insert to up regulation
                
                k = len(up_curve[i])-1
                while k >= 0 :
                    
                    if price_BM[index, 2] >= up_curve[i][k-1] and price_BM[index, 2] <= up_curve[i][k]:
                        up_curve[i].insert(k, price_BM[index, 2])
                        up_quant[i].insert(k, up.iloc[i,j])
                        
                    k -= 1
                        
            if price_BM[index, 2] < price_DA[i,0]:
                #insert to down regulation
                
                k = len(down_curve[i]) - 1
                while k >= 0:
                    if price_BM[index, 2] >= down_curve[i][k-1] and price_BM[index, 2] <= down_curve[i][k]:
                        down_curve[i].insert(k, price_BM[index, 2])
                        down_quant[i].insert(k, down.iloc[i,j])
                    
                    k -= 1
                    
    for i in range(t):
        up_quant[i][-1] = up_quant[i][-2]
        up_quant[i].pop(0)
        
        down_quant[i].pop(0)
        if len(down_quant[i]) >= 2:
            down_quant[i][0] = down_quant[i][1]
        
        up_curve[i].insert(0, h+i)
        up_quant[i].insert(0, h+i)
        down_curve[i].insert(0, h+i)
        down_quant[i].insert(0, h+i)
    return up_curve, up_quant, down_curve, down_quant

#%% model start
revenue_day = np.ones([24, 30])
error = np.random.normal(3.31,14.67,[24,3*30])
pi_norm = norm.pdf(error, 3.31, 14.67)

price_data = pd.read_csv('UK Day Ahead Auction Prices.csv')
l = [[40] for i in range(30)]
d = [[30] for i in range(30)]

start = timeit.default_timer()
for idx_day in range(30):
    d_status = [30]
    l_status = [40]
    u_status = [0]
    up_curve_final = []
    down_curve_final = []
    up_quant_final = []
    down_quant_final = []
    revenue_hour = []
    for h in range(1, 23):
        bid_for_hour = h
        #%% dara generate
        t = 3 #number of blocks foreseen in future blocks
        n = 3 #number of scenarios
        
        # day ahead price 
        day_DA = price_data.iloc[:,idx_day]
        price_DA = np.ones([t,1])
        for i in range(t):
            index = i+bid_for_hour-1
            price_DA[i] = day_DA.iloc[index]
            
        price_DA_dt = pd.Series(
            data = price_DA.ravel(), 
            index = np.arange(1, t+1))
        
        price_DA_dt = price_DA_dt.to_dict()
        
        #balancing market price
        
        day_BM = np.ones([24,n])
        for i in range(24):
            for j in range(n):
                if day_DA.to_numpy()[i] <= 0:
                    # if error[i, idx_day*3 + j] <= 0:
                    day_BM[i,j] = error[i,idx_day*3 + j] + day_DA.to_numpy()[i]
                    # else:
                    #     day_BM[i,j] = -error[i,idx_day*3 + j] + day_DA.to_numpy()[i]
                else:
                    # if error[i, idx_day*3 + j] <= 0 and error[i,idx_day*3 + j] + day_DA.to_numpy()[i] <= 0:
                    #     day_BM[i,j] = -error[i,idx_day*3 + j] + day_DA.to_numpy()[i]
                    # else:
                    day_BM[i,j] = error[i,idx_day*3 + j] + day_DA.to_numpy()[i]
                
        price_BM = np.ones([t*n,n])
        for i in range(t):
            for j in range(n):
                
                index = i+bid_for_hour-1
                x = n*i+j
                price_BM[x,0] = i+1
                price_BM[x,1] = j+1
                price_BM[x,2] = day_BM[index,j]
        
        price_BM_dt = pd.DataFrame(
            data = price_BM, 
            columns = np.arange(1, n+1),
            index = np.arange(1, t*n+1))
        
        price_BM_dt = price_BM_dt.set_index([1,2]).squeeze().to_dict()
        
        #pi
        pi = np.ones([t*n,n])
        
        for i in range(t):
            for j in range(n):
                
                index = i * n + j
                
                pi[index, 0] = i+1
                pi[index, 1] = j+1
                pi[index, 2] = pi_norm[h,idx_day*3 + j] / sum(pi_norm[h,idx_day*3:idx_day*3+3])
                    
        pi_dt = pd.DataFrame(
            data = pi,
            columns = np.arange(1, n+1),
            index = np.arange(1, t*n+1)
            )
        
        pi_dt = pi_dt.set_index([1,2]).squeeze().to_dict()
        
        #day ahead quantity
        q_DA = np.ones([t,2])
        one_third = day_DA[day_DA > 0].quantile([1/3,2/3]).iloc[0]
        two_third = day_DA[day_DA > 0].quantile([1/3,2/3]).iloc[1]
        for i in range(t):
            index = i+bid_for_hour-1
            q_DA[i,0] = i+1
            if day_DA.iloc[index] <= 0:
                q_DA[i,1] = 0
            elif day_DA.iloc[index] <= one_third:
                q_DA[i,1] = 50
            elif day_DA.iloc[index] <= two_third:
                q_DA[i,1] = 60
            else :
                q_DA[i,1] = 70
                
            
        q_DA_dt = pd.DataFrame(
            data = q_DA, 
            columns = [1,2],
            index = np.arange(1, t+1))
        
        q_DA_dt = q_DA_dt.set_index([1]).squeeze().to_dict()
        
        #M_up and M_dw
        M_up = np.zeros([t*n*n, 4], dtype=int)
        M_dw = np.zeros([t*n*n, 4], dtype=int)
        index = -1
        index_dw = -1
        for i in range(t):
            for j in range(n):
                x = n*i + j
                y = 2
                for k in range(n):
                    index += 1
                    z = n*i + k
                    if price_BM[x,y] >= price_BM[z,y] and price_BM[x,y] > price_DA[i] and price_BM[x,y] > 0:
                        M_up[index, 0] = i+1
                        M_up[index, 1] = j+1
                        M_up[index, 2] = k+1
                        M_up[index, 3] = 1     
                    else:
                        M_up[index, 0] = i+1
                        M_up[index, 1] = j+1
                        M_up[index, 2] = k+1
                        M_up[index, 3] = 0
                        
        for i in range(t):
            for j in range(n):
                x = n*i + j
                y = 2
                for k in range(n):
                    index_dw += 1
                    z = 3*i + k
                    if price_BM[x,y] <= price_BM[z,y] and price_BM[x,y] < price_DA[i] and price_BM[x,y] > 0:
                        M_dw[index_dw, 0] = i+1
                        M_dw[index_dw, 1] = j+1
                        M_dw[index_dw, 2] = k+1
                        M_dw[index_dw, 3] = 1     
                    else:
                        M_dw[index_dw, 0] = i+1
                        M_dw[index_dw, 1] = j+1
                        M_dw[index_dw, 2] = k+1
                        M_dw[index_dw, 3] = 0
                        
                    
        M_up_dt = pd.DataFrame(
            data = M_up,
            columns = np.arange(1, 5),
            index = np.arange(1, t*n*n+1)
            )
        
        M_dw_dt = pd.DataFrame(
            data = M_dw,
            columns = np.arange(1, 5),
            index = np.arange(1, t*n*n+1)
            )
        
        M_up_dt = M_up_dt.set_index([1,2,3]).squeeze().to_dict()
        M_dw_dt = M_dw_dt.set_index([1,2,3]).squeeze().to_dict()
        
        temp_d, temp_l, temp_q_up, temp_q_dw, revenue, temp_u = terreModel(pi_dt, price_DA_dt, price_BM_dt, q_DA_dt, M_up_dt, M_dw_dt, t, n,
                                    d_status, l_status, h, u_status)
        
        up_curve, up_quant, down_curve, down_quant = createCurve(temp_q_up, temp_q_dw, price_DA, price_BM, t, n, h)
        
        d_status.append(temp_d)
        d[idx_day].append(temp_d)
        l_status.append(temp_l)
        l[idx_day].append(temp_l)
        revenue_hour.append(revenue[1])
        u_status.append(temp_u)
        
        up_curve_final.extend(k for k in up_curve)
        up_quant_final.extend(k for k in up_quant)
        down_curve_final.extend(k for k in down_curve)
        down_quant_final.extend(k for k in down_quant)
        
        up_curve_df = pd.DataFrame(
            data = up_curve_final,
            ).set_index([0]).squeeze()
        
        up_quant_df = pd.DataFrame(
            data = up_quant_final
            ).set_index([0]).squeeze()
        
        down_curve_df = pd.DataFrame(
            data = down_curve_final
            ).set_index([0]).squeeze()
        
        down_quant_df = pd.DataFrame(
            data = down_quant_final
            ).set_index([0]).squeeze()
        
        up_curve_df.to_csv('up_curve_df.csv')
        up_quant_df.to_csv('up_quant_df.csv')
    
    quantity_for_final_hour = [q_DA[-2,1],q_DA[-1,1]]
    C = 10
    for i in range(2):
        # quantity = quantity_for_final_hour[i]
        # revenue = (price_DA[i+1, 0] * quantity - C * quantity)
            
        revenue_hour.append(revenue[i+2])
        
    revenue_day[:,idx_day] = revenue_hour
    
stop = timeit.default_timer()
print('Time: ', stop - start)  
q = np.ones([24,30])
for j in range(30):
    day_DA = price_data.iloc[:,j]
    one_third = day_DA[day_DA > 0].quantile([1/3,2/3]).iloc[0]
    two_third = day_DA[day_DA > 0].quantile([1/3,2/3]).iloc[1]
    for i in range(24):
        if day_DA.iloc[i] <= 0:
            q[i,j] = 0
        elif day_DA.iloc[i] <= one_third:
            q[i,j] = 50
        elif day_DA.iloc[i] <= two_third:
            q[i,j] = 60
        else :
            q[i,j] = 70
                
def plot_price(idx_day, save=True):
    plt.figure(figsize=[8,6])
    day_DA = price_data.iloc[:,idx_day]
    day_BM = np.ones([24,n])
    for i in range(n):
        day_BM[:,i] = error[:,idx_day*3 + i] + day_DA.to_numpy()
        
    plt.plot(list(range(1,25)), day_DA,'o-', color='#3498db')
    plt.plot(list(range(1,25)), day_BM, marker='o', linestyle='',color="#e74c3c")
    plt.legend(['day-ahead market price', 'balancing market price scenario'], bbox_to_anchor=(0,1)
               , loc='lower left', ncol=2, fancybox=True)
    plt.xlabel('hour/h')
    plt.ylabel('price/£')
    plt.grid()
    plt.tight_layout()
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    if save:
        plt.savefig('day' + str(idx_day+1) +'.eps', format='eps')
    plt.show()
    
def plot_hourly(idx_day, save=True):
    day_DA = price_data.iloc[:,idx_day]
    plt.figure(figsize=[12, 6])
    if idx_day == 2:
        color = []
        for i in day_DA:
            if i <= 0:
                color.append('Negative day-ahead market price')
            else:
                color.append('Positive day-ahead market price')
        ax = sns.barplot(list(range(1,25)),revenue_day[:,idx_day],
                hue=color, dodge=False, palette=['#3498db', "#e74c3c"])
    else:
        color = []
        for i in revenue_day[:,idx_day]:
            if i >= 3000:
                color.append('High profit')
            elif i > 1000:
                color.append('Median profit')
            else:
                color.append('Low profit')
        ax = sns.barplot(list(range(1,25)),revenue_day[:,idx_day],
                hue=color, dodge=False, palette=['#3498db', '#E9FF33', "#e74c3c"])
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    
    ax.tick_params(bottom=False, left=False)
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    # plt.tight_layout()
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # if idx_day == 2:
    #     plt.ylim([-500,2000])
    plt.legend(loc='upper left')
    plt.xlabel('hour/h')
    plt.ylabel('profit/£')
    if save:
        plt.savefig('hourly' + str(idx_day+1) + '.eps', dpi=600, format='eps')
        
def plot_thermal(idx_day, save=True):
    plt.figure(figsize=[12,6])
    plt.plot(list(range(0,23)), d[idx_day], marker='o', linestyle='-',color='#3498db')
    plt.xlabel('hour/h')
    plt.ylabel('quantity/MWh')
    
    plt.grid()
    # plt.tight_layout()
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    if save:
        plt.savefig('thermal' + str(idx_day+1)+'.eps', dpi=600, format='eps')
        
def plot_storage(idx_day, save=True):
    plt.figure(figsize=[12,6])
    plt.plot(list(range(0,23)), l[idx_day], marker='o', linestyle='-',color='#3498db')
    plt.xlabel('hour/h')
    plt.ylabel('quantity/MWh')
    
    plt.grid()
    # plt.tight_layout()
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    if save:
        plt.savefig('storage' + str(idx_day+1) + '.eps', dpi=600, format='eps')
        
def plot_daily(save=True):
    x = sum(revenue_day)
    color = []
    for i in x:
        if i < 30000:
            color.append('Strong wind day')
        else:
            color.append('Common wind day')
            
    plt.figure(figsize=[12, 6])
    ax = sns.barplot(list(range(1,31)), x, dodge=False, hue=color, palette=['#3498db', "#e74c3c"])
    # plt.legend()
    plt.legend(bbox_to_anchor=(0,1)
               , loc='lower left', ncol=2, fancybox=True)
    plt.xlabel('day')
    plt.ylabel('profit/£')
    
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    
    ax.tick_params(bottom=False, left=False)
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.ylim([0,50000])
    if save:
        plt.savefig('daily.eps', dpi=600, format='eps')
        
def differences():
    plt.figure(figsize=[8,6])
    price_data = pd.read_csv('UK Day Ahead Auction Prices.csv')
    bm_data =pd.read_csv('10days_data.csv', header=None)
    
    for i in range(96):
        if (i+1) % 4 == 0:
            continue
        else:
            bm_data.drop([i], inplace=True)
    bm_data.reset_index(drop=True, inplace=True)
    difference = []
    for i in range(29,19, -1):
        difference.extend((price_data.iloc[:,i] - bm_data.iloc[:,29-i]).to_list())
        
    sns.kdeplot(difference)
    plt.xlabel('price deviation')
    plt.ylabel('density')
    plt.grid()
    plt.tight_layout()
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.savefig('difference.eps', format='eps')
