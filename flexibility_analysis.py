'''
This file contains a function that runs the optimization model one scenario at a time and returns the frequency of scenarios in each operating mode

Created by Madelynn Watson and the University of Notre Dame
'''

#Import Necessary Packages
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from stochastic_model_v2 import *
import idaes
import pandas as pd

def flexibility_data_gen(flex_eth, flex_jui,cost,conv,premium,market_prices,scenarios):

    '''
    This function runs the optimization model one scenario at a time for 527 total scenarios and returns the frequency of scenarios in each operating mode.


    Inputs: 
    
            flex_eth: weekly flexibility of ethanol split (fraction) in stage 2 decisions, bounds: (0,1)
            flex_jui: weekly flexibility of juice split (fraction) in stage 2 decisions, bounds: (0,1)
            premium: premium price paid for sustainable aviation fuel (SAF), bounds: (0,infinity), Units: $R/m3
            market_prices: dictionary of market prices indexed by product and scenarios (weeks)
            scenarios: list of time scenarios (weeks) for market prices

    Returns:

            sorted_dat
    '''

    prof = np.zeros((11,11))
    theta = {}
    gamma = {}

    for i in range(len(flex_jui)):
        print(i)
        for j in range(len(flex_eth)):
            print(j)
            #Create an instance of the optimization model
            m = create_stochastic_model_v2(premium, market_prices, scenarios, 0.2, 0.4, 0.4, 0.4, flex_jui[i], flex_eth[j], 0.42, 3200, 2600)
            
            m.jet_conv = conv
            m.jet_pc = cost
            
            #Solve the model
            sol =pyo.SolverFactory('gurobi', tee=True)
            sol.solve(m)
            #record the profit
            prof[i,j] = pyo.value(m.obj)

            j1 = []
            j2 = []
            e1 = []
            e2 = []
            #record ethanol and juice split flows at each scenario
            for k in scenarios:
                j1.append(pyo.value(m.x['j1',k]))
                j2.append(pyo.value(m.x['j2',k]))
                e1.append(pyo.value(m.x['e1',k]))
                e2.append(pyo.value(m.x['e2',k]))

            #Calculate Theta and Gamma at each scenario
            theta[i,j] = np.array(j1)/(np.array(j1) + np.array(j2))
            gamma[i,j] = np.array(e1)/(np.array(e1) + np.array(e2))

    profit_df = pd.DataFrame(prof)
    theta_df = pd.DataFrame.from_dict(theta)
    gamma_df = pd.DataFrame.from_dict(gamma)

    return profit_df, theta_df, gamma_df