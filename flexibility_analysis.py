'''
This file contains a function that comducts a 2D sensitivity analysis on the optimization model over a range of flexibility levels for the juice and
ethanol splits and returns the expected profit for each case and gamma and theta values for each scenario in each case.

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
    This function runs the optimization model at varying flexibility levels for the juice and ethanol split and returns expected profit
    and values of theta annd gamma.


    Inputs: 
    
            flex_eth: weekly flexibility of ethanol split (fraction) in stage 2 decisions, bounds: (0,1)
            flex_jui: weekly flexibility of juice split (fraction) in stage 2 decisions, bounds: (0,1)
            premium: premium price paid for sustainable aviation fuel (SAF), Units: $R/m3
            market_prices: dictionary of market prices indexed by product and scenarios (weeks), Units: R$/m3 or R$/tonne
            scenarios: list of time scenarios (weeks) for market prices

    Returns:

            profit_df: Data frame storing the expected value of the profit for each flexibility level combination
            theta_df: Data frame storing an array of theta values for each flexibility level combination
            gamma_df: Data frame storing an array of gamma values for each flexibility level combination
    '''

    #Create an empty matrix to store profit values
    prof = np.zeros((11,11))

    #Create empty dictionaries to store theta and gamma values
    theta = {}
    gamma = {}

    #Loop through flexibility levels
    for i in range(len(flex_jui)):
        for j in range(len(flex_eth)):
            #Specify the fixed input data for the model
            min_eth_market = 0.2 
            min_eth = 0.4
            min_sug = 0.4
            min_saf = 0.4
            jet_energy = 0.42 #MWh/m3
            d_price = 3200 #R$/m3
            g_price = 2600 #R$/m3

            #Create an instance of the optimization model
            m = create_stochastic_model_v2(premium, market_prices, scenarios, min_eth_market, min_eth, min_saf, min_sug, flex_jui[i], flex_eth[j], jet_energy, d_price, g_price)
            
            #Update cost and conversion parameters for the ATJ technology of interest
            m.jet_conv = conv
            m.jet_pc = cost
            
            #Solve the model
            sol =pyo.SolverFactory('gurobi', tee=True)
            sol.solve(m)

            #Record the profit
            prof[i,j] = pyo.value(m.obj)

            #Create empty arrays to store ethanol and juice split flowrates
            j1 = []
            j2 = []
            e1 = []
            e2 = []

            #Record ethanol and juice split flows at each scenario
            for k in scenarios:
                j1.append(pyo.value(m.x['j1',k]))
                j2.append(pyo.value(m.x['j2',k]))
                e1.append(pyo.value(m.x['e1',k]))
                e2.append(pyo.value(m.x['e2',k]))

            #Calculate Theta and Gamma at each scenario
            theta[i,j] = np.array(j1)/(np.array(j1) + np.array(j2))
            gamma[i,j] = np.array(e1)/(np.array(e1) + np.array(e2))

    #Convert to Data frames
    profit_df = pd.DataFrame(prof)
    theta_df = pd.DataFrame.from_dict(theta)
    gamma_df = pd.DataFrame.from_dict(gamma)

    return profit_df, theta_df, gamma_df