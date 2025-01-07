'''
This file contains a function that conducts a 2D sensitivity analysis on the optimization model over a range of SAF conversion factors and OPEX values
and returns the fraction of scenarios that maximize bio-jet fuel for each case.

Created by Madelynn Watson and the University of Notre Dame
'''

#Import Necessary Packages
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from stochastic_model_v2 import *
import idaes
import pandas as pd

def sensitivity2d_data_gen(conv,cost,premium,market_prices,scenarios, gasconv = 0, dconv=0):

    '''
    This function runs the optimization model at varying SAF conversions and OPEX values and returns the fraction of scenarios that maximize SAF.


    Inputs: 
    
            conv: Conversion factor for ethanol to jet fuel, Units: m3 jet/m3 eth
            cost: Production cost for upgrading ethanol to jet fuel, Units: $R/m3 jet
            premium: premium price paid for sustainable aviation fuel (SAF), Units: $R/m3
            market_prices: dictionary of market prices indexed by product and scenarios (weeks), Units: $R/m3
            scenarios: list of time scenarios (weeks) for market prices
            gasconv: Conversion factor for ethanol to gasoline, Units: m3 gas/m3 eth, Default: 0 
            dconv: Conversion factor for ethanol to diesel, Units: m3 diesel/m3 eth, Default: 0

    Returns:

            gamma_frac_df: Data frame storing the fraction of scenarios that maximize SAF at every cost and conversion combination
    '''
    
    #Create and empty matrix to store the fraction of scenarios that maximize SAF
    gamma_frac = np.zeros((11,11))
    
    #Create the optimization model with full flexibility (equivilent to solving one scenario at a time)
    m = create_stochastic_model_v2(premium, market_prices, scenarios, 0.2, 0.4, 0.4, 0.4, 1, 1,0.42,0,0)
    
    m.gas_conv = gasconv
    m.diesel_conv = dconv

    # Loop through conversion and opex ranges
    
    for i in range(len(conv)):
        for j in range(len(cost)):
                #Update model parameters for jet fuel conversion and cost
                m.jet_conv = conv[i]
                m.jet_pc = cost[j]
                #Solve the optimization model
                sol =pyo.SolverFactory('gurobi', tee=True)
                sol.solve(m)
                #Sort through results and collect scenarios that maximize SAF (gamma < 0.6))
                #create empty array to count scenarios that maximize saf
                gamma_saf = []
                #initialize gamma
                gamma = 0
                #loop through scenarios
                for k in scenarios:
                        gamma = pyo.value(m.x['e1',k])/ (pyo.value(m.x['e1',k]) + pyo.value(m.x['e2',k]))
                        if gamma < 0.6:
                                #Count all scenarios that maximize SAF
                                gamma_saf.append(gamma)
                #Calculate the fraction of total scenarios that maximize SAF
                gamma_frac[i,j] = len(gamma_saf)/len(scenarios)
    gamma_frac_df = pd.DataFrame.from_dict(gamma_frac)
        
    return gamma_frac_df