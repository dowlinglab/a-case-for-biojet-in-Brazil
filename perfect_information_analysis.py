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

def perfect_information_data_gen(conv,cost,premium,market_prices,scenarios, gasconv = 0.07, dconv=0.112):

    '''
    This function runs the optimization model one scenario at a time for 527 total scenarios and returns the frequency of scenarios in each operating mode.


    Inputs: 
    
            conv: Conversion factor for ethanol to jet fuel, Units: m3 jet/m3 eth
            cost: Production cost for upgrading ethanol to jet fuel, Units: $R/m3 jet
            premium: premium price paid for sustainable aviation fuel (SAF), bounds: (0,infinity), Units: $R/m3
            market_prices: dictionary of market prices indexed by product and scenarios (weeks)
            scenarios: list of time scenarios (weeks) for market prices
            gasconv: Conversion factor for ethanol to gasoline, Units: m3 gas/m3 eth, Default: 0.07 (conventional)
            dconv: Conversion factor for ethanol to diesel, Units: m3 diesel/m3 eth, Default: .112 (conventional)

    Returns:

            sorted_data1_df: Frequency of scenarios in operating mode 1
            sorted_data2_df: Frequency of scenarios in operating mode 2
            sorted_data3_df: Frequency of scenarios in operating mode 3
            sorted_data4_df: Frequency of scenarios in operating mode 4
    '''

    #Create an empty dictionary to store all results
    data_dict = {}
    #Create empty arrays to store the split flow rates and objective (profit)
    juice_to_dist = []
    juice_to_fact = []
    eth_to_market = []
    eth_to_SAF = []
    profit = []
    
    #Run the model at each scenario and save the split flow rates and objective (profit) (527 instances)
    #Specify the fixed input data for the model
    min_eth_market = 0.2 
    min_eth = 0.4
    min_sug = 0.4
    min_saf = 0.4
    juice_flex = 1
    eth_flex = 1
    jet_energy = 0.42 #MWh/m3
    d_price = 3200 #R$/m3
    g_price = 2600 #R$/m3

    #Loop through scenarios
    for i in scenarios:
        #Create the model
        m = create_stochastic_model_v2(premium, market_prices, [i], min_eth_market, min_eth, min_saf, min_sug, juice_flex, eth_flex, jet_energy, d_price, g_price)
        #Update the parameters that change for the conventional and optimistic cases
        m.jet_conv = conv
        m.jet_pc = cost
        m.gas_conv = gasconv
        m.diesel_conv = dconv
        #Solve
        sol =pyo.SolverFactory('gurobi', tee=True)
        sol.solve(m)
        #Store results
        juice_to_dist.append(pyo.value(m.x['j2',i])) #Amount of juice sent to distillary for each scenario
        juice_to_fact.append(pyo.value(m.x['j1',i])) #Amount of juice sent to sugar factory for each scenario
        eth_to_market.append(pyo.value(m.x['e1',i])) #Amount of ethanol sent to market for each scenario
        eth_to_SAF.append(pyo.value(m.x['e2',i])) #Amount of ethanol sent to upgrade to SAF for each scenario
        profit.append(pyo.value(m.obj)) #Profit

    #Organize results in dictionaries 
    data_dict['juice to dist'] = juice_to_dist
    data_dict['juice to fact'] = juice_to_fact
    data_dict['eth to market'] = eth_to_market
    data_dict['eth to saf'] = eth_to_SAF
    data_dict['profit'] = profit

    #Calculate theta and Gamma
    theta = np.array(juice_to_fact)/(np.array(juice_to_fact) + np.array(juice_to_dist))
    gamma = np.array(eth_to_market)/(np.array(eth_to_market) + np.array(eth_to_SAF))

    data_dict['theta'] = theta
    data_dict['gamma'] = gamma

    #Create empty dict to store results sorted by operating mode
    sorted_data_dict1 = {}
    sorted_data_dict2 = {}
    sorted_data_dict3 = {}
    sorted_data_dict4 = {}

    #Create empty arrays to store individual region results
    region1_profit = []
    region2_profit = []
    region3_profit = []
    region4_profit = []
    region1_scenario = []
    region2_scenario = []
    region3_scenario = []
    region4_scenario = []
    region1_ratio1 = []
    region1_ratio2 = []
    region2_ratio1 = []
    region2_ratio2 = []
    region3_ratio1 = []
    region3_ratio2 = []
    region4_ratio1 = []
    region4_ratio2 = []


    #Sort data by operating mode rules, calculate ratios defined in main text
    for i in range(len(theta)):
        #Max sugar, max ethanol to market
        if theta[i] > 0.6 and gamma[i] > 0.5:
            region1_profit.append(profit[i])
            region1_scenario.append(i)
            region1_ratio1.append(((market_prices['a'][i] + premium - pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv) - pyo.value(m.mu_el)*pyo.value(m.jet_energy))/market_prices['e'][i])
            region1_ratio2.append(pyo.value(m.alpha)*(market_prices['e'][i]*gamma[i] - pyo.value(m.eth_pc) + (1-gamma[i])*((market_prices['a'][i] + premium -pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv)))/((market_prices['s'][i]-pyo.value(m.sug_pc))*pyo.value(m.alpha) + pyo.value(m.mu_el)* (169704 - pyo.value(m.jet_energy)*(1-gamma[i])*(pyo.value(m.alpha)/-1.64))))
        
        #Max sugar, max SAF
        elif theta[i] > 0.6 and gamma[i] < 0.4:
            region2_profit.append(profit[i])
            region2_scenario.append(i)
            region2_ratio1.append(((market_prices['a'][i] + premium - pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv) - pyo.value(m.mu_el)*pyo.value(m.jet_energy))/market_prices['e'][i])
            region2_ratio2.append(pyo.value(m.alpha)*(market_prices['e'][i]*gamma[i] - pyo.value(m.eth_pc) + (1-gamma[i])*((market_prices['a'][i] + premium -pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv)))/((market_prices['s'][i]-pyo.value(m.sug_pc))*pyo.value(m.alpha) + pyo.value(m.mu_el)* (169704 - pyo.value(m.jet_energy)*(1-gamma[i])*(pyo.value(m.alpha)/-1.64))))
        
        #Max ethanol, max ethanol to market
        elif theta[i] < 0.5 and gamma[i] > 0.5:
            region3_profit.append(profit[i])
            region3_scenario.append(i)
            region3_ratio1.append(((market_prices['a'][i] + premium - pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv) - pyo.value(m.mu_el)*pyo.value(m.jet_energy))/market_prices['e'][i])
            region3_ratio2.append(pyo.value(m.alpha)*(market_prices['e'][i]*gamma[i] - pyo.value(m.eth_pc) + (1-gamma[i])*((market_prices['a'][i] + premium -pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv)))/((market_prices['s'][i]-pyo.value(m.sug_pc))*pyo.value(m.alpha) + pyo.value(m.mu_el)* (-47964 - pyo.value(m.jet_energy)*(1-gamma[i])*(pyo.value(m.alpha)/-1.64))))
        
        #Max ethanol, max SAF
        elif theta[i] < 0.5 and gamma[i] < 0.4:
            region4_profit.append(profit[i])
            region4_scenario.append(i)
            region4_ratio1.append(((market_prices['a'][i] + premium - pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv) - pyo.value(m.mu_el)*pyo.value(m.jet_energy))/market_prices['e'][i])
            region4_ratio2.append(pyo.value(m.alpha)*(market_prices['e'][i]*gamma[i] - pyo.value(m.eth_pc) + (1-gamma[i])*((market_prices['a'][i] + premium -pyo.value(m.jet_pc))*pyo.value(m.jet_conv) + pyo.value(m.diesel_price)*pyo.value(m.diesel_conv) + pyo.value(m.gas_price)*pyo.value(m.gas_conv)))/((market_prices['s'][i]-pyo.value(m.sug_pc))*pyo.value(m.alpha) + pyo.value(m.mu_el)* (-47964 - pyo.value(m.jet_energy)*(1-gamma[i])*(pyo.value(m.alpha)/-1.64))))
        
    sorted_data_dict1['profit'] = region1_profit
    sorted_data_dict2['profit'] = region2_profit
    sorted_data_dict3['profit'] = region3_profit
    sorted_data_dict4['profit'] = region4_profit

    sorted_data_dict1['scenario'] = region1_scenario
    sorted_data_dict2['scenario'] = region2_scenario
    sorted_data_dict3['scenario'] = region3_scenario
    sorted_data_dict4['scenario'] = region4_scenario

    sorted_data_dict1['ratio1'] = region1_ratio1
    sorted_data_dict2['ratio1'] = region2_ratio1
    sorted_data_dict3['ratio1'] = region3_ratio1
    sorted_data_dict4['ratio1'] = region4_ratio1

    sorted_data_dict1['ratio2'] = region1_ratio2
    sorted_data_dict2['ratio2'] = region2_ratio2
    sorted_data_dict3['ratio2'] = region3_ratio2
    sorted_data_dict4['ratio2'] = region4_ratio2

    #Covert dictionaries to dataframes 
    sorted_data1_df = pd.DataFrame.from_dict(sorted_data_dict1)
    sorted_data2_df = pd.DataFrame.from_dict(sorted_data_dict2)
    sorted_data3_df = pd.DataFrame.from_dict(sorted_data_dict3)
    sorted_data4_df = pd.DataFrame.from_dict(sorted_data_dict4)

    return sorted_data1_df,sorted_data2_df,sorted_data3_df,sorted_data4_df