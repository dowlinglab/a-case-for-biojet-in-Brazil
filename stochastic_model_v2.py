'''
This file contains a function defining a stochastic model for an integrated sugarcane mill including the production
of sustainable aviation fuel (SAF) and diesel to support the manuscript "A Case for Bio-jet Fuel from Bioethanol in Brazil: An Optimization-based
Analysis Using Historical Market Data."

Created by Madelynn Watson and the University of Notre Dame
'''

#Import Necessary Packages
import pandas as pd
import numpy as np
import pyomo.environ as pyo

#Define a function to create the optimization model in the pyomo enviornment
def create_stochastic_model_v2(premium, market_prices, scenarios, eth_market, eth_min, jet_min, sugar_min, juice_flex, eth_flex, jet_energy, d_price, g_price):

    '''
    This function buils a two-stage linear stochastic model in Pyomo for an integrated sugarcane mill.

    Inputs: 
    
            premium: premium price paid for sustainable aviation fuel (SAF), bounds: (0,infinity), Units: $R/m3
            sugar_min: minimum fraction of juice sent to produce sugar, bounds: (0,1), units: tonne jui-f/tonne jui
            eth_market: minimum fraction of ethanol sent to market, bounds: (0,1), units: m3 eth-p/m3 eth
            eth_min: minimum fraction of juice sent to produce ethanol, bounds: (0,1), units: tonne jui-d/tonne jui
            saf_min: minimum fraction of ethanol sent to produce SAF, bounds:(0,1), units: m3 eth-j/m3 eth
            juice_flex: weekly flexibility of juice split (fraction) in stage 2 decisions, bounds: (0,1)
            eth_flex: weekly flexibility of ethanol split (fraction) in stage 2 decisions, bounds: (0,1)
            market_prices: dictionary of market prices indexed by product and scenarios (weeks)
            scenarios: list of time scenarios (weeks) for market prices
            jet_energy: An effciency factor for converting ethanol to jet fuel, units (MWh/m3 ethanol used to make SAF)
            d_price: Average diesel price ($R/m3 diesel)
            gas_price: Average gas_price

    Returns: Pyomo model m
    '''

#Create a concrete model in the pyomo enviornment
    m=pyo.ConcreteModel()

#SETS
    products_and_intermediates = ['s', 'e', 'f', 'c', 'j', 'm', 'b', 'v', 'p','e1', 'e2', 'a', 'j1', 'j2', 'd','g']
    process_units = [1,2,3,4,5,6]
    piecewise_intervals1 = np.arange(1,10) #peicewise surrogate intervals exluding the last element
    piecewise_intervals2 = np.arange(1,11)

    #PYOMO SETS
    m.PRODUCTS_AND_INTERMEADIATES = pyo.Set(initialize = products_and_intermediates)
    m.UNITS = pyo.Set(initialize = process_units)
    m.PIECEWISE_INTERVALS1 = pyo.Set(initialize = piecewise_intervals1)
    m.PIECEWISE_INTERVALS2 = pyo.Set(initialize = piecewise_intervals2)
    m.SCENARIOS = pyo.Set(initialize = scenarios)

#PARAMETERS
    # Read in nominal parameter data from excel sheet
    df_maxcap = pd.read_excel('Mutran_jet_datav2.xlsx', sheet_name='MaxAnnualCap')
    df_prodcost = pd.read_excel('Mutran_jet_datav2.xlsx', sheet_name='ProductionCost')
    df_conv = pd.read_excel('Mutran_jet_datav2.xlsx', sheet_name='Conversions')
    df_gen = pd.read_excel('Mutran_jet_datav2.xlsx', sheet_name='Generation')
    df_electricity_gen = pd.read_excel('Mutran_jet_datav2.xlsx', sheet_name='LinCoeffs')
    df_jfracs = pd.read_excel('Mutran_jet_datav2.xlsx', sheet_name='JuiceFracs')
    
    #Create empty dictionaries to store parameter data
    max_cap = {} #Maximum capacity data for process units
    prodcost = {} #Production cost data for saleable products
    conv = {} #Conversion data for products and intermeadiates
    gen = {} #Generation data for products and intermeadiates
    electricity_gen = {} #Electricty generation for linear piecewise intervals
    jfracs = {} #Juice fractions for linear piecwise intervals

    #Populate dictionaries
    for i in range(len(df_maxcap['process'])):
            max_cap[df_maxcap['process'][i]] = df_maxcap['MaxAnnualCap'][i]

    for i in range(len(df_prodcost['saleable_product'])):
            prodcost[df_prodcost['saleable_product'][i]] = df_prodcost['cost'][i]

    for k in products_and_intermediates:
        for i in range(len(df_conv[k])):
            conv[(df_conv['resource'][i],df_conv['process'][i],k)] = df_conv[k][i]
            
    for k in products_and_intermediates:
        for i in range(len(df_gen[k])):
            gen[(df_gen['process'][i],k)] = df_gen[k][i]
            
    for i in range(len(df_electricity_gen['el'])):
        electricity_gen[(df_electricity_gen['split_fracs'][i],'p')] = df_electricity_gen['el'][i]
            
    for i in range(len(df_jfracs['split_points'])):
            jfracs[df_jfracs['split_points'][i]] = df_jfracs['juice_fraction'][i]
            

#PYOMO PARAMETERS
    #Create pyomo objects for scalar model parameters
    m.jet_conv = pyo.Param(initialize = conv['e2',4,'a'], mutable = True) #SAF conversion factor for unit 4 (ethanol upgrading), m3 a/m3 e2
    m.gas_conv = pyo.Param(initialize = conv['e2',4,'g'], mutable=True) #Gasoline conversion factor for unit 4 (ethanol upgrading), m3 g/m3 e2
    m.diesel_conv = pyo.Param(initialize = conv['e2',4,'d'], mutable=True) #Diesel conversion factor for unit 4 (ethanol upgrading), m3 d/m3 e2
    m.jet_pc = pyo.Param(initialize = prodcost['a'], mutable = True) #Production cost of SAF, R$/m3 a
    m.eth_pc = pyo.Param(initialize = prodcost['e']) #Production cost of ethanol, R$/m3 e
    m.sug_pc = pyo.Param(initialize = prodcost['s']) #Production cost of sugar, R$/tonne s
    m.mu_el = pyo.Param(initialize = 274.6 - prodcost['p']) #Production cost of electricity R$/MWh
    m.jet_energy = pyo.Param(initialize = jet_energy) #Energy consumption for SAF production MWh/m3 e2
    m.Ca = pyo.Param(initialize = 3000000)  #Mill capacity in tonnes of sugarcane cane, tonne/year
    m.diesel_price = pyo.Param(initialize = d_price) #Diesel price, R$/m3 d
    m.gas_price = pyo.Param(initialize = g_price) #Gasoline price, R$/m3 g
    m.premium = pyo.Param(initialize = premium , mutable = True, within = pyo.NonNegativeReals) #SAF premium, R$/m3 a
    m.sugar_min = pyo.Param(initialize = sugar_min, mutable = True) #Minimum production fraction of sugar
    m.saf_min = pyo.Param(initialize = jet_min, mutable = True) #Minimum production fraction of SAF
    m.eth_market = pyo.Param(initialize = eth_market, mutable = True) #Minimum fraction of ethanol sold to the market
    m.eth_min = pyo.Param(initialize = eth_min, mutable = True) #Minimum production fraction of ethanol
    m.juice_flex = pyo.Param(initialize = juice_flex, mutable = True) #Juice split flexibility
    m.eth_flex = pyo.Param(initialize = eth_flex, mutable = True) #Ethanol split flexibility
    m.N = pyo.Param(initialize = len(scenarios)) #Total number of scenarios

#VARIABLES
    #Positive Variables
    m.x = pyo.Var(m.PRODUCTS_AND_INTERMEADIATES, m.SCENARIOS, within = pyo.NonNegativeReals)
    m.y = pyo.Var(m.PRODUCTS_AND_INTERMEADIATES, within = pyo.NonNegativeReals)
    m.csi = pyo.Var(m.SCENARIOS,m.PIECEWISE_INTERVALS2, within = pyo.NonNegativeReals, bounds=(0,1))
    m.profit = pyo.Var(m.SCENARIOS, within = pyo.NonNegativeReals)

    #Binary Variables for surrogate model
    m.z = pyo.Var(m.SCENARIOS,m.PIECEWISE_INTERVALS1, within = pyo.Binary)

#CONSTRAINTS
#Superstructure Constraints (Scenario Based)
    def mill1(m,s):
        return m.x['j',s] == m.Ca * conv['c',1,'j']
    m.mill_mass_bal1 = pyo.Constraint(m.SCENARIOS, rule = mill1)

    def mill2(m,s):
        return m.x['b',s] == m.Ca * conv['c',1,'b']
    m.mill_mass_bal2 = pyo.Constraint(m.SCENARIOS,rule = mill2)

    def juice1(m,s):
        return m.x['j',s] == m.x['j1',s] + m.x['j2',s]
    m.juice_split = pyo.Constraint(m.SCENARIOS, rule = juice1)

    def juice2(m,s):
        return m.x['j1',s] <= max_cap[2]
    m.factory_max_cap = pyo.Constraint(m.SCENARIOS, rule = juice2)

    def juice4(m,s):
        return m.x['j2',s] <= max_cap[3]
    m.distillary_max_cap = pyo.Constraint(m.SCENARIOS, rule = juice4)

    def sugar1(m,s):
        return m.x['s',s] == conv['j', 2, 's'] * m.x['j1',s]
    m.sugar_prod = pyo.Constraint(m.SCENARIOS, rule = sugar1)

    def sugar2(m,s):
        return m.x['m',s] == gen[2,'m'] * m.x['s',s]
    m.mol_generation = pyo.Constraint(m.SCENARIOS, rule = sugar2)

    def ethanol1(m,s):
        return m.x['e',s] == conv['j',3,'e'] * m.x['j2',s] + conv['m',3,'e']*m.x['m',s]
    m.eth_prod = pyo.Constraint(m.SCENARIOS, rule = ethanol1)

    def ethanol2(m,s):
        return m.x['v',s] == gen[3,'v'] * m.x['e',s]
    m.vin_generation = pyo.Constraint(m.SCENARIOS, rule = ethanol2)

    def vin2(m,s):
        return m.x['f',s] == conv['v',5,'f']* m.x['v',s]
    m.fert_prod =pyo.Constraint(m.SCENARIOS, rule = vin2)

    def eth_split(m,s):
        return m.x['e',s] == m.x['e2',s] + m.x['e1',s]
    m.ethanol_split = pyo.Constraint(m.SCENARIOS, rule = eth_split)

    def jet_fuel(m,s):
        return m.x['a',s] == m.x['e2',s] * m.jet_conv
    m.jet_prod = pyo.Constraint(m.SCENARIOS, rule = jet_fuel)

    def diesel(m,s):
        return m.x['d',s] == m.x['e2',s] * conv['e2',4,'d']
    m.d_prod = pyo.Constraint(m.SCENARIOS, rule = diesel)

    def gas(m,s):
        return m.x['g',s] == m.x['e2',s] * conv['e2',4,'g']
    m.g_prod = pyo.Constraint(m.SCENARIOS, rule = gas)


#Surplus Bagasse Surrogate Models 
    def bag0(m,s):
        return m.x['j1',s]/(m.Ca*conv['c',1,'j']) == jfracs[1] + sum((jfracs[i+1]-jfracs[i])*m.csi[s,i] for i in m.PIECEWISE_INTERVALS2)
    m.pwl0 = pyo.Constraint(m.SCENARIOS,rule = bag0)

    def bag1(m,s):
        return m.x['p',s] <= electricity_gen[1,'p'] + sum((electricity_gen[i+1, 'p'] - electricity_gen[i,'p'])*m.csi[s,i] for i in m.PIECEWISE_INTERVALS2)
    m.pwl1 = pyo.Constraint(m.SCENARIOS, rule = bag1)

    def bag3(m,s,b):
        return m.csi[s,b] >= m.z[s,b]
    m.pwl3 = pyo.Constraint(m.SCENARIOS,m.PIECEWISE_INTERVALS1, rule = bag3)

    def bag4(m,s,b):
        return m.z[s,b] >= m.csi[s,b+1]
    m.pwl4 = pyo.Constraint(m.SCENARIOS,m.PIECEWISE_INTERVALS1, rule = bag4)

#Minimum Production Constraints
    def min_eth(m,s):
        return m.x['e',s] >= m.eth_min*m.x['j',s]*conv['j',3,'e']
    m.minimum_ethanol = pyo.Constraint(m.SCENARIOS, rule=min_eth)

    def min_eth_market(m,s):
        return m.x['e1',s] >= m.eth_market*m.x['e',s]
    m.minimum_eth_market = pyo.Constraint(m.SCENARIOS, rule = min_eth_market)

    def mini_saf(m,s):
        return m.x['e2',s] >= m.saf_min*m.x['e',s]
    m.minimum_SAF = pyo.Constraint(m.SCENARIOS,rule = mini_saf)

    def mini_sug(m,s):
        return m.x['j1',s] >= m.sugar_min*m.x['j',s]
    m.minimum_sug = pyo.Constraint(m.SCENARIOS,rule=mini_sug)

#Flexibility Constraints
    def jui_flex_1(m,s):
      return (1 - m.juice_flex) * m.y['j1'] <= m.x['j1',s] 
    m.flex_jui_1 = pyo.Constraint(m.SCENARIOS, rule = jui_flex_1)

    def jui_flex_3(m,s):
      return (1 + m.juice_flex) * m.y['j1'] >= m.x['j1',s] 
    m.flex_jui_3 = pyo.Constraint(m.SCENARIOS, rule = jui_flex_3)

    def jui_flex_2(m,s):
        return (1 - m.juice_flex) * m.y['j2'] <= m.x['j2',s]
    m.flex_jui_2 = pyo.Constraint(m.SCENARIOS, rule = jui_flex_2)

    def jui_flex_4(m,s):
        return (1 + m.juice_flex) * m.y['j2'] >= m.x['j2',s] 
    m.flex_jui_4 = pyo.Constraint(m.SCENARIOS, rule = jui_flex_4)

    def eth_flex_1(m,s):
        return (1-m.eth_flex) * m.y['e1'] <= m.x['e1',s]
    m.flex_eth_1 = pyo.Constraint(m.SCENARIOS, rule = eth_flex_1)

    def eth_flex_3(m,s):
        return (1+m.eth_flex) * m.y['e1'] >= m.x['e1',s]
    m.flex_eth_3 = pyo.Constraint(m.SCENARIOS, rule = eth_flex_3)

    def eth_flex_2(m,s):
        return (1-m.eth_flex) * m.y['e2'] <= m.x['e2',s] 
    m.flex_eth_2 = pyo.Constraint(m.SCENARIOS, rule = eth_flex_2)

    def eth_flex_4(m,s):
        return (1+m.eth_flex) * m.y['e2'] >= m.x['e2',s]
    m.flex_eth_4 = pyo.Constraint(m.SCENARIOS, rule = eth_flex_4)

#EXPRESSIONS
    #Calculation for alpha in derived ratios
    def alpha_calc(m):
        return m.Ca*conv['c',1,'j'] * conv['j',2,'s']
    m.alpha = pyo.Expression(rule=alpha_calc)

#Profit
    def profit(m,s):
        return m.profit[s] == m.x['s',s]*(market_prices['s'][s]-prodcost['s']) - m.x['e',s]*prodcost['e'] + market_prices['e'][s]*m.x['e1',s] + m.x['a',s]*(market_prices['a'][s] + m.premium - m.jet_pc) + (m.x['p',s] - jet_energy*m.x['e2',s])*m.mu_el + m.x['d',s] * m.diesel_price + m.x['g',s] * m.gas_price
    m.prof_scenario = pyo.Constraint(m.SCENARIOS, rule = profit)

#OBJECTIVE
    def obj_rule(m):
        return (1/m.N) * sum(m.profit[s] for s in m.SCENARIOS)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m