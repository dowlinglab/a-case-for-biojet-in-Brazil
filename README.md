# a-case-for-biojet-in-Brazil
Supporting codes for "A Case for Bio-jet Fuel from Bioethanol in Brazil: An Optimization-based Analysis Using Historical Market Data."

## Citation
If you find this useful, consider citing us as: Watson, M., da Silva, A. V., Machado, P. G., de Oliveira Ribeiro, C., do Nascimento, C. A. O., & Dowling, A. (2024). The case for bio-jet fuel from bioethanol in Brazil: An optimization-based analysis using historical market data. ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-j5l9x This content is a preprint and has not been peer-reviewed.

## Dependencies
The scripts in this repository build an optimization model via Pyomo (v6.6.1) and solve using Gurobi (v10.0.3).

## Repository Content
The content of this repository is detailed below:
### Python Scripts
stochastic_model_v2: contains a function to create and initialize the optimization model

perfect_information_analysis: contains a function for sensitivity analysis to SAF premiums considering the perfect information case with no flexibility

technology_sensitivty: contains a function for a 2D sensitivity analysis to ATJ technology parameters considering no flexibility

flexibility_analysis: contains a function for a 2D sensitivity analysis to the flexibility levels in the ethanol and sugarcane juice split flows

### Jupyter Notebooks
Historical_Price_Data_Analysis: make plots to visualize the historical market data in Brazil for sugar, ethanol, and jet fuel (scatter plot)

Perfect_Information_Case_Analysis: make plots to visualize four operating regions (scatter and pie plots), SAF premium sensitivity (area plot), and ATJ technology sensitivity (contour plot)

Flexibility_Analysis: make plots to visualize the flexibility sensitivity results (contour plot and histogram)

Capex_Budget: make plots to quantify the additional revenue from SAF as a function of SAF premiums and compare with reported CAPEX and OPEX values (line plot)

### Folders
premium_sensitivity_data: results files for sensitivity analysis on SAF premium (results from Perfect_Information_Case_Analysis)

flexibility_data: results files for sensitivity analysis on flexibility considering the conventional ATJ technology (results from Flexibility_Analysis)

flexibility_data_opt: results files for sensitivity analysis on flexibility considering the optimistic ATJ technology (results from Flexibility_Analysis)

Results_Figures: all figures produced for the manuscript

### Other Files
README: this file

Mutran_jet_datav2: excel file containing historical price data and nominal model model parameters

