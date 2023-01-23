# Predicting and Assessing Wildfire Evacuation Decision-Making Using Machine Learning: Findings from the 2019 Kincade Fire
Implemented by Ningzhe Xu.

# Required Software
R version 3.6.1

# Required Libraries
* randomForest
* tree
* class
* e1071
* xgboost
* nnet

# File Specifications
* wildfire evacuation.R: R code for comparing machine learning models and the logistic regression, and testing whether the difference in their performances is significant.

# Paper
Xu, N., Lovreglio, R., Kuligowski, E.D., Cova, T.J., Nilsson, D., & Zhao, X. Predicting and Assessing Wildfire Evacuation Decision-Making Using Machine Learning: Findings from the 2019 Kincade Fire. Fire Technol (2023). https://doi.org/10.1007/s10694-023-01363-1

# Data
The original dataset has the data structure as detailed below:
1.	Each row is an observation and each column is a variable.
2.	There are 31 variables (i.e., 31 columns) in total, including Residence_Less5, Residence_10more, Own_House, Detached, Multi-family, Mobile_Manufactured, Warning_Trust_Source, Warning_In_Person, Fire_Cues, Evacuation_Decision, Female, Children, Adult, Animals, Emergency_plan, Medical_condition, Age_45_54, Age_55_64, Age_65more, Preparation, Bachelor, Graduate, Income_50000_74999, Income_75000_99999, Income_100000_124999, Income_125000_149999, Income_150000_174999, Income_175000more, Prefire_perceptions_of_safety, Risk_Perceiption, Prior_Awareness_Threat. Please refer to the paper (Subsection 3.2) for more details about the data and variables.

A simulated dataset with 5 observations (i.e., demo.csv) is provided as an example to illustrate the data structure. The original dataset cannot be publicly released under IRB regulations. 

For any questions, please contact Ningzhe Xu (nxu3@crimson.ua.edu).
