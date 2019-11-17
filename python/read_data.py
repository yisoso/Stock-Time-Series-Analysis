import pandas as pd

ibm_data_path = 'F:\pythonProject\Stock-Time-Series-Analysis\data\IBM_2006-01-01_to_2018-01-01.csv'
data = pd.read_csv(ibm_data_path)

# 'Close' is used as the analysis objection
cl_time = data['Close']
