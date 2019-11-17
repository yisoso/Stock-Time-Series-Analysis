import statsmodels.tsa.stattools as ts
import numpy as np
from python.read_data import cl_time
import statsmodels.api as sm


def stationarity_checking(timeserise):
    ADF_result = ts.adfuller(timeserise)
    return ADF_result


pure_result = stationarity_checking(cl_time)
print(pure_result)

cl_time_log = np.log(cl_time)

diff1 = cl_time.diff(1).dropna()
diff1_result = stationarity_checking(diff1)
print(diff1_result)

# AIC
AIC = sm.tsa.arma_order_select_ic(cl_time, max_ar=6, max_ma=4, ic='aic')['aic_min_order']

# BIC
BIC = sm.tsa.arma_order_select_ic(cl_time, max_ar=6, max_ma=4, ic='bic')['bic_min_order']

# HQIC
HQIC = sm.tsa.arma_order_select_ic(cl_time, max_ar=6, max_ma=4, ic='hqic')['hqic_min_order']
print('the AIC is{},\nthe BIC is{}\n the HQIC is{}'.format(AIC, BIC, HQIC))
