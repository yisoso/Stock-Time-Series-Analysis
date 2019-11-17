from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python.checking_stationarity import diff1

arima_model = ARIMA(diff1, order=(3, 1, 2))
result = arima_model.fit(disp=-1, method='css')

from python.checking_stationarity import cl_time

# prediction
pre_results = result.predict()
idx = range(0, 3015, 1)

pre_list = []
for i in range(len(pre_results)):
    pre_list.append(np.array(pre_results)[i])
pre_results_series = pd.Series(np.array(pre_list), index=idx)

# restored
pre_results_restored = pd.Series(np.array(cl_time)[5], index=idx).append(pre_results_series).cumsum()
pre_result = cl_time + pre_results_series

ts = cl_time[pre_results_series.index]
pre_result.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('comparison')
plt.show()
