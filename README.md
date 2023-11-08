# Time Series
# EXP 1 - Time series data
```py
df.Close.resample("Q").mean()
monthly=df.Close.resample("M").mean()
plt.plot(monthly)
plt.xlabel("Date")
plt.ylabel("BTC")
```
# EXP 2 - Decomposyion Analysis
```py
df.set_index('date',inplace=True)
df.index = pd.to_datetime(df.index)
df.plot()

comp = seasonal_decompose(df['meantemp'].iloc[:100],model="addition")
comp.plot()
```
# EXP 3 - Log
```py
df['#Passengers']=np.log(df['#Passengers'])
df.plot()
log_comp = seasonal_decompose(df['#Passengers'],model="addition")
log_comp.plot()
```
```py
import numpy as np
import pandas as pd
data= pd.read_csv('AirPassengers.csv')
data.head()
data.dropna(inplace=True)
x=data['Month']
y=data['#Passengers']
data_log=np.log(data['#Passengers'])
X=data['Month']
Y=data_log
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.xlabel('Original Data')
plt.plot(X,Y)
plt.xlabel('Log- Transformed data')

```
# EXP 4 - ACF
```py
import matplotlib.pyplot as plt
import numpy as np
data = np.array([3, 16, 156, 47, 246, 176, 233, 140, 130,
 101, 166, 201, 200, 116, 118, 247,
 209, 52, 153, 232, 128, 27, 192, 168, 208,
 187, 228, 86, 30, 151, 18, 254,
 76, 112, 67, 244, 179, 150, 89, 49, 83, 147, 90,
 33, 6, 158, 80, 35, 186, 127])
lags = range(35)
acorr = len(lags) * [0]
mean = sum(data) / len(data)
var = sum([(x - mean)**2 for x in data]) / len(data)
ndata = [x - mean for x in data]
for l in lags:
    c = 1 # Self correlation

    if (l > 0):
        tmp = [ndata[l:][i] * ndata[:-l][i]
            for i in range(len(data) - l)]

        c = sum(tmp) / len(data) / var
        print(c)
        acorr[l] = c
plt.acorr(tmp)
print("The Autocorrelation plot for the data is:")
plt.grid(True)
plt.show()

```
# EXP 5 - Auto Regression
```py
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
dtest=adfuller(X,autolag='AIC')
print("ADF:",dtest[0])
print("P value:",dtest[1])
print("No. of lags:",dtest[2])
print("No. of observations used for ADF regression:",dtest[3])
X_train=X[:len(X)-15]
X_test=X[len(X)-15:]
AR_model=AutoReg(X_train,lags=13).fit()
print(AR_model.summary())
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(X,lags=25)
acf=plot_acf(X,lags=25)
pred=AR_model.predict(start=len(X_train),end=len(X_train)+len(X_test)-1,dynamic=False)
pred.plot()
import sklearn.metrics
mse=sklearn.metrics.mean_squared_error(X_test,pred) 
mse**0.5
X_test.plot()
pred.plot()
```
# EXP 6 - Moving Average Model
```py
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import ExponentialSmoothing,SimpleExpSmoothing,Holt
import rcParams
rcParams['figure.figsize']= 20,5
import warnings
warnings.filterwarnings('ignore')
electricitytimeseries = pd.read_csv('Electric_Production.csv',header=0,index_col=0)
electricitytimeseries.shape
electricitytimeseries.head(20)
# MOVING AVERAGE METHOD
plt.plot(electricitytimeseries[1:50]['Value'])
plt.xticks(rotation=30)
plt.show()
# rolling average transfrom
rollingseries = electricitytimeseries[1:50].rolling(window=5)
rollingmean = rollingseries.mean()
# finding rolling mean MA(5)
print(rollingmean.head(10))
# plot transfrom dataset
rollingmean.plot(color='purple')
pyplot.show()
# rolling average transfrom
rollingseries = electricitytimeseries[1:50].rolling(window=10)
rollingmean = rollingseries.mean()
#finding rolling mean MA()
print(rollingmean.head(10))
# Exponential smoothing - single
data = electricitytimeseries[1:50]
fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2,optimized=False)
fit2 = SimpleExpSmoothing(data).fit(smoothing_level=0.8,optimized=False)
plt.figure(figsize=(18,8))
plt.plot(electricitytimeseries[1:50],marker='o',color='black')
plt.xticks(rotation=30)
plt.plot(fit1.fittedvalues,marker='o',color='blue')
plt.plot(fit2.fittedvalues,marker='o',color='red')
```
# EXP 7 - ARMA
```py
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [10, 7.5]

ar1 = np.array([1,0.33])
ma1 = np.array([1,0.9])
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)

ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```
# EXP 8 - ARIMA
```py
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import sklearn.metrics
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("electricity.csv")
df
X = df["temperature"][:500]
X
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(X,autolag='AIC')
print("ADF :",dftest[0])
print("P value:",dftest[1])
print("No of lags:",dftest[2])
print("No of observations used for ADF regression:",dftest[3])

stepwise_fit= auto_arima(X,trace=True,supress_warning=True)
train = X[:len(X)-20]
test  = X[len(X) - 20:]
stepwise_fit.summary()

model=ARIMA(train,order=(2,0,2)).fit()
pred=model.predict(start=len(train),end=len(train)+len(test)+20,typ = "levels")
pred.plot(legend=True)
test.plot(legend=True)
```
# EXP 9 - Linear and Poly
```py
def calculateB(x,y):
    sx = sum(x)
    sy = sum(y)
    sxy = 0
    sx2 = 0
    n = len(x)
    for i in range(n):
        sxy += x[i] * y[i]
        sx2 += x[i] * x[i]
    b = (n * sxy - sx * sy)/(n * sx2 - sx * sx)
    return b

def calculateA(x,y,b):
    n = len(x)
    meanX = sum(x) / n
    meanY = sum(y) / n
    a = meanY - b*meanX
    return a

def predictY(X,a,b):
    return [a+b*x for x in X]

x = [95,85,80,70,60]
y = [90,80,70,65,60]
b = calculateB(x,y)
a = calculateA(x,y,b)
pred = predictY(x,a,b)
plt.scatter(x,y)
plt.plot(x,pred, color="red")

print("Line Equation : Y =",a,"+",b,"* X")
print("Trend values: ",pred)
```
```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
df = pd.read_csv('BTC.csv')
df

Y = df["Close"].values
Y
plt.plot(Y)

df['sno'] = range(len(df))
X = df['sno'].values
X

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
lin.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
 
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
 
poly.fit(X_poly, Y)
lin2 = LinearRegression()
lin2.fit(X_poly, Y)


plt.scatter(X, Y, color='blue')
 
plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression') 
plt.show()

plt.scatter(X, Y, color='blue')
 
plt.plot(X, lin2.predict(poly.fit_transform(X)),color='red')
plt.title('Polynomial Regression')
plt.show()

```
# EXP 10 - Holt Winter
```py
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

import pandas as pd
airline  = pd.read_csv('AirPassengers.csv',index_col='Month',parse_dates=True)
airline.plot()

train_airline = airline[:108] 
test_airline = airline[108:] 
fitted_model = ExponentialSmoothing(train_airline['#Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

test_predictions = fitted_model.forecast(36).rename('HW Test Forecast')
test_predictions[:10]

train_airline['#Passengers'].plot(legend=True,label='TRAIN')
test_airline['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
plt.title('Train and Test Data');

train_airline['#Passengers'].plot(legend=True,label='TRAIN')
test_airline['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters');

print("Mean Absolute Error = ",mean_absolute_error(test_airline,test_predictions))

final_model = ExponentialSmoothing(airline['#Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
forecast_predictions = final_model.forecast(steps=36)
airline['#Passengers'].plot(figsize=(12,8),legend=True,label='Current Airline Passengers')
forecast_predictions.plot(legend=True,label='Forecasted Airline Passengers')
plt.title('Airline Passenger Forecast');

```

