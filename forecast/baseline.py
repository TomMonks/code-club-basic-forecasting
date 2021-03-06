'''
Contains classes for creating simple baseline forecasts

These methods might serve as the forecast themselves, but are more likely to be used
as a baseline to determine if more complex models are good enough to employ.

Naive1: Carry last value forward across forecast horizon (random walk)
SNaive: Carry forward value from last seasonal period
Average: Carry forward average of last n periods (default whole time series)
Drift: Carry forward last time period, but allow for upwards/downwards drift.

'''

import numpy as np
import pandas as pd
from scipy.stats import norm


def interval_multipler(level):
    return norm.ppf(1-(1-level)/2)

class Naive1(object):
    '''
    Naive1 or NF1: Carry the last value foreward across a
    forecast horizon
    
    See Makridakis, Wheelwright and Hyndman (1998)

    '''
    def __init__(self):
        '''
        Constructor method

        Parameters:
        -------
        level - list, confidence levels for prediction intervals (e.g. [90, 95])
        '''
        self._in_sample = None
        self._fitted = None
        
    def _get_fitted(self):
        return self._fitted['pred']

    def _get_resid(self):
        return self._fitted['resid']

    def fit(self, train):
        '''
        Train the naive model

        Parameters:
        --------
        train - numpy.array, vector of the time series used for training
        
        '''
        self._pred = train[-1]
        self._fitted = pd.DataFrame(train)
        self._fitted.columns=['actual']
        self._fitted['pred'] = self._fitted['actual'].shift(periods=1)
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']
        self._resid_std = self._fitted['resid'].std()

    def predict(self, horizon):
        '''
        Parameters:
        --------
        horizon - int, forecast horizon. 

        Returns:
        ------
        np.array, vector of predictions. length=horizon
        '''
        return np.full(shape=horizon, fill_value=self._pred, dtype=float)

    def prediction_interval(self, horizon, levels=None):
        '''
        Prediction intervals for naive forecast 1 (NF1)

        lower = pred - z * std_h
        upper = pred + z * std_h

        where 

        std_h = resid_std * sqrt(h)

        resid_std = standard deviation of in-sample residuals

        h = horizon

        See and credit: https://otexts.com/fpp2/prediction-intervals.html

        Pre-requisit: Must have called fit()

        Parameters:
        ---------
        horizon - int, forecast horizon

        levels - list, of floats representing prediction limits
        e.g. [0.80, 0.90, 0.95] will calculate three sets ofprediction intervals
        giving limits for which will include the actual future value with probability 
        80, 90 and 95 percent, respectively (default = [0.8, 0.95]).

        Returns:
        --------
        list - of np.array matricies that contain the lower and upper prediction
        limits for each prediction interval specified.

        '''
        if levels is None:
            levels = [0.80, 0.95]

        zs = [interval_multipler(level) for level in levels]
        
        pis = []

        std_h = self._std_h(horizon)

        for z in zs:
            hw = z * std_h
            pis.append(np.array([self.predict(horizon) - hw, 
                                 self.predict(horizon) + hw]).T)
            
        return pis

    def _std_h(self, horizon):
        indexes = np.sqrt(np.arange(1, horizon+1))
        
        std = np.full(shape=horizon, 
                      fill_value=self._resid_std,
                      dtype=np.float) 

        std *= indexes
        return std

    #breaks PEP8 to fit with statsmodels
    fittedvalues = property(_get_fitted)
    resid = property(_get_resid)
        


class SNaive(object):
    '''
    Seasonal Naive Forecast SNF

    Each forecast to be equal to the last observed value from the same season of the year 
    (e.g., the same month of the previous year).
    
    SNF is useful for highly seasonal data.

    See Hyndman: https://otexts.com/fpp2/simple-methods.html

    '''
    def __init__(self, period):
        '''
        Parameters:
        --------
        period - int, the seasonal period of the daya
                 e.g. weekly = 7, monthly = 12, daily = 24
        '''
        self._period = period
        self._fitted = None

    def _get_fitted(self):
        return self._fitted['pred']

    def _get_resid(self):
        return self._fitted['resid']
        
    def fit(self, train):
        '''
        Seasonal naive forecast - train the model

        Parameters:
        --------
        train - numpy.array, vector of the time series used for training
        '''
        self._f = train[-self._period:]
        self._fitted = pd.DataFrame(train)
        self._fitted.columns=['actual']
        self._fitted['pred'] = self._fitted['actual'].shift(periods=self._period)
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']
        self._resid_std = self._fitted['resid'].std()
        #testing
        lower = np.percentile(self._fitted['resid'].dropna(), 5)
        upper = np.percentile(self._fitted['resid'].dropna(), 95)
        self._resid_std = self._fitted['resid'].clip(lower, upper).std()
        
        
        

    def predict(self, horizon):
        '''
        Predict time series over a horizon

        Parameters:
        --------
        horizon - int, forecast horizon. 

        Returns:
        ------
        np.array, vector of predictions. length=horizon
        '''

        preds = np.array([], dtype=float)
        
        for i in range(0, int(horizon/self._period)):
            preds = np.concatenate([preds, self._f.copy()], axis=0)
            
        preds = np.concatenate([preds, self._f.copy()[:horizon%self._period]], axis=0)
        
        return preds


        
    def prediction_interval(self, horizon, levels=None):
        '''
        Prediction intervals for seasonal naive

        lower = pred - z * std_h
        upper = pred + z * std_h

        where 

        std_h = resid_std * sqrt(k + 1)

        resid_std = standard deviation of in-sample residuals

        k = integer part of (h - 1) / m 

        h = horizon

        m = self._period

        See and credit: https://otexts.com/fpp2/prediction-intervals.html

        Pre-requisit: Must have called fit()

        Parameters:
        ---------
        horizon - int, forecast horizon

        levels - list, of floats representing prediction limits
        e.g. [0.80, 0.90, 0.95] will calculate three sets ofprediction intervals
        giving limits for which will include the actual future value with probability 
        80, 90 and 95 percent, respectively (default = [0.8, 0.95]).

        Returns:
        --------
        list - of np.array matricies that contain the lower and upper prediction
        limits for each prediction interval specified.

        '''

        if levels is None:
            levels = [0.80, 0.95]

        zs = [interval_multipler(level) for level in levels]
        
        pis = []

        std_h = self._std_h(horizon)

        for z in zs:
            hw = z * std_h
            pis.append(np.array([self.predict(horizon) - hw, 
                                 self.predict(horizon) + hw]).T)
            
        return pis

    def _std_h(self, horizon):
        
        h = np.arange(1, horizon+1)
        #need to query if should be +1 or not.
        return self._resid_std * np.sqrt(((h - 1) / self._period).astype(np.int)+1)


    #breaks PEP8 to fit with statsmodels
    fittedvalues = property(_get_fitted)
    resid = property(_get_resid)
        

class Average(object):
    '''
    Average forecast.  Forecast is set to the average
    of the historical data.
    
    See Makridakis, Wheelwright and Hyndman (1998)

    '''
    def __init__(self, min_t=0):
        self._min_t = min_t
        self._pred = None
        self._fitted = None

    def _get_fitted(self):
        return self._fitted['pred']

    def _get_resid(self):
        return self._fitted['resid']
    
    def fit(self, train):
        '''
        Train the model

        Parameters:
        --------
        train - numpy.array, vector of the time series used for training
        '''
        
        self._t = len(train[-self._min_t:])
        self._pred = train[-self._min_t:].mean()
        self._resid_std = (train[-self._min_t:] - self._pred).std()
        self._fitted = pd.DataFrame(train[-self._min_t:])
        self._fitted.columns=['actual']
        self._fitted['pred'] = self._pred
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']

        
    def predict(self, horizon):
        '''
        Predict time series over a horizon

        Parameters:
        --------
        horizon - int, forecast horizon. 

        Returns:
        ------
        np.array, vector of predictions. length=horizon
        '''
        return np.full(shape=horizon, fill_value=self._pred, dtype=float)

    def prediction_interval(self, horizon, levels=None):
        
        if levels is None:
            levels = [0.80, 0.95]

        zs = [interval_multipler(level) for level in levels]

        hws = []
        for z in zs:
            hw = self._half_width(horizon, z)
            hws.append(np.array([self._pred - hw, self._pred + hw]).T)

        return hws


    def _half_width(self, horizon, z):
        hw =  z * self._resid_std * np.sqrt(1 + (1/self._t))
        return np.full(shape=horizon, fill_value=hw, dtype=np.float)

    #breaks PEP8 to fit with statsmodels
    fittedvalues = property(_get_fitted)
    resid = property(_get_resid)


       

         


class Drift(object):
    '''
    Naive1 with drift: Carry the last value foreward across a
    forecast horizon but allow for upwards of downwards drift.

    Drift = average change in the historical data.   
    
    https://otexts.com/fpp2/simple-methods.html

    '''
    def __init__(self):
        self._fitted = None

    def _get_fitted(self):
        return self._fitted['pred']

    def _get_resid(self):
        return self._fitted['resid']
    
    def fit(self, train):
        '''
        Train the naive model

        Parameters:
        --------
        train - pd.DataFrame, the time series used for training
        
        '''
        self._last_value = train[-1:][0]
        self._t = train.shape[0]
        self._gradient = ((self._last_value - train[0]) / (self._t - 1))

        self._fitted = pd.DataFrame(train)
        self._fitted.columns=['actual']
        self._fitted['pred'] = train[0] + np.arange(1, self._t+1, dtype=float) * self._gradient
        self._fitted['resid'] = self._fitted['actual'] - self._fitted['pred']
        self._resid_std = self._fitted['resid'].std()


    def predict(self, horizon):
        '''
        Parameters:
        --------
        horizon - int, forecast horizon. 

        Returns:
        ------
        np.array, vector of predictions. length=horizon
        '''
        preds = np.arange(1, horizon+1, dtype=float) * self._gradient
        preds += self._last_value
        return preds

    def prediction_interval(self, horizon, levels=None):
        '''
        Prediction intervals for seasonal naive

        lower = pred - z * std_h
        upper = pred + z * std_h

        where 

        std_h = resid_std * sqrt(h* (1 + h/T))

        resid_std = standard deviation of in-sample residuals

        h = horizon

        T = length of time series

        See and credit: https://otexts.com/fpp2/prediction-intervals.html

        Pre-requisit: Must have called fit()

        Parameters:
        ---------
        horizon - int, forecast horizon

        levels - list, of floats representing prediction limits
        e.g. [0.80, 0.90, 0.95] will calculate three sets ofprediction intervals
        giving limits for which will include the actual future value with probability 
        80, 90 and 95 percent, respectively (default = [0.8, 0.95]).

        Returns:
        --------
        list - of np.array matricies that contain the lower and upper prediction
        limits for each prediction interval specified.

        '''

        if levels is None:
            levels = [0.80, 0.95]

        zs = [interval_multipler(level) for level in levels]
        
        pis = []

        std_h = self._std_h(horizon)

        for z in zs:
            hw = z * std_h
            pis.append(np.array([self.predict(horizon) - hw, 
                                 self.predict(horizon) + hw]).T)
            
        return pis

    def _std_h(self, horizon):
        
        h = np.arange(1, horizon+1)
        return self._resid_std * np.sqrt(h * (1 + (h / self._t)))

    #breaks PEP8 to fit with statsmodels
    fittedvalues = property(_get_fitted)
    resid = property(_get_resid)



class EnsembleNaive(object):
    def __init__(self, seasonal_periods=7, average_lookback=180):
        self._estimators = {'NF1':Naive1(),
                            'SNaive':SNaive(period=seasonal_periods),
                            'Average':Average(),
                            'Average_'+str(average_lookback):Average(min_t=average_lookback),
                            'Drift':Drift(),
                            'SES':StatsModelsForecastObject(SimpleExpSmoothingWrapper())
                            }
       

    def fit(self, train):
        for key, estimator in self._estimators.items():
            estimator.fit(train)
        
    def predict(self, horizon):
        preds = []
        for key, estimator in self._estimators.items():
            model_preds = estimator.predict(horizon)
            preds.append(model_preds)

        return np.array(preds).mean(axis=0)



def baseline_estimators(seasonal_periods, average_lookback=None):
    '''
    Generate a collection of baseline forecast objects
    
    Parameters:
    --------
    seasonal_periods - int, number of seasonal periods in the data (e.g daily = 7)
    average_lookback - int, number of lagged periods that average baseline includes
    
    Returns:
    --------
    dict, forecast objects
    '''
    
    estimators = {'NF1':Naive1(),
                  'SNaive':SNaive(period=seasonal_periods),
                  'Average':Average(),
                  'Drift':Drift(),
                  'Ensemble':EnsembleNaive(seasonal_periods=seasonal_periods, 
                                           average_lookback=average_lookback),
                  }

    if average_lookback != None:
        estimators['Average_'+str(average_lookback)] = Average(min_t=average_lookback)
    
    return estimators


def boot_prediction_intervals(preds, resid, horizon, levels=None, boots=1000):
    '''
    Constructs bootstrap prediction intervals for forecasting.

    Parameters:
    -----------

    preds - array-like, predictions over forecast horizon

    resid - array-like, in-sample prediction residuals

    horizon - int, forecast horizon (e.g. 12 months or 7 days)

    levels - list of floats, prediction interval precisions (default=[0.80, 0.95])

    boots - int, number of bootstrap datasets to construct (default = 1000)

    Returns:
    ---------

    list of numpy arrays.  Each numpy array contains two columns of the upper 
    and lower prediction limits across the forecast horizon.
    '''
    
    if levels == None:
        levels = [0.80, 0.95]

    resid = _drop_na_from_series(resid)
    
    sample = np.random.choice(resid, size=(boots, horizon))
    sample = np.cumsum(sample,axis=1) 
    
    data = preds + sample

    pis = []

    for level in levels:
        upper = np.percentile(data, level*100, interpolation='higher', axis=0)
        lower = np.percentile(data, 100-(level*100), interpolation='higher', axis=0)
        pis.append(np.array([lower, upper]))
       
    return pis


def _drop_na_from_series(data):
    '''
    Drops all NaN from numpy array or pandas series.
    
    Parameters:
    -------
    data, array-like,
    np.ndarray or pd.Series.  

    Returns:
    -------
    np.ndarray removing NaN.

    '''
    if isinstance(data, pd.Series):
        return data.dropna().to_numpy()
    else:
        return data[~np.isnan(data)]

