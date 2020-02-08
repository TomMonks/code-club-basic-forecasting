import numpy as np

def time_series_cv(model, error_func, train, val, horizons, step=1):
    '''
    Time series cross validation across multiple horizons for a single model.

    Incrementally adds additional training data to the model and tests
    across a provided list of forecast horizons. Note that function tests a
    model only against complete validation sets.  E.g. if horizon = 15 and 
    len(val) = 12 then no testing is done.  In the case of multiple horizons
    e.g. [7, 14, 28] then the function will use the maximum forecast horizon
    to calculate the number of iterations i.e if len(val) = 365 and step = 1
    then no. iterations = len(val) - max(horizon) = 365 - 28 = 337.
    
    Parameters:
    --------
    model - forecasting model

    error_func - function to measure forecast error

    train - np.array - vector of training data

    val - np.array - vector of validation data

    horizon - list of ints, forecast horizon e.g. [7, 14, 28] days

    step -- step taken in cross validation 
            e.g. 1 in next cross validation training data includes next point 
            from the validation set.
            e.g. 7 in the next cross validation training data includes next 7 points
            (default=1)
            
    Returns:
    -------
    np.array - vector of forecast errors from the CVs.
    '''
    cvs = []

    #change here: max(horizons) + 1
    for i in range(0, len(val) - max(horizons) + 1, step):
        
        train_cv = np.concatenate([train, val[:i]], axis=0)
        model.fit(train_cv)
        
        #predict the maximum horizon 
        preds = model.predict(horizon=len(val[i:i+max(horizons)]))

        horizon_errors = []
        for h in horizons:
            #would be useful to return multiple prediction errors in one go.
            pred_error = error_func(preds[:h], val[i:i+h])
            horizon_errors.append(pred_error)
        
        cvs.append(horizon_errors)
    
    return np.array(cvs)