from metrics.tsp import mi_assessment as tsp
from sklearn.metrics import mean_squared_error,r2_score

def test(args,arrays):
    y_min, y_max = args.get('y_min', None), args.get('y_max', None)
    if y_min is not None and y_max is not None:
        def rescale(y):
            return y * (y_max - y_min) + y_min
    else:
        def rescale(y):
            return y
    y = rescale(arrays['y_test'])
    y = y.reshape(-1,1)
    prediction = rescale(arrays['prediction'])
    prediction = prediction.reshape(-1,1)
    mse = mean_squared_error(y, prediction)
    r2 = r2_score(y, prediction)
    mi = tsp(arrays['x_test'], y, prediction)
    return mse, mi, r2