from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def metriques(nom, y_vrai, y_pred):
    y_vrai = np.array(y_vrai)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_vrai, y_pred)
    rmse = np.sqrt(mean_squared_error(y_vrai, y_pred))
    mask = y_vrai != 0
    wmape = np.sum(np.abs(y_vrai[mask] - y_pred[mask])) / np.sum(np.abs(y_vrai[mask]))
    return {"Split": nom, "MAE": mae, "RMSE": rmse, "wMAPE": wmape}
