from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


class ErrorScores:
    def __init__(self, model_name):
        self.model_name = model_name
        self.scores = {}

    def evaluate(self, actual, prediction, is_nn=False, is_lstm=False): #TODO
        mean_prediction = prediction.mean()
        if is_nn:
            mean_actual = (actual.mean()).numpy()
        else:
            mean_actual = actual.mean()

        self.scores['EVS'] = round(explained_variance_score(actual, prediction), 4)
        self.scores['MSE'] = round(mean_squared_error(actual, prediction), 4)
        self.scores['MAE'] = round(mean_absolute_error(actual, prediction), 4)
        self.scores['R2'] = round(r2_score(actual, prediction), 4)

        self.scores['Annual Mean WS'] = round(mean_prediction, 4)

        self.scores['Error WS [m/s]'] = round(abs(mean_prediction - mean_actual), 4)
        self.scores['Error WS [%]'] = round(100 * abs(mean_prediction - mean_actual)/mean_actual, 4)
