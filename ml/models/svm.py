import shap
import logging
from sklearn.svm import SVC


class SVM:
    
    def __init__(self, kernel='rbf', random_state=0):
        self._kernel = kernel
        self._random_state = random_state
        self.model = SVC(
            kernel=self._kernel, 
            random_state=self._random_state
        )
        self._logger = logging.getLogger(__name__)
    
    def score(self, x_train, y_train, env=None):
        acc_score = round(self.model.score(x_train, y_train) * 100, 2)
        if env == "vm":
            print(f"Accuracy SVM score: {acc_score}")
        else:
            self._logger.info(f"Accuracy SVM score: {acc_score}")
        
    def shap(self, x_train, x_test, feature_names, plot_type="bar"):
        explainer = shap.Explainer(self.model.predict, x_train)
        shap_values = explainer(x_test).abs
        shap.summary_plot(shap_values, x_test, plot_type=plot_type, feature_names=feature_names)

    def fit_predict(self, x_train, x_test, y_train):
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        return y_pred
    
    def set_params(self, **params):
        self.model.set_params(**params)
