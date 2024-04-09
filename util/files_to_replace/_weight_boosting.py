'''add the following function "_get_all_predict" in class AdaBoostRegressor of sklearn/ensemble/_weight_boosting.py'''

class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):

    def _boost(self, iboost, X, y, sample_weight, random_state):
        ''''''

    def _get_all_predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_]).T
        return predictions
    
    def _get_median_predict(self, X, limit):
        ''''''