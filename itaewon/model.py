import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class Model():

    """
    Base Model. Provides performance of :

    - Linear Regression or Logistic Regression
    with default parameters. Implemented with sklearn.

    - Gradient Boosting. (Regressor or Classifier)
    implemented with xgboost

    Parameters
    ----------
    X : dataframe object
        feature set
    y : dataframe object or numpy series
        target variable
    regression: boolean
        indicating if solving for regression
        or classification

    """

    def __init__(self,X,y,regression):

        self.X = X
        self.y = y
        self.regression = regression



    def model(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=1)

        if self.regression == True:

            linear = LinearRegression().fit(X_train,y_train)
            rmse_linear = np.sqrt(mean_squared_error(y_test,linear.predict(X_test)))

            print('Linear Regression RMSE',rmse_linear)

            xgboost = xgb.XGBRegressor(objective='reg:squarederror').fit(X_train,y_train)
            rmse_xgboost = np.sqrt(mean_squared_error(y_test,xgboost.predict(X_test)))

            print('XGBOOST RMSE',rmse_xgboost)




            if rmse_linear > rmse_xgboost:
                difference = rmse_linear - rmse_xgboost
                return print('XGBOOST Performed Better with %s difference'%difference)
            else:
                difference = rmse_xgboost - rmse_linear
                return print('Linear Regression Performed Better with %s difference'%difference)

        else:
            linear = LogisticRegression().fit(X_train,y_train)
            linear_pred = linear.predict(X_test)
            linear_matrix = confusion_matrix(y_test,linear_pred)
            linear_report = classification_report(y_test,linear_pred)

            xgboost = xgb.XGBClassifier().fit(X_train,y_train)
            xgboost_pred = xgboost.predict(X_test)
            xgboost_matrix = confusion_matrix(y_test,xgboost_pred)
            xgboost_report = classification_report(y_test,xgboost_pred)

            print('Logistic Regression Result'), print(linear_matrix), print(linear_report),
            print('XGBOOST Result'), print(xgboost_matrix), print(xgboost_report)
