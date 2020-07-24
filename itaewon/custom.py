import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class Custom():

    def __init__(self,X,y,regression,model):
        self.X = X
        self.y = y
        self.regression = regression
        self.model = model


    def custom(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=1)

        if self.regression == True:

            input_model = self.model.fit(X_train,y_train)
            rmse = np.sqrt(mean_squared_error(y_test,input_model.predict(X_test)))
            return print('RMSE for %s :'%self.model, rmse)


        else:

            input_model = self.model.fit(X_train,y_train)
            pred = input_model.predict(X_test)
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            return matrix,report
