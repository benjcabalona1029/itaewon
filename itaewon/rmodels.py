import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Load R packages
stats = importr('stats')
base = importr('base')




class RModels():

    def __init__(self,data,formula):

        self.data = data
        self.formula = formula


    def convert_r(self):

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data = ro.conversion.py2rpy(self.data)
        return r_data

    def linear_regression(self):
        model = stats.lm(self.formula, data=self.convert_r(),)
        return model



    def predict(self,model,new_data,show_interval=False,interval='confidence'):

        with localconverter(ro.default_converter + pandas2ri.converter):
            test_data = ro.conversion.py2rpy(new_data)

        if show_interval == False:

            preds = stats.predict(model,test_data)
            return pd.DataFrame(np.asarray(preds),columns=['Predicted'])

        else:

            if interval == 'confidence':
                preds = (stats.predict(model,test_data,interval='confidence'))
                return pd.DataFrame(np.asarray(preds),columns=['Predicted','Lower','Upper'])
            elif interval == 'prediction':
                preds = (stats.predict(model,test_data,interval='prediction'))
                return pd.DataFrame(np.asarray(preds),columns=['Predicted','Lower','Upper'])
            else:
                print('Error. Should be prediction interval or confidence interval')




    def infer_linear(self,model):
        result = base.summary(model)
        f = result.rx('fstatistic')[0]
        p_value = stats.pf(f[0],f[1],f[2],lower_tail=False)
        residuals = []
        for i in (dict(zip(result.names, list(result)))['residuals']):
            residuals.append(i)
        resid = pd.DataFrame()
        resid['Residuals'] = residuals
        with localconverter(ro.default_converter + pandas2ri.converter):
            residual = ro.conversion.py2rpy(resid)
        print(base.summary(residual))
        print(result.rx('coefficients'))
        print(result.rx('sigma'))
        print(result.rx('r.squared'))
        print(result.rx('adj.r.squared'))
        print('fstatistic',f[0])
        print('p_value',p_value)
