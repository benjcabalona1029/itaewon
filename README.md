# Itaewon

[![Downloads](https://pepy.tech/badge/itaewon)](https://pepy.tech/project/itaewon)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/benjcabalona1029/itaewon/blob/master/LICENSE)

My personal python library to speed-up my workflow.

Also, i found myself switching to R and Python quite frequently. This is my attempt to minimize that.

The implementation is quite simple, since i'm simply running `R` scripts in the background.

Sample usage is shown below.



# Installation


```python
# !pip install itaewon==0.1.4
```


```python
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
```


```python
data = pd.DataFrame()
error = np.random.normal(0,5,300)
data['X'] = np.random.normal(5,2.6,300)
data['Z'] = np.random.binomial(1,0.3,300)
data['Y'] = 10*data['X'] + 2*data['Z'] + error
```


# Testing RModels


```python
from itaewon.rmodels import RModels
```


```python
formula = 'Y ~ X + Z'
rmodel = RModels(data,formula)
model = rmodel.linear_regression()
rmodel.infer_linear(model)
```

       Residuals       
     Min.   :-14.6393  
     1st Qu.: -3.4848  
     Median : -0.1583  
     Mean   :  0.0000  
     3rd Qu.:  3.3146  
     Max.   : 16.2795  

    $coefficients
                 Estimate Std. Error   t value      Pr(>|t|)
    (Intercept) -1.109010  0.6718716 -1.650627  9.987172e-02
    X           10.163182  0.1150433 88.342270 2.862585e-215
    Z            1.426163  0.6222261  2.292033  2.260318e-02


    $sigma
    [1] 4.922223


    $r.squared
    [1] 0.9634044


    $adj.r.squared
    [1] 0.963158


    fstatistic 3909.3683074439355
    p_value [1] 4.668023e-214




```python
new_data = pd.DataFrame()
new_data['X'] = [0,1,2,3]
new_data['Z'] = [1,0,0,1]
```


```python
rmodel.predict(model,new_data,True,'confidence')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
      <th>Lower</th>
      <th>Upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.317153</td>
      <td>-1.233912</td>
      <td>1.868218</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.054173</td>
      <td>7.921668</td>
      <td>10.186677</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.217355</td>
      <td>18.258526</td>
      <td>20.176184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30.806700</td>
      <td>29.671831</td>
      <td>31.941568</td>
    </tr>
  </tbody>
</table>
</div>




```python
rmodel.predict(model,new_data,True,'prediction')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
      <th>Lower</th>
      <th>Upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.317153</td>
      <td>-9.493093</td>
      <td>10.127399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.054173</td>
      <td>-0.698657</td>
      <td>18.807003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.217355</td>
      <td>9.483164</td>
      <td>28.951546</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30.806700</td>
      <td>21.053595</td>
      <td>40.559804</td>
    </tr>
  </tbody>
</table>
</div>




```python
rmodel.predict(model,new_data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.317153</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.054173</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.217355</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30.806700</td>
    </tr>
  </tbody>
</table>
</div>

# Testing the custom module


```python
from itaewon.custom import Custom
```


```python
from sklearn.linear_model import LinearRegression, LogisticRegression
```


```python
lr = LinearRegression()
```


```python
custom = Custom(data[['X','Z']],data['Y'],True,lr)
custom.custom()
```

    RMSE for LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False) : 4.752017508515921



```python
lr = LogisticRegression()
```


```python
custom = Custom(data[['X','Y']],data['Z'],False,lr)
print(custom.custom()[0])
print(custom.custom()[1])
```

    [[70  0]
     [29  0]]
                  precision    recall  f1-score   support

               0       0.71      1.00      0.83        70
               1       0.00      0.00      0.00        29

        accuracy                           0.71        99
       macro avg       0.35      0.50      0.41        99
    weighted avg       0.50      0.71      0.59        99
