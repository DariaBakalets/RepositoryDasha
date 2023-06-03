```python
# let's import pandas library and set options to be able to view data right in the browser
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.style as style
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
style.use('fivethirtyeight')
import numpy as np
```


```python
rent_df_cleaned = pd.read_csv('cleaned_dataset.csv')
```


```python
rent_df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_day_exposition</th>
      <th>last_day_exposition</th>
      <th>last_price</th>
      <th>floor</th>
      <th>open_plan</th>
      <th>rooms</th>
      <th>studio</th>
      <th>area</th>
      <th>agent_fee</th>
      <th>renovation</th>
      <th>last_price_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-24T00:00:00+03:00</td>
      <td>2016-01-19T00:00:00+03:00</td>
      <td>20000.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28.0</td>
      <td>100.0</td>
      <td>3.0</td>
      <td>9.903488</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-17T00:00:00+03:00</td>
      <td>2016-03-04T00:00:00+03:00</td>
      <td>24000.0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>59.0</td>
      <td>100.0</td>
      <td>3.0</td>
      <td>10.085809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-11-17T00:00:00+03:00</td>
      <td>2016-04-24T00:00:00+03:00</td>
      <td>18000.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>100.0</td>
      <td>3.0</td>
      <td>9.798127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-02-04T00:00:00+03:00</td>
      <td>2016-02-28T00:00:00+03:00</td>
      <td>18000.0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>39.0</td>
      <td>90.0</td>
      <td>0.0</td>
      <td>9.798127</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-02-28T00:00:00+03:00</td>
      <td>2016-04-02T00:00:00+03:00</td>
      <td>19000.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>50.0</td>
      <td>11.0</td>
      <td>9.852194</td>
    </tr>
  </tbody>
</table>
</div>




```python
rent_df_cleaned.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 155391 entries, 0 to 155390
    Data columns (total 11 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   first_day_exposition  155391 non-null  object 
     1   last_day_exposition   155391 non-null  object 
     2   last_price            155391 non-null  float64
     3   floor                 155391 non-null  int64  
     4   open_plan             155391 non-null  int64  
     5   rooms                 155391 non-null  int64  
     6   studio                155391 non-null  int64  
     7   area                  155391 non-null  float64
     8   agent_fee             122840 non-null  float64
     9   renovation            155391 non-null  float64
     10  last_price_log        155391 non-null  float64
    dtypes: float64(5), int64(4), object(2)
    memory usage: 13.0+ MB



```python
rent_df_cleaned['renovation'] = rent_df_cleaned['renovation'].astype(int)
```


```python
rent_df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_day_exposition</th>
      <th>last_day_exposition</th>
      <th>last_price</th>
      <th>floor</th>
      <th>open_plan</th>
      <th>rooms</th>
      <th>studio</th>
      <th>area</th>
      <th>agent_fee</th>
      <th>renovation</th>
      <th>last_price_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-24T00:00:00+03:00</td>
      <td>2016-01-19T00:00:00+03:00</td>
      <td>20000.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>9.903488</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-17T00:00:00+03:00</td>
      <td>2016-03-04T00:00:00+03:00</td>
      <td>24000.0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>59.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>10.085809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-11-17T00:00:00+03:00</td>
      <td>2016-04-24T00:00:00+03:00</td>
      <td>18000.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>9.798127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-02-04T00:00:00+03:00</td>
      <td>2016-02-28T00:00:00+03:00</td>
      <td>18000.0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>39.0</td>
      <td>90.0</td>
      <td>0</td>
      <td>9.798127</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-02-28T00:00:00+03:00</td>
      <td>2016-04-02T00:00:00+03:00</td>
      <td>19000.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>50.0</td>
      <td>11</td>
      <td>9.852194</td>
    </tr>
  </tbody>
</table>
</div>




```python
rent_df_cleaned.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 155391 entries, 0 to 155390
    Data columns (total 11 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   first_day_exposition  155391 non-null  object 
     1   last_day_exposition   155391 non-null  object 
     2   last_price            155391 non-null  float64
     3   floor                 155391 non-null  int64  
     4   open_plan             155391 non-null  int64  
     5   rooms                 155391 non-null  int64  
     6   studio                155391 non-null  int64  
     7   area                  155391 non-null  float64
     8   agent_fee             122840 non-null  float64
     9   renovation            155391 non-null  int64  
     10  last_price_log        155391 non-null  float64
    dtypes: float64(4), int64(5), object(2)
    memory usage: 13.0+ MB



```python
rent_df_cleaned['first_day_exposition'] = pd.to_datetime(rent_df_cleaned.first_day_exposition)
```


```python
rent_df_cleaned['last_day_exposition'] = pd.to_datetime(rent_df_cleaned.last_day_exposition)
```


```python
rent_df_cleaned['offer_time'] = (rent_df_cleaned['last_day_exposition'] - rent_df_cleaned['first_day_exposition']).dt.days
```


```python
rent_df_cleaned.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_day_exposition</th>
      <th>last_day_exposition</th>
      <th>last_price</th>
      <th>floor</th>
      <th>open_plan</th>
      <th>rooms</th>
      <th>studio</th>
      <th>area</th>
      <th>agent_fee</th>
      <th>renovation</th>
      <th>last_price_log</th>
      <th>offer_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-24 00:00:00+03:00</td>
      <td>2016-01-19 00:00:00+03:00</td>
      <td>20000.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>9.903488</td>
      <td>360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-17 00:00:00+03:00</td>
      <td>2016-03-04 00:00:00+03:00</td>
      <td>24000.0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>59.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>10.085809</td>
      <td>108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-11-17 00:00:00+03:00</td>
      <td>2016-04-24 00:00:00+03:00</td>
      <td>18000.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>9.798127</td>
      <td>159</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-02-04 00:00:00+03:00</td>
      <td>2016-02-28 00:00:00+03:00</td>
      <td>18000.0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>39.0</td>
      <td>90.0</td>
      <td>0</td>
      <td>9.798127</td>
      <td>24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-02-28 00:00:00+03:00</td>
      <td>2016-04-02 00:00:00+03:00</td>
      <td>19000.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>50.0</td>
      <td>11</td>
      <td>9.852194</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>




```python
# select all offers added the first 3 months of 2018 as train dataset.
# '&' means 'and' and should be used when both conditions are satisfied
# pay attention that it's better always to put conditions in brackets to embrace the right priority of operations
train_df = rent_df_cleaned[(rent_df_cleaned.first_day_exposition >= '2018-01-01') 
                          & (rent_df_cleaned.first_day_exposition < '2018-04-01')]
```


```python
len(train_df)
```




    16974




```python
# select all offers added in april and may 2018 as test dataset.
test_df = rent_df_cleaned[(rent_df_cleaned.first_day_exposition >= '2018-04-01') 
                          & (rent_df_cleaned.first_day_exposition < '2018-06-01')]
```


```python
len(test_df)
```




    14974




```python
# let's use latest data from 2018-06-01 as a hodout dataset to simulate how algorithms would
# behave in production
holdout_df = rent_df_cleaned[rent_df_cleaned.first_day_exposition >= '2018-06-01']
```


```python
len(holdout_df)
```




    21112




```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_day_exposition</th>
      <th>last_day_exposition</th>
      <th>last_price</th>
      <th>floor</th>
      <th>open_plan</th>
      <th>rooms</th>
      <th>studio</th>
      <th>area</th>
      <th>agent_fee</th>
      <th>renovation</th>
      <th>last_price_log</th>
      <th>offer_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>2018-01-05 00:00:00+03:00</td>
      <td>2018-01-16 00:00:00+03:00</td>
      <td>26000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>10.165852</td>
      <td>11</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2018-01-20 00:00:00+03:00</td>
      <td>2018-02-28 00:00:00+03:00</td>
      <td>17500.0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>50.0</td>
      <td>1</td>
      <td>9.769956</td>
      <td>39</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2018-02-09 00:00:00+03:00</td>
      <td>2018-03-03 00:00:00+03:00</td>
      <td>16000.0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>90.0</td>
      <td>0</td>
      <td>9.680344</td>
      <td>22</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2018-03-19 00:00:00+03:00</td>
      <td>2018-04-18 00:00:00+03:00</td>
      <td>22000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>9.998798</td>
      <td>30</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018-03-26 00:00:00+03:00</td>
      <td>2018-03-30 00:00:00+03:00</td>
      <td>20000.0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>50.0</td>
      <td>0</td>
      <td>9.903488</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.drop(columns=['first_day_exposition','last_day_exposition','last_price_log', 'agent_fee'], inplace=True)
test_df.drop(columns=['first_day_exposition','last_day_exposition', 'last_price_log', 'agent_fee'], inplace=True)
```

    /opt/conda/lib/python3.9/site-packages/pandas/core/frame.py:4906: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().drop(



```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_price</th>
      <th>floor</th>
      <th>open_plan</th>
      <th>rooms</th>
      <th>studio</th>
      <th>area</th>
      <th>renovation</th>
      <th>offer_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>26000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>24</th>
      <td>17500.0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>1</td>
      <td>39</td>
    </tr>
    <tr>
      <th>25</th>
      <td>16000.0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>26</th>
      <td>22000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>27</th>
      <td>20000.0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import metrics
```


```python
X_train = train_df.drop('last_price', axis=1)
y_train = train_df['last_price']
X_valid = test_df.drop('last_price', axis=1)
y_valid = test_df['last_price']
y_train= y_train.values.reshape(-1,1)
y_valid= y_valid.values.reshape(-1,1)
print(X_train)
```

            floor  open_plan  rooms  studio  area  renovation  offer_time
    8          12          0      1       0  36.0           0          11
    24          9          0      1       0  32.0           1          39
    25          4          0      1       0  38.0           0          22
    26         12          0      1       0  32.0           0          30
    27          5          0      1       0  32.0           0           4
    ...       ...        ...    ...     ...   ...         ...         ...
    154884      2          0      1       0  46.0          11         143
    154921      6          0      2       0  50.0          11         176
    155028      4          0      3       0  66.0          11         165
    155079      4          0      2       0  52.0           0         132
    155271      1          0      2       0  46.0           0         193
    
    [16974 rows x 7 columns]



```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.fit_transform(X_valid)
y_train = sc_y.fit_transform(y_train)
y_valid = sc_y.fit_transform(y_valid)
```


```python
X_train.shape, y_train.shape
```




    ((16974, 7), (16974, 1))




```python
##Building random forest regressor
```


```python
import xgboost as xgb
```


```python
xgboost_model = xgb.XGBRegressor(objective="reg:linear", random_state=99)
# train the model on training dataset
xgboost_model.fit(X_train, y_train)
predictions = xgboost_model.predict(X_valid)
print('MAE:', metrics.mean_absolute_error(y_valid, predictions))
print('MSE:', metrics.mean_squared_error(y_valid, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))
```

    [16:27:04] WARNING: ../src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    MAE: 0.37168259015263216
    MSE: 0.446780861069617
    RMSE: 0.6684166822197192



```python
##Save the model, encoder and the scaler as a pipeline
```


```python
import joblib
```


```python
xgboost_model.save_model('model.json')
```


```python
model_file = 'model.pkl'
scaler_x = 'scaler_x.pkl'
scaler_y = 'scaler_y.pkl'
joblib.dump(sc_X, scaler_x)
joblib.dump(sc_y, scaler_y)
joblib.dump(xgboost_model, model_file)
```




    ['model.pkl']




```python
sc_y.inverse_transform(X_valid[0])
```




    array([14857.34699652, 28457.49686123, 20274.45194545, 21596.96683618,
           18402.66434523, 18500.37016272, 17792.62704576])


