## Lab 1.3 - Predicting Real Estate Data in St. Petersburg
We have data from Yandex.Realty classified https://realty.yandex.ru containing real estate listings for apartments in St. Petersburg and Leningrad Oblast from 2016 till the middle of August 2018. In this Lab you'll learn how to apply machine learning algorithms to solve business problems. Accurate price prediction can help to find fraudsters automatically and help Yandex.Realty users to make better decisions when buying and selling real estate.

Using python with machine learning algotithms is the #1 option for prototyping solutions among data scientists today. We'll take a look at it in this lab.

### Main objectives
After successful completion of the lab work students will be able to:
-	Apply machine learning for solving price prediction problem
-   Calculate metrics which can help us find out whether our machine learning model is ready for production

### Tasks
-	Encode dataset
-	Split dataset to train and validation datasets
-	Apply decision tree algorithm to build ML (machine learning) model for price predictions
-   Calculate metrics
-   Try other algorithms and factors to get a better solution 


### 1. Load data with real estate prices


```python
!python -m pip install scikit-learn --upgrade!pip install --upgrade pip
!pip install sklearn_pandas
```

    
    Usage:   
      /opt/conda/bin/python -m pip install [options] <requirement specifier> [package-index-options] ...
      /opt/conda/bin/python -m pip install [options] -r <requirements file> [package-index-options] ...
      /opt/conda/bin/python -m pip install [options] [-e] <vcs project url> ...
      /opt/conda/bin/python -m pip install [options] [-e] <local project path> ...
      /opt/conda/bin/python -m pip install [options] <archive url/path> ...
    
    no such option: --upgrade!pip
    Collecting sklearn_pandas
      Using cached sklearn_pandas-2.2.0-py2.py3-none-any.whl (10 kB)
    Requirement already satisfied: scipy>=1.5.1 in /opt/conda/lib/python3.9/site-packages (from sklearn_pandas) (1.7.1)
    Requirement already satisfied: pandas>=1.1.4 in /opt/conda/lib/python3.9/site-packages (from sklearn_pandas) (1.3.3)
    Requirement already satisfied: numpy>=1.18.1 in /opt/conda/lib/python3.9/site-packages (from sklearn_pandas) (1.20.3)
    Requirement already satisfied: scikit-learn>=0.23.0 in /opt/conda/lib/python3.9/site-packages (from sklearn_pandas) (0.24.2)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/lib/python3.9/site-packages (from pandas>=1.1.4->sklearn_pandas) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /opt/conda/lib/python3.9/site-packages (from pandas>=1.1.4->sklearn_pandas) (2021.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.9/site-packages (from scikit-learn>=0.23.0->sklearn_pandas) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/lib/python3.9/site-packages (from scikit-learn>=0.23.0->sklearn_pandas) (1.0.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.9/site-packages (from python-dateutil>=2.7.3->pandas>=1.1.4->sklearn_pandas) (1.15.0)
    Installing collected packages: sklearn_pandas
    Successfully installed sklearn_pandas-2.2.0
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip available: [0m[31;49m22.3.1[0m[39;49m -> [0m[32;49m23.1.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.style as style
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
style.use('fivethirtyeight')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
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


### Self-control stops
1. Compete with other teams to create the best solution. You can play with factors and algorithm parameters to come up with it.


```python
rent_df_cleaned['f_day_exposition'] = pd.to_datetime(rent_df_cleaned.first_day_exposition)
```


```python
rent_df_cleaned['l_day_exposition'] = pd.to_datetime(rent_df_cleaned.last_day_exposition)
```


```python
rent_df_cleaned['offer_time'] = (rent_df_cleaned['l_day_exposition'] - rent_df_cleaned['f_day_exposition']).dt.days
```


```python
rent_df_cleaned.drop(columns=['last_price_log'], inplace = True)
rent_df_cleaned.drop(columns=['f_day_exposition'], inplace = True)
rent_df_cleaned.drop(columns=['l_day_exposition'], inplace = True)
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
      <th>offer_time</th>
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
      <td>360</td>
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
      <td>108</td>
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
      <td>159</td>
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
      <td>24</td>
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
      <td>34</td>
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
     10  offer_time            155391 non-null  int64  
    dtypes: float64(4), int64(5), object(2)
    memory usage: 13.0+ MB



```python
#split dataset to train and test samples
train_df = rent_df_cleaned[(rent_df_cleaned.first_day_exposition >= '2018-01-01') 
                          & (rent_df_cleaned.first_day_exposition < '2018-04-01')]
```


```python
len(train_df)
```




    16974




```python
test_df = rent_df_cleaned[(rent_df_cleaned.first_day_exposition >= '2018-04-01') 
                          & (rent_df_cleaned.first_day_exposition < '2018-06-01')]
```


```python
len(test_df)
```




    14974




```python
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
      <th>offer_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>2018-01-05T00:00:00+03:00</td>
      <td>2018-01-16T00:00:00+03:00</td>
      <td>26000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2018-01-20T00:00:00+03:00</td>
      <td>2018-02-28T00:00:00+03:00</td>
      <td>17500.0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2018-02-09T00:00:00+03:00</td>
      <td>2018-03-03T00:00:00+03:00</td>
      <td>16000.0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>90.0</td>
      <td>0.0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2018-03-19T00:00:00+03:00</td>
      <td>2018-04-18T00:00:00+03:00</td>
      <td>22000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2018-03-26T00:00:00+03:00</td>
      <td>2018-03-30T00:00:00+03:00</td>
      <td>20000.0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>32.0</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.drop(columns=['first_day_exposition','last_day_exposition'], inplace=True)
test_df.drop(columns=['first_day_exposition','last_day_exposition'], inplace=True)
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
      <th>agent_fee</th>
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
      <td>NaN</td>
      <td>0.0</td>
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
      <td>50.0</td>
      <td>1.0</td>
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
      <td>90.0</td>
      <td>0.0</td>
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
      <td>NaN</td>
      <td>0.0</td>
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
      <td>50.0</td>
      <td>0.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = train_df.drop('last_price', axis=1)
y_train = train_df['last_price']
X_valid = test_df.drop('last_price', axis=1)
y_valid = test_df['last_price']
y_train= y_train.values.reshape(-1,1)
y_valid= y_valid.values.reshape(-1,1)
print(X_train)
```

            floor  open_plan  rooms  studio  area  agent_fee  renovation  offer_time
    8          12          0      1       0  36.0        NaN         0.0          11
    24          9          0      1       0  32.0       50.0         1.0          39
    25          4          0      1       0  38.0       90.0         0.0          22
    26         12          0      1       0  32.0        NaN         0.0          30
    27          5          0      1       0  32.0       50.0         0.0           4
    ...       ...        ...    ...     ...   ...        ...         ...         ...
    154884      2          0      1       0  46.0        0.0        11.0         143
    154921      6          0      2       0  50.0       90.0        11.0         176
    155028      4          0      3       0  66.0        0.0        11.0         165
    155079      4          0      2       0  52.0      100.0         0.0         132
    155271      1          0      2       0  46.0       50.0         0.0         193
    
    [16974 rows x 8 columns]



```python
numeric_features = ['area', 'offer_time', 'agent_fee' ] 
nominal_features = ['renovation','open_plan', 'rooms', 'floor', 'studio']
```


```python
mapper = DataFrameMapper([([feature], SimpleImputer()) for feature in numeric_features] +\
                         [([feature], OneHotEncoder(handle_unknown='ignore')) for feature in nominal_features],
                             df_out=True)

pipeline = Pipeline(steps = [('preprocessing', mapper), 
                             ('scaler', StandardScaler()),
                             ('LinearRegression', LinearRegression())])
                             
pipeline
```




    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['studio'],
                                                OneHotEncoder(handle_unknown='ignore'))])),
                    ('scaler', StandardScaler()),
                    ('LinearRegression', LinearRegression())])




```python
result = mapper.fit_transform(rent_df_cleaned)
result.head()
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
      <th>area</th>
      <th>offer_time</th>
      <th>agent_fee</th>
      <th>renovation_x0_0.0</th>
      <th>renovation_x0_1.0</th>
      <th>renovation_x0_2.0</th>
      <th>renovation_x0_3.0</th>
      <th>renovation_x0_4.0</th>
      <th>renovation_x0_5.0</th>
      <th>renovation_x0_6.0</th>
      <th>renovation_x0_7.0</th>
      <th>renovation_x0_8.0</th>
      <th>renovation_x0_10.0</th>
      <th>renovation_x0_11.0</th>
      <th>open_plan_x0_0</th>
      <th>open_plan_x0_1</th>
      <th>rooms_x0_0</th>
      <th>rooms_x0_1</th>
      <th>rooms_x0_2</th>
      <th>rooms_x0_3</th>
      <th>rooms_x0_4</th>
      <th>rooms_x0_5</th>
      <th>floor_x0_1</th>
      <th>floor_x0_2</th>
      <th>floor_x0_3</th>
      <th>floor_x0_4</th>
      <th>floor_x0_5</th>
      <th>floor_x0_6</th>
      <th>floor_x0_7</th>
      <th>floor_x0_8</th>
      <th>floor_x0_9</th>
      <th>floor_x0_10</th>
      <th>floor_x0_11</th>
      <th>floor_x0_12</th>
      <th>floor_x0_13</th>
      <th>floor_x0_14</th>
      <th>floor_x0_15</th>
      <th>floor_x0_16</th>
      <th>floor_x0_17</th>
      <th>floor_x0_18</th>
      <th>floor_x0_19</th>
      <th>floor_x0_20</th>
      <th>floor_x0_21</th>
      <th>floor_x0_22</th>
      <th>floor_x0_23</th>
      <th>floor_x0_24</th>
      <th>floor_x0_25</th>
      <th>floor_x0_26</th>
      <th>floor_x0_27</th>
      <th>floor_x0_28</th>
      <th>floor_x0_29</th>
      <th>floor_x0_30</th>
      <th>floor_x0_32</th>
      <th>floor_x0_33</th>
      <th>floor_x0_34</th>
      <th>floor_x0_35</th>
      <th>floor_x0_36</th>
      <th>studio_x0_0</th>
      <th>studio_x0_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28.0</td>
      <td>360.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59.0</td>
      <td>108.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36.0</td>
      <td>159.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39.0</td>
      <td>24.0</td>
      <td>90.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.0</td>
      <td>34.0</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pipeline.fit(X_train, y_train)
```




    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['studio'],
                                                OneHotEncoder(handle_unknown='ignore'))])),
                    ('scaler', StandardScaler()),
                    ('LinearRegression', LinearRegression())])




```python
train_pred = pipeline.predict(X_train)
rmse = mean_squared_error(y_true=y_train, y_pred = train_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_train, y_pred=train_pred)
accuracy = pipeline.score(X_train,y_train)


print (f'LR RMSE train = {round(rmse, 3)}')
print (f'LR MAPE train = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    LR RMSE train = 10309.222
    LR MAPE train = 0.22
    Accuracy =  59.90070722938139 %



```python
test_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_true=y_valid, y_pred = test_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_valid, y_pred=test_pred)
accuracy = pipeline.score(X_valid,y_valid)

print (f'LR RMSE test = {round(rmse, 3)}')
print (f'LR MAPE test = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    LR RMSE test = 11318.566
    LR MAPE test = 0.23
    Accuracy =  61.11019540630518 %



```python
mapper = DataFrameMapper([([feature], SimpleImputer()) for feature in numeric_features] +\
                         [([feature], OneHotEncoder(handle_unknown='ignore')) for feature in nominal_features],
                             df_out=True)

pipeline = Pipeline(steps = [('preprocessing', mapper), 
                             ('scaler', StandardScaler()),
                             ('decision_tree', DecisionTreeRegressor(max_depth=10, min_samples_leaf=8, max_features=4))])
                             
pipeline
```




    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['studio'],
                                                OneHotEncoder(handle_unknown='ignore'))])),
                    ('scaler', StandardScaler()),
                    ('decision_tree',
                     DecisionTreeRegressor(max_depth=10, max_features=4,
                                           min_samples_leaf=8))])




```python
pipeline.fit(X_train, y_train)
```




    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['studio'],
                                                OneHotEncoder(handle_unknown='ignore'))])),
                    ('scaler', StandardScaler()),
                    ('decision_tree',
                     DecisionTreeRegressor(max_depth=10, max_features=4,
                                           min_samples_leaf=8))])




```python
train_pred = pipeline.predict(X_train)
rmse = mean_squared_error(y_true=y_train, y_pred = train_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_train, y_pred=train_pred)
accuracy = pipeline.score(X_train,y_train)


print (f'RMSE train = {round(rmse, 3)}')
print (f'MAPE train = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    RMSE train = 10654.852
    MAPE train = 0.208
    Accuracy =  57.16687748796476 %



```python
test_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_true=y_valid, y_pred = test_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_valid, y_pred=test_pred)
accuracy = pipeline.score(X_valid,y_valid)

print (f'RMSE test = {round(rmse, 3)}')
print (f'MAPE test = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    RMSE test = 12055.573
    MAPE test = 0.218
    Accuracy =  55.880697614429565 %



```python
mapper = DataFrameMapper([([feature], SimpleImputer()) for feature in numeric_features] +\
                         [([feature], OneHotEncoder(handle_unknown='ignore')) for feature in nominal_features],
                             df_out=True)

pipeline = Pipeline(steps = [('preprocessing', mapper), 
                             ('scaler', StandardScaler()),
                             ('random_forest', RandomForestRegressor(random_state=99))])
pipeline
```




    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['studio'],
                                                OneHotEncoder(handle_unknown='ignore'))])),
                    ('scaler', StandardScaler()),
                    ('random_forest', RandomForestRegressor(random_state=99))])




```python
pipeline.fit(X_train, y_train)
```

    /opt/conda/lib/python3.9/site-packages/sklearn/pipeline.py:346: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      self._final_estimator.fit(Xt, y, **fit_params_last_step)





    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['studio'],
                                                OneHotEncoder(handle_unknown='ignore'))])),
                    ('scaler', StandardScaler()),
                    ('random_forest', RandomForestRegressor(random_state=99))])




```python
train_pred = pipeline.predict(X_train)
rmse = mean_squared_error(y_true=y_train, y_pred = train_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_train, y_pred=train_pred)
accuracy = pipeline.score(X_train,y_train)


print (f'RMSE train = {round(rmse, 3)}')
print (f'MAPE train = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    RMSE train = 4056.847
    MAPE train = 0.08
    Accuracy =  93.7904206471246 %



```python
test_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_true=y_valid, y_pred = test_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_valid, y_pred=test_pred)
accuracy = pipeline.score(X_valid,y_valid)

print (f'RMSE test = {round(rmse, 3)}')
print (f'MAPE test = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    RMSE test = 11431.507
    MAPE test = 0.22
    Accuracy =  60.33021500917766 %



```python
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
mapper = DataFrameMapper([([feature], SimpleImputer()) for feature in numeric_features] +\
                         [([feature], OneHotEncoder(handle_unknown='ignore')) for feature in nominal_features],
                             df_out=True)

pipeline = Pipeline(steps = [('preprocessing', mapper), 
                             ('scaler', StandardScaler()),
                             ('xgb', xgb.XGBRegressor(objective="reg:linear", random_state=99))])
pipeline
```




    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                O...
                                  grow_policy=None, importance_type=None,
                                  interaction_constraints=None, learning_rate=None,
                                  max_bin=None, max_cat_threshold=None,
                                  max_cat_to_onehot=None, max_delta_step=None,
                                  max_depth=None, max_leaves=None,
                                  min_child_weight=None, missing=nan,
                                  monotone_constraints=None, n_estimators=100,
                                  n_jobs=None, num_parallel_tree=None,
                                  objective='reg:linear', predictor=None, ...))])




```python
pipeline.fit(X_train, y_train)
```

    [15:55:11] WARNING: ../src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.





    Pipeline(steps=[('preprocessing',
                     DataFrameMapper(df_out=True, drop_cols=[],
                                     features=[(['area'], SimpleImputer()),
                                               (['offer_time'], SimpleImputer()),
                                               (['agent_fee'], SimpleImputer()),
                                               (['renovation'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['open_plan'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['rooms'],
                                                OneHotEncoder(handle_unknown='ignore')),
                                               (['floor'],
                                                O...
                                  feature_types=None, gamma=0, gpu_id=-1,
                                  grow_policy='depthwise', importance_type=None,
                                  interaction_constraints='',
                                  learning_rate=0.300000012, max_bin=256,
                                  max_cat_threshold=64, max_cat_to_onehot=4,
                                  max_delta_step=0, max_depth=6, max_leaves=0,
                                  min_child_weight=1, missing=nan,
                                  monotone_constraints='()', n_estimators=100,
                                  n_jobs=0, num_parallel_tree=1,
                                  objective='reg:linear', predictor='auto', ...))])




```python
train_pred = pipeline.predict(X_train)
rmse = mean_squared_error(y_true=y_train, y_pred = train_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_train, y_pred=train_pred)
accuracy = pipeline.score(X_train,y_train)


print (f'RMSE train = {round(rmse, 3)}')
print (f'MAPE train = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    RMSE train = 6548.027
    MAPE train = 0.162
    Accuracy =  83.8227061344956 %



```python
test_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_true=y_valid, y_pred = test_pred, squared=False)
mape = mean_absolute_percentage_error(y_true=y_valid, y_pred=test_pred)
accuracy = pipeline.score(X_valid,y_valid)

print (f'RMSE test = {round(rmse, 3)}')
print (f'MAPE test = {round(mape, 3)}')
print('Accuracy = ', accuracy*100,'%')
```

    RMSE test = 11262.311
    MAPE test = 0.204
    Accuracy =  61.49581308774685 %



```python
test_pred
```




    array([21045.004, 25907.129, 51685.703, ..., 37575.45 , 20018.14 ,
           20862.496], dtype=float32)




```python
import joblib
```


```python
model_file = 'model.pkl'

joblib.dump(pipeline, model_file)
```




    ['model.pkl']




```python

```
