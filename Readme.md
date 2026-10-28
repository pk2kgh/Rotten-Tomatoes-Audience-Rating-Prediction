<h1 align="center" style="font-weight: bold; font-size: 48px;">Rotten Tomatoes - Audience Rating Prediction</h1>



<img src="https://www.rottentomatoes.com/assets/pizza-pie/images/rottentomatoes_logo_40.336d6fe66ff.png" alt="Rotten Tomatoes">


## **Author**

Pawan Kumar - (for feedback reach out to pwkr1992@gmail.com)

# **Introduction**
## Objective:
1. Preprocess and transform data, including handling missing values and encoding features.
2. Build and evaluate multiple regression models to predict audience ratings.
    - Linear Regression (Ridge & Lasso)
    - XGBoost Regressor
    - CatBoost Regressor
    - Neural Networks
3. Compare model performance using metrics like RMSE and R² to identify the best model.

# **Importing the required libraries**


```python
!pip install catboost
```

    Collecting catboost
      Downloading catboost-1.2.10-cp313-cp313-manylinux2014_x86_64.whl.metadata (1.4 kB)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.13/dist-packages (from catboost) (0.21)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.13/dist-packages (from catboost) (3.10.0)
    Requirement already satisfied: numpy<3.0,>=1.16.0 in /usr/local/lib/python3.13/dist-packages (from catboost) (2.1.3)
    Requirement already satisfied: pandas<4.0,>=0.24 in /usr/local/lib/python3.13/dist-packages (from catboost) (2.2.3)
    Requirement already satisfied: scipy in /usr/local/lib/python3.13/dist-packages (from catboost) (1.16.3)
    Requirement already satisfied: plotly in /usr/local/lib/python3.13/dist-packages (from catboost) (5.24.1)
    Requirement already satisfied: six in /usr/local/lib/python3.13/dist-packages (from catboost) (1.17.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas<4.0,>=0.24->catboost) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas<4.0,>=0.24->catboost) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas<4.0,>=0.24->catboost) (2026.5)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (1.4.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (4.67.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (1.5.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (26.3)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.13/dist-packages (from matplotlib->catboost) (3.3.3)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.13/dist-packages (from plotly->catboost) (9.1.4)
    Downloading catboost-1.2.10-cp313-cp313-manylinux2014_x86_64.whl (97.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m97.1/97.1 MB[0m [31m50.8 MB/s[0m  [33m0:00:01[0m
    [?25hInstalling collected packages: catboost
    Successfully installed catboost-1.2.10



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from textblob import TextBlob
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

# **Data Loading**


```python
# loading the dataset as a pandas dataframe
rt_df = pd.read_csv("Rotten_Tomatoes_Movies.csv", encoding='latin-1')
print(f"The Rotten Tomatoes Dataset contains {rt_df.shape[0]} rows and {rt_df.shape[1]} columns")
```

    The Rotten Tomatoes Dataset contains 16638 rows and 16 columns



```python
# looking at the dataset
rt_df.head()
```





  <div id="df-29917d4d-92d9-4f82-aeb0-a6f2a963da3e" class="colab-df-container">
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
      <th>movie_title</th>
      <th>movie_info</th>
      <th>critics_consensus</th>
      <th>rating</th>
      <th>genre</th>
      <th>directors</th>
      <th>writers</th>
      <th>cast</th>
      <th>in_theaters_date</th>
      <th>on_streaming_date</th>
      <th>runtime_in_minutes</th>
      <th>studio_name</th>
      <th>tomatometer_status</th>
      <th>tomatometer_rating</th>
      <th>tomatometer_count</th>
      <th>audience_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>A teenager discovers he's the descendant of a ...</td>
      <td>Though it may seem like just another Harry Pot...</td>
      <td>PG</td>
      <td>Action &amp; Adventure, Comedy, Drama, Science Fic...</td>
      <td>Chris Columbus</td>
      <td>Craig Titley</td>
      <td>Logan Lerman, Brandon T. Jackson, Alexandra Da...</td>
      <td>12/02/2010</td>
      <td>29/06/2010</td>
      <td>83.0</td>
      <td>20th Century Fox</td>
      <td>Rotten</td>
      <td>49</td>
      <td>144</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Please Give</td>
      <td>Kate has a lot on her mind. There's the ethics...</td>
      <td>Nicole Holofcener's newest might seem slight i...</td>
      <td>R</td>
      <td>Comedy</td>
      <td>Nicole Holofcener</td>
      <td>Nicole Holofcener</td>
      <td>Catherine Keener, Amanda Peet, Oliver Platt, R...</td>
      <td>30/04/2010</td>
      <td>19/10/2010</td>
      <td>90.0</td>
      <td>Sony Pictures Classics</td>
      <td>Certified Fresh</td>
      <td>86</td>
      <td>140</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>Blake Edwards' 10 stars Dudley Moore as George...</td>
      <td>NaN</td>
      <td>R</td>
      <td>Comedy, Romance</td>
      <td>Blake Edwards</td>
      <td>Blake Edwards</td>
      <td>Dudley Moore, Bo Derek, Julie Andrews, Robert ...</td>
      <td>05/10/1979</td>
      <td>27/08/1997</td>
      <td>118.0</td>
      <td>Waner Bros.</td>
      <td>Fresh</td>
      <td>68</td>
      <td>22</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12 Angry Men (Twelve Angry Men)</td>
      <td>A Puerto Rican youth is on trial for murder, a...</td>
      <td>Sidney Lumet's feature debut is a superbly wri...</td>
      <td>NR</td>
      <td>Classics, Drama</td>
      <td>Sidney Lumet</td>
      <td>Reginald Rose</td>
      <td>Martin Balsam, John Fiedler, Lee J. Cobb, E.G....</td>
      <td>13/04/1957</td>
      <td>06/03/2001</td>
      <td>95.0</td>
      <td>Criterion Collection</td>
      <td>Certified Fresh</td>
      <td>100</td>
      <td>51</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20,000 Leagues Under The Sea</td>
      <td>This 1954 Disney version of Jules Verne's 20,0...</td>
      <td>One of Disney's finest live-action adventures,...</td>
      <td>G</td>
      <td>Action &amp; Adventure, Drama, Kids &amp; Family</td>
      <td>Richard Fleischer</td>
      <td>Earl Felton</td>
      <td>James Mason, Kirk Douglas, Paul Lukas, Peter L...</td>
      <td>01/01/1954</td>
      <td>20/05/2003</td>
      <td>127.0</td>
      <td>Disney</td>
      <td>Fresh</td>
      <td>89</td>
      <td>27</td>
      <td>74.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-29917d4d-92d9-4f82-aeb0-a6f2a963da3e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-29917d4d-92d9-4f82-aeb0-a6f2a963da3e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-29917d4d-92d9-4f82-aeb0-a6f2a963da3e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# looking at the schema of the dataset
rt_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16638 entries, 0 to 16637
    Data columns (total 16 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   movie_title         16638 non-null  object 
     1   movie_info          16614 non-null  object 
     2   critics_consensus   8309 non-null   object 
     3   rating              16638 non-null  object 
     4   genre               16621 non-null  object 
     5   directors           16524 non-null  object 
     6   writers             15289 non-null  object 
     7   cast                16354 non-null  object 
     8   in_theaters_date    15823 non-null  object 
     9   on_streaming_date   16636 non-null  object 
     10  runtime_in_minutes  16483 non-null  float64
     11  studio_name         16222 non-null  object 
     12  tomatometer_status  16638 non-null  object 
     13  tomatometer_rating  16638 non-null  int64  
     14  tomatometer_count   16638 non-null  int64  
     15  audience_rating     16386 non-null  float64
    dtypes: float64(2), int64(2), object(12)
    memory usage: 2.0+ MB



```python
# converting date time
rt_df['in_theaters_date'] = pd.to_datetime(rt_df['in_theaters_date'], format='mixed')
rt_df['on_streaming_date'] = pd.to_datetime(rt_df['on_streaming_date'], format='mixed')
```

# **Handling Duplicates**

## Handling Duplicates in Rows


```python
# checking for duplicates in the dataset
rt_df.duplicated().max()
```




    np.True_



**There are duplicate rows present in the datset**


```python
# Checking for the count of duplicate rows
rt_df.duplicated().sum()
```




    np.int64(1)



**There is only 1 duplicated row**


```python
# viewing the duplicated rows
rt_df[rt_df.duplicated(keep=False)]
```





  <div id="df-fd57d031-74c1-48ea-870b-0f6b7d830dbe" class="colab-df-container">
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
      <th>movie_title</th>
      <th>movie_info</th>
      <th>critics_consensus</th>
      <th>rating</th>
      <th>genre</th>
      <th>directors</th>
      <th>writers</th>
      <th>cast</th>
      <th>in_theaters_date</th>
      <th>on_streaming_date</th>
      <th>runtime_in_minutes</th>
      <th>studio_name</th>
      <th>tomatometer_status</th>
      <th>tomatometer_rating</th>
      <th>tomatometer_count</th>
      <th>audience_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8495</th>
      <td>King Charles III</td>
      <td>An adaptation of the Broadway drama about Prin...</td>
      <td>NaN</td>
      <td>NR</td>
      <td>Drama</td>
      <td>Rupert Goold</td>
      <td>Mike Bartlett</td>
      <td>Oliver Chris, Richard Goulding, Charlotte Rile...</td>
      <td>2017-05-14</td>
      <td>2017-06-27</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>Fresh</td>
      <td>100</td>
      <td>9</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>8496</th>
      <td>King Charles III</td>
      <td>An adaptation of the Broadway drama about Prin...</td>
      <td>NaN</td>
      <td>NR</td>
      <td>Drama</td>
      <td>Rupert Goold</td>
      <td>Mike Bartlett</td>
      <td>Oliver Chris, Richard Goulding, Charlotte Rile...</td>
      <td>2017-05-14</td>
      <td>2017-06-27</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>Fresh</td>
      <td>100</td>
      <td>9</td>
      <td>48.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fd57d031-74c1-48ea-870b-0f6b7d830dbe')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-fd57d031-74c1-48ea-870b-0f6b7d830dbe button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fd57d031-74c1-48ea-870b-0f6b7d830dbe');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# dropping the duplicated row
rt_df = rt_df.drop_duplicates()
```


```python
# verifying for duplicates
rt_df.duplicated().max()
```




    np.False_



**No duplicate rows are present in the dataset**

# **Exploring the Dataset**


```python
# extracting numerical and categorical/Textual columns
txt_col = []
num_col = []
for col in rt_df.columns:
    if rt_df[col].dtype == 'O':
        txt_col.append(col)
    else:
        num_col.append(col)
```


```python
print(f"Categorical/Textual columns: {txt_col}")
print(f"Numerical columns: {num_col}")
```

    Categorical/Textual columns: ['movie_title', 'movie_info', 'critics_consensus', 'rating', 'genre', 'directors', 'writers', 'cast', 'in_theaters_date', 'on_streaming_date', 'studio_name', 'tomatometer_status']
    Numerical columns: ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'audience_rating']


## Exploring Numerical Columns


```python
# visulaizing correlation between numerical columns
plt.figure(figsize = (16, 10))
sns.heatmap(rt_df[[col for col in num_col]].corr(), annot = True, fmt = '.2f')
plt.show()
```


    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_26_0.png)
    


**Observation**

1. High correlation between **tomatometer_rating** & **audience_rating**
2. Negligible correlation between other features/columns with **audience_rating**

## Exploring Categorical/Textual Columns


```python
txt_status = {'Feature':[], 'Unique Values':[] }
for col in txt_col:
    txt_status['Feature'].append(col)
    txt_status['Unique Values'].append(len(rt_df[col].dropna().unique()))
pd.DataFrame(txt_status)
```





  <div id="df-02208d02-16ee-4a35-bc08-f97cc74bbdbb" class="colab-df-container">
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
      <th>Feature</th>
      <th>Unique Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>movie_title</td>
      <td>16106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movie_info</td>
      <td>16613</td>
    </tr>
    <tr>
      <th>2</th>
      <td>critics_consensus</td>
      <td>8307</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rating</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>genre</td>
      <td>1080</td>
    </tr>
    <tr>
      <th>5</th>
      <td>directors</td>
      <td>8314</td>
    </tr>
    <tr>
      <th>6</th>
      <td>writers</td>
      <td>12121</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cast</td>
      <td>16326</td>
    </tr>
    <tr>
      <th>8</th>
      <td>studio_name</td>
      <td>2886</td>
    </tr>
    <tr>
      <th>9</th>
      <td>tomatometer_status</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-02208d02-16ee-4a35-bc08-f97cc74bbdbb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-02208d02-16ee-4a35-bc08-f97cc74bbdbb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-02208d02-16ee-4a35-bc08-f97cc74bbdbb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
rt_df['rating'].unique()
```




    array(['PG', 'R', 'NR', 'G', 'PG-13', 'PG-13)', 'NC17', 'R)'],
          dtype=object)



**Observation**

From above in Distribution of Rating we can see that 'PG-13)' and 'R)' has typo in them, lets fix them.


```python
# fixing typo in dataset
rt_df['rating']= rt_df['rating'].replace({'PG-13)': 'PG-13', 'R)': 'R'})
```


```python
rt_df['rating'].unique()
```




    array(['PG', 'R', 'NR', 'G', 'PG-13', 'NC17'], dtype=object)




```python
rt_df['tomatometer_status'].unique()
```




    array(['Rotten', 'Certified Fresh', 'Fresh'], dtype=object)



**Observation**

It can be observed that except 'rating' & 'tomatometer_status' remaining columns have too many unique values with suggests that we consider them as textual columns while 'rating' & 'tomatometer_status' are considered as categorical columns


```python
# Visualizing categorical columns 'rating' & 'tomatometer_status'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(rt_df, x='rating', hue='rating', ax=ax1)
ax1.set_title('Distribution of Rating', fontsize=14)

sns.histplot(rt_df, x='tomatometer_status', hue='tomatometer_status', ax=ax2)
ax2.set_title('Distribution of Tomatometer Status', fontsize=14)

fig.suptitle('Distribution of Categorical Data', fontsize=16)
plt.tight_layout()
plt.show()

```


    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_36_0.png)
    


# **Handling missing values**

## Handling missing values in Target variable


```python
# checking for missing values in target variable 'audience_rating'
print(f"No of missing values in audience_rating: {rt_df['audience_rating'].isna().sum()}")
```

    No of missing values in audience_rating: 252


**Observation**

There are 252 missing values in audience_rating. Since it is the target trying to fill these missing values will introduce bias into the model. Hence we will drop the rows with missing values in audience_rating


```python
# dropping columns with missing values in audience_rating
rt_df = rt_df.dropna(subset=['audience_rating'])
```


```python
# verifying
print(f"No of missing values in audience_rating: {rt_df['audience_rating'].isna().sum()}")
```

    No of missing values in audience_rating: 0


**Missing values in audience_rating has been handled (Target)**

## Handling missing values in numerical columns


```python
# total missing values in numerical columns
pd.concat([rt_df[num_col].count(), rt_df[num_col].isna().sum(), round((rt_df[num_col].isna().sum()/len(rt_df))*100, 2)],
          axis = 1,
          keys = ["Total Count of Values", "Total Missing Values", "Percent of Missing Values"])
```





  <div id="df-37597162-5ac4-4387-adec-bf038e38c187" class="colab-df-container">
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
      <th>Total Count of Values</th>
      <th>Total Missing Values</th>
      <th>Percent of Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>in_theaters_date</th>
      <td>15666</td>
      <td>719</td>
      <td>4.39</td>
    </tr>
    <tr>
      <th>on_streaming_date</th>
      <td>16384</td>
      <td>1</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>runtime_in_minutes</th>
      <td>16238</td>
      <td>147</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>tomatometer_rating</th>
      <td>16385</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>tomatometer_count</th>
      <td>16385</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>audience_rating</th>
      <td>16385</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-37597162-5ac4-4387-adec-bf038e38c187')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-37597162-5ac4-4387-adec-bf038e38c187 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-37597162-5ac4-4387-adec-bf038e38c187');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# visualizing missing values in the dataset with numerical columns
msno.matrix(rt_df[num_col], color=(0.4,0.2,0.5))
```




    <Axes: >




    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_46_1.png)
    



```python
missing_num_col = [col for col in num_col if rt_df[col].isna().sum() > 0]
missing_num_col
```




    ['in_theaters_date', 'on_streaming_date', 'runtime_in_minutes']




```python
# creating histograms for all the numerical columns with missing values
missing_num_col = [col for col in num_col if rt_df[col].isna().sum() > 0]

number_cols = len(missing_num_col)
cols = 2
rows = (number_cols//cols+1)

fig, axis = plt.subplots(nrows = rows, ncols = cols, figsize=(12, 4*rows))

axis = axis.flatten()

for i, col in enumerate(missing_num_col):
    sns.histplot(rt_df, x=col, kde=True, bins = 20, color=sns.color_palette('hls', len(missing_num_col))[i], ax=axis[i])
    axis[i].set_title(col)
    axis[i].set_xlabel(col)
    axis[i].set_ylabel('Count')

    axis[i].text(0.95, 0.95, f"Mean: {rt_df[col].mean()}", ha='right', va='top', transform=axis[i].transAxes, fontsize=10, color='blue')
    axis[i].text(0.95, 0.85, f"Median: {rt_df[col].median()}", ha='right', va='top', transform=axis[i].transAxes, fontsize=10, color='green')
    axis[i].text(0.95, 0.75, f"Mode: {rt_df[col].mode()[0]}", ha='right', va='top', transform=axis[i].transAxes, fontsize=10, color='red')

    axis[i].axvline(rt_df[col].mean(), color='blue', linestyle='dashed', linewidth=1)
    axis[i].axvline(rt_df[col].median(), color='green', linestyle='dashed', linewidth=1)
    axis[i].axvline(rt_df[col].mode()[0], color='red', linestyle='dashed', linewidth=1)


for i in range(number_cols, len(axis)):
    axis[i].axis('off')

fig.suptitle('Distribution of Numerical columns', fontsize=16)
plt.tight_layout()
plt.show()
```


    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_48_0.png)
    


**Observation**
1. **runtime_in_minutes** has an almost symmetrical distribution hence we go with **mean imputation** to fill missing values
2. **in_theaters_date** and **on_streaming_date** have skewed distribution hebce we go with **median impytation** to dill missing values


```python
# mean imputation
# filling missing values in 'runtime_in_minutes' using mean
rt_df['runtime_in_minutes']=rt_df['runtime_in_minutes'].fillna(rt_df['runtime_in_minutes'].mean())
```


```python
# median imputation
# filling missing values using median
for col in ['in_theaters_date', 'on_streaming_date']:
    rt_df[col]=rt_df[col].fillna(rt_df[col].median())
```


```python
# verifying
print(f"Missing values in numeric columns: {rt_df[[col for col in num_col]].isna().max().sum()}")
```

    Missing values in numeric columns: 0


**Missing values in numeric columns has been handled**

## Handling missing values in Categorical/Textual columns


```python
# total missing values in Categorical/Textual columns
pd.concat([rt_df[txt_col].count(), rt_df[txt_col].isna().sum(), round((rt_df[txt_col].isna().sum()/len(rt_df))*100, 2)],
          axis = 1,
          keys = ["Total Count of Values", "Total Missing Values", "Percent of Missing Values"])
```





  <div id="df-5ee61723-ee8b-4564-a238-d453a908086c" class="colab-df-container">
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
      <th>Total Count of Values</th>
      <th>Total Missing Values</th>
      <th>Percent of Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>movie_title</th>
      <td>16385</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>movie_info</th>
      <td>16367</td>
      <td>18</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>critics_consensus</th>
      <td>8281</td>
      <td>8104</td>
      <td>49.46</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>16385</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>genre</th>
      <td>16368</td>
      <td>17</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>directors</th>
      <td>16281</td>
      <td>104</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>writers</th>
      <td>15108</td>
      <td>1277</td>
      <td>7.79</td>
    </tr>
    <tr>
      <th>cast</th>
      <td>16125</td>
      <td>260</td>
      <td>1.59</td>
    </tr>
    <tr>
      <th>studio_name</th>
      <td>16010</td>
      <td>375</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>tomatometer_status</th>
      <td>16385</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5ee61723-ee8b-4564-a238-d453a908086c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5ee61723-ee8b-4564-a238-d453a908086c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5ee61723-ee8b-4564-a238-d453a908086c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# visualizing missing values in the dataset with Categorical/Textual columns
msno.matrix(rt_df[txt_col], color=(0.4,0.2,0.5))
```




    <Axes: >




    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_56_1.png)
    



```python
missing_txt_col = [col for col in txt_col if rt_df[col].isna().sum() > 0]
print(missing_txt_col)
```

    ['movie_info', 'critics_consensus', 'genre', 'directors', 'writers', 'cast', 'studio_name']


**Observation**
1. **critics_consensus** has near 50% missing values but it contains info which might be relevant and impact the target (audience_rating). Hence lets transform the values using sentiment analysis.
2. **'movie_info', 'genre', 'directors', 'writers', 'cast', 'studio_name'** have considerably less missing values but mean, median or mode imputation cannot be used as each value is highly associated to the movie, hence here we will use placeholders.

### Transforming 'critics_consensus'


```python
# taking a copy of the current dataset before processing 'critics_consensus' as it will be need in later
critics_consensus_df = rt_df
```


```python
# transforming 'critics_consensus' column
def classify_sentiment(text):
    if pd.isnull(text):
        return 'Unknown'
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

rt_df['critics_sentiment'] = rt_df['critics_consensus'].apply(classify_sentiment)
rt_df['critics_sentiment'].value_counts()
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
      <th>count</th>
    </tr>
    <tr>
      <th>critics_sentiment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unknown</th>
      <td>8104</td>
    </tr>
    <tr>
      <th>Positive</th>
      <td>5717</td>
    </tr>
    <tr>
      <th>Negative</th>
      <td>2122</td>
    </tr>
    <tr>
      <th>Neutral</th>
      <td>442</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# Dropping 'critics_consensus' column
rt_df = rt_df.drop(columns=['critics_consensus'])

# Fill missing values in text columns with placeholders
rt_df['movie_info'] = rt_df['movie_info'].fillna('Unknown')
rt_df['genre'] = rt_df['genre'].fillna('Unknown Genre')
rt_df['directors'] = rt_df['directors'].fillna('Unknown Director')
rt_df['writers'] = rt_df['writers'].fillna('Unknown Writer')
rt_df['cast'] = rt_df['cast'].fillna('Unknown Cast')
rt_df['studio_name'] = rt_df['studio_name'].fillna('Unknown Studio')
```


```python
# verifying
txt_col.remove('critics_consensus')
print(f"Missing values in Categorical/Textual columns: {rt_df[[col for col in txt_col]].isna().max().sum()}")
```

    Missing values in Categorical/Textual columns: 0


## Verifying whole dataset for missing values


```python
print(f"Missing values: {rt_df.isna().max().sum()}")
```

    Missing values: 0



```python
msno.matrix(rt_df, color=(0.4,0.2,0.5))
```




    <Axes: >




    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_66_1.png)
    


**Missing values has been handled**

# **Feature Encoding**

## Encoding Date Columns


```python
rt_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 16385 entries, 0 to 16637
    Data columns (total 16 columns):
     #   Column              Non-Null Count  Dtype         
    ---  ------              --------------  -----         
     0   movie_title         16385 non-null  object        
     1   movie_info          16385 non-null  object        
     2   rating              16385 non-null  object        
     3   genre               16385 non-null  object        
     4   directors           16385 non-null  object        
     5   writers             16385 non-null  object        
     6   cast                16385 non-null  object        
     7   in_theaters_date    16385 non-null  datetime64[ns]
     8   on_streaming_date   16385 non-null  datetime64[ns]
     9   runtime_in_minutes  16385 non-null  float64       
     10  studio_name         16385 non-null  object        
     11  tomatometer_status  16385 non-null  object        
     12  tomatometer_rating  16385 non-null  int64         
     13  tomatometer_count   16385 non-null  int64         
     14  audience_rating     16385 non-null  float64       
     15  critics_sentiment   16385 non-null  object        
    dtypes: datetime64[ns](2), float64(2), int64(2), object(10)
    memory usage: 2.1+ MB



```python
# Encoding Dates into day, month and year

# Encoding in_theaters_date
rt_df['in_theaters_day'] = rt_df['in_theaters_date'].dt.day
rt_df['in_theaters_month'] = rt_df['in_theaters_date'].dt.month
rt_df['in_theaters_year'] = rt_df['in_theaters_date'].dt.year

# Encoding on_streaming_date
rt_df['on_streaming_day'] = rt_df['on_streaming_date'].dt.day
rt_df['on_streaming_month'] = rt_df['on_streaming_date'].dt.month
rt_df['on_streaming_year'] = rt_df['on_streaming_date'].dt.year

# droping in_theaters_date & on_streaming_date
rt_df.drop(columns={'in_theaters_date', 'on_streaming_date'}, inplace=True)
```

## Encoding Category Columns


```python
# OneHot encoding 'rating', 'tomatometer_status' & 'critics_sentiment'
rt_df = pd.get_dummies(rt_df, columns=['rating', 'tomatometer_status', 'critics_sentiment'])

# Replacing True & False with 1 & 0
rt_df = rt_df.replace({True: 1, False: 0})
```


```python
# Lets check if we can encode other textual columns
# most of them seems to be seperated by commas
for column in [col for col in rt_df.columns if rt_df[col].dtype == 'O']:
    unique_values = set(
                        str(value).strip()
                        for values in rt_df[column]
                            for value in str(values).split(',')
                        )
    print(f"Number of unique values in {column}: {len(unique_values)}")
```

    Number of unique values in movie_title: 16068
    Number of unique values in movie_info: 129530
    Number of unique values in genre: 22
    Number of unique values in directors: 8848
    Number of unique values in writers: 14514
    Number of unique values in cast: 197742
    Number of unique values in studio_name: 2827


**Observations & Actions**
1. **genre** has 22 unique values this needs to be encoded
2. **movie_title** can be dropped as it does not have impact on audience_rating
3. **movie_info** can also be droped as transforming it does not make sense and most of its content is already captured in **genre** column
4. **directors, writers, cast, studio_name** lets try to frequency encode these columns

### Multi Lable Encoding


```python
# Encoding 'genre' column
genre_lists = rt_df['genre'].apply(lambda x: [i.strip() for i in str(x).split(',')])

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(genre_lists), columns=mlb.classes_, index=rt_df.index)

rt_df = pd.concat([rt_df, genre_encoded], axis=1).drop(columns=['genre'])
```


```python
# dropping 'movie_title' and 'movie_info' columns
rt_df.drop(columns=['movie_title', 'movie_info'], inplace=True)
```

### Frequency Encoding


```python
# encoding directors, writers, cast & studio_name
def frequency_encode(column):

    all_values = [value.strip() for item in column for value in str(item).split(',')]
    frequency = pd.Series(all_values).value_counts()
    return column.apply(lambda x: sum(frequency.get(value.strip()) for value in str(x).split(',')))

rt_df['directors_freq'] = frequency_encode(rt_df['directors'])
rt_df['writers_freq'] = frequency_encode(rt_df['writers'])
rt_df['cast_freq'] = frequency_encode(rt_df['cast'])
rt_df['studio_name_freq'] = frequency_encode(rt_df['studio_name'])
```


```python
rt_df[['directors', 'directors_freq', 'writers', 'writers_freq', 'cast', 'cast_freq', 'studio_name', 'studio_name_freq']].head()
```





  <div id="df-1f880b97-1eb2-4097-bef0-4a5200286d05" class="colab-df-container">
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
      <th>directors</th>
      <th>directors_freq</th>
      <th>writers</th>
      <th>writers_freq</th>
      <th>cast</th>
      <th>cast_freq</th>
      <th>studio_name</th>
      <th>studio_name_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chris Columbus</td>
      <td>13</td>
      <td>Craig Titley</td>
      <td>2</td>
      <td>Logan Lerman, Brandon T. Jackson, Alexandra Da...</td>
      <td>719</td>
      <td>20th Century Fox</td>
      <td>414</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nicole Holofcener</td>
      <td>5</td>
      <td>Nicole Holofcener</td>
      <td>6</td>
      <td>Catherine Keener, Amanda Peet, Oliver Platt, R...</td>
      <td>260</td>
      <td>Sony Pictures Classics</td>
      <td>259</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blake Edwards</td>
      <td>27</td>
      <td>Blake Edwards</td>
      <td>21</td>
      <td>Dudley Moore, Bo Derek, Julie Andrews, Robert ...</td>
      <td>315</td>
      <td>Waner Bros.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sidney Lumet</td>
      <td>30</td>
      <td>Reginald Rose</td>
      <td>3</td>
      <td>Martin Balsam, John Fiedler, Lee J. Cobb, E.G....</td>
      <td>208</td>
      <td>Criterion Collection</td>
      <td>110</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Richard Fleischer</td>
      <td>17</td>
      <td>Earl Felton</td>
      <td>3</td>
      <td>James Mason, Kirk Douglas, Paul Lukas, Peter L...</td>
      <td>242</td>
      <td>Disney</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1f880b97-1eb2-4097-bef0-4a5200286d05')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1f880b97-1eb2-4097-bef0-4a5200286d05 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1f880b97-1eb2-4097-bef0-4a5200286d05');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




**Observation**

The columns **directors, writers, cast & studio_name** has been frequency encoded where each value for example 'Chris Columbus' is replaced with no of time times they appear in the dataset.


```python
rt_df.loc[rt_df['directors'] == 'Unknown Director', ['directors', 'directors_freq']].sample(5)
```





  <div id="df-6d86a28a-311a-484e-b6cd-85df92a1185b" class="colab-df-container">
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
      <th>directors</th>
      <th>directors_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11650</th>
      <td>Unknown Director</td>
      <td>104</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>Unknown Director</td>
      <td>104</td>
    </tr>
    <tr>
      <th>2876</th>
      <td>Unknown Director</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9736</th>
      <td>Unknown Director</td>
      <td>104</td>
    </tr>
    <tr>
      <th>9173</th>
      <td>Unknown Director</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6d86a28a-311a-484e-b6cd-85df92a1185b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6d86a28a-311a-484e-b6cd-85df92a1185b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6d86a28a-311a-484e-b6cd-85df92a1185b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




**Observation**

The values which was used to fill the missing values in directors, writers, cast & studio_name has also been frequency encoded which we do not want. Hence we will replace all the frequesncy for these to 0.


```python
# replacing frequency of unknown values with 0
rt_df.loc[rt_df['writers'] == 'Unknown Writer', 'writers_freq'] = 0
rt_df.loc[rt_df['directors'] == 'Unknown Director', 'directors_freq'] = 0
rt_df.loc[rt_df['cast'] == 'Unknown Cast', 'cast_freq'] = 0
rt_df.loc[rt_df['studio_name'] == 'Unknown Studio', 'studio_name_freq']  = 0
```


```python
# dropping directors, writers, cast & studio_name
rt_df.drop(columns = ['directors', 'writers', 'cast', 'studio_name'], inplace=True)
```

**Data Cleaning and Processing Complete**


```python
# final dataset
print(f"The dataset has {rt_df.shape[0]} rows and {rt_df.shape[1]} Columns")
```

    The dataset has 16385 rows and 49 Columns


# **Train Test Split**


```python
# splitting dataset to train and test datasets

X = rt_df.drop(columns = 'audience_rating')
y = rt_df['audience_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f"Train dataset shape = {X_train.shape}")
print(f"Test dataset shape = {X_test.shape}")
```

    Train dataset shape = (13108, 48)
    Test dataset shape = (3277, 48)


# **Feature Scaling - Normalization (Min-Max Scaling)**


```python
# scaled traing data will be used in the required ML model only
scalar = MinMaxScaler()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
```


```python
print(f"Scaled train dataset shape = {scaled_X_train.shape}")
```

    Scaled train dataset shape = (13108, 48)


# **Linear Regression**


```python
# creating a variable to store the results of all the models
model_perf = []
def evaluate_model(model, pred):
    rmse = round(root_mean_squared_error(y_test, pred), 2)
    r2 = round(r2_score(y_test, pred), 2)
    model_perf.append({'Model': model, 'RMSE': rmse, 'R Squared': r2})
    print(f"RMSE of {model}: {rmse}")
    print(f"R Squared of {model}: {r2}")
```


```python
# initializing Linear Regression model
lreg = LinearRegression()

# training model with normalized dataset (since linear regression is sensitive to scale of values)
lreg.fit(scaled_X_train, y_train)

# predicting on scaled test dataset
pred_lreg = lreg.predict(scaled_X_test)

# evaluating model
evaluate_model('Linear Regression', pred_lreg)
```

    RMSE of Linear Regression: 13.99
    R Squared of Linear Regression: 0.53



```python
# checking for overfitting
print(f"RMSE with train data: {root_mean_squared_error(y_train, lreg.predict(scaled_X_train)):.2f}")
print(f"R Squared with train data: {r2_score(y_train, lreg.predict(scaled_X_train)):.2f}")
```

    RMSE with train data: 13.99
    R Squared with train data: 0.53



```python
# lets check the consitancy of the model accross different splits of the dataset
cv_scores = cross_val_score(lreg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Mean Cross-Validation RMSE: {(np.sqrt(-cv_scores)).mean():.2f}")

cv_r2 = cross_val_score(lreg, X_train, y_train, cv=5, scoring='r2')
print(f"Mean Cross-Validation R²: {cv_r2.mean():.2f}")
```

    Mean Cross-Validation RMSE: 14.23
    Mean Cross-Validation R²: 0.52


**Observation**

The RMSE and R Squared for both train dataset and test dataset are comparable which indicates that the model is not overfitting. From K fold cross validation the model is consistant across data splits.

# **Ridge and Lasso Regression**


```python
# Ridge Regression
alpha = [0.1, 1.0, 10.0, 100.0]
for i in alpha:
    Rreg = Ridge(alpha=i, random_state=42)
    Rreg.fit(scaled_X_train, y_train)
    print(f"alpha = {i}")
    print(f"RMSE with test data: {root_mean_squared_error(y_test, Rreg.predict(scaled_X_test)):.2f}")
    print(f"R Squared with test data: {r2_score(y_test, Rreg.predict(scaled_X_test)):.2f}\n")
```

    alpha = 0.1
    RMSE with test data: 13.99
    R Squared with test data: 0.53
    
    alpha = 1.0
    RMSE with test data: 14.01
    R Squared with test data: 0.53
    
    alpha = 10.0
    RMSE with test data: 14.08
    R Squared with test data: 0.52
    
    alpha = 100.0
    RMSE with test data: 14.22
    R Squared with test data: 0.51
    



```python
# Lasso Regression
alpha = [0.1, 1.0, 10.0, 100.0]
for i in alpha:
    Lareg = Lasso(alpha=i, random_state=42)
    Lareg.fit(scaled_X_train, y_train)
    print(f"alpha = {i}")
    print(f"RMSE with test data: {root_mean_squared_error(y_test, Lareg.predict(scaled_X_test)):.2f}")
    print(f"R Squared with test data: {r2_score(y_test, Lareg.predict(scaled_X_test)):.2f}\n")
```

    alpha = 0.1
    RMSE with test data: 14.32
    R Squared with test data: 0.51
    
    alpha = 1.0
    RMSE with test data: 15.62
    R Squared with test data: 0.41
    
    alpha = 10.0
    RMSE with test data: 20.36
    R Squared with test data: -0.00
    
    alpha = 100.0
    RMSE with test data: 20.36
    R Squared with test data: -0.00
    



```python
# best alpha value
Rreg = Ridge(alpha=0.1, random_state=42)
Rreg.fit(scaled_X_train, y_train)
evaluate_model('Ridge Linear Regression', Rreg.predict(scaled_X_test))

Lareg = Lasso(alpha=0.1, random_state=42)
Lareg.fit(scaled_X_train, y_train)
evaluate_model('Lasso Linear Regression', Lareg.predict(scaled_X_test))
```

    RMSE of Ridge Linear Regression: 13.99
    R Squared of Ridge Linear Regression: 0.53
    RMSE of Lasso Linear Regression: 14.32
    R Squared of Lasso Linear Regression: 0.51


# **Random Forest Regression**


```python
# Initialize Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

# Training the model
rf_reg.fit(X_train, y_train)

# Predicting on the test dataset
pred_rf_reg = rf_reg.predict(X_test)

# Evaluating the model
print(f"RMSE: {root_mean_squared_error(y_test, pred_rf_reg):.2f}")
print(f"R Squared: {r2_score(y_test, pred_rf_reg):.2f}")
```

    RMSE: 13.58
    R Squared: 0.56



```python
# hyperparameter tuninig
result_rf_reg = []
for criterion in ['squared_error', 'friedman_mse', 'poisson']:
    for n_estimators in [100, 200, 300]:
        for max_depth in [10, 15, None]:

            rf_reg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=42)
            rf_reg.fit(X_train, y_train)
            pred_rf_reg = rf_reg.predict(X_test)

            result_rf_reg.append(
                {
                    'No of Estimators': n_estimators,
                    'Criterion': criterion,
                    'Max Depth': max_depth,
                    'RMSE': round(root_mean_squared_error(y_test, pred_rf_reg), 2),
                    'R Squared': r2_score(y_test, pred_rf_reg)
                }
            )

print(f"Training Done!")
```

    Training Done!



```python
# best RMSE values
result_rf_reg = pd.DataFrame(result_rf_reg)
result_rf_reg.loc[result_rf_reg['RMSE'].idxmin()]
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
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No of Estimators</th>
      <td>300</td>
    </tr>
    <tr>
      <th>Criterion</th>
      <td>squared_error</td>
    </tr>
    <tr>
      <th>Max Depth</th>
      <td>15.0</td>
    </tr>
    <tr>
      <th>RMSE</th>
      <td>13.51</td>
    </tr>
    <tr>
      <th>R Squared</th>
      <td>0.559768</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>




```python
# best R Squared values
result_rf_reg.loc[result_rf_reg['R Squared'].idxmax()]
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
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No of Estimators</th>
      <td>300</td>
    </tr>
    <tr>
      <th>Criterion</th>
      <td>friedman_mse</td>
    </tr>
    <tr>
      <th>Max Depth</th>
      <td>15.0</td>
    </tr>
    <tr>
      <th>RMSE</th>
      <td>13.51</td>
    </tr>
    <tr>
      <th>R Squared</th>
      <td>0.559794</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>



**Observation**

From above we can observe that the paramenters with best RMSE and R Squared are same (n_estimators=300, criterion='squared_error', max_depth=None)


```python
# Random Forest Regressor with optimal parameters
rf_reg = RandomForestRegressor(n_estimators=300, criterion='squared_error', max_depth=None, random_state=42)
rf_reg.fit(X_train, y_train)

evaluate_model('Random Forest Regression', rf_reg.predict(X_test))
```

    RMSE of Random Forest Regression: 13.52
    R Squared of Random Forest Regression: 0.56


# **XGBoost Regressor**


```python
# Initialize XGBoost Regressor
xgb_reg = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

# Training the model
xgb_reg.fit(X_train, y_train)

# Predict on test data
y_pred = xgb_reg.predict(X_test)

# Evaluate the model
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"R Squared: {r2_score(y_test, y_pred):.2f}")
```

    RMSE: 13.26
    R Squared: 0.58



```python
# hyperparameter tuninig
result_xgb_reg = []
for n_estimators in [100, 200, 300]:
    for learning_rate in [0.01, 0.05, 0.1, 0.2]:
        for max_depth in [3, 5, 7, 9]:
                xgb_reg = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                xgb_reg.fit(X_train, y_train)
                pred_xgb_reg = xgb_reg.predict(X_test)

                result_xgb_reg.append(
                    {
                        'No of Estimators': n_estimators,
                        'Learning Rate': learning_rate,
                        'Max Depth': max_depth,
                        'RMSE': round(root_mean_squared_error(y_test, pred_xgb_reg), 2),
                        'R Squared': r2_score(y_test, pred_xgb_reg)
                    }
                )
print(f"Training Done!")
```

    Training Done!



```python
# best RMSE values
result_xgb_reg = pd.DataFrame(result_xgb_reg)
result_xgb_reg.loc[result_xgb_reg['RMSE'].idxmin()]
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
      <th>21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No of Estimators</th>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>Learning Rate</th>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>Max Depth</th>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>RMSE</th>
      <td>13.230000</td>
    </tr>
    <tr>
      <th>R Squared</th>
      <td>0.577738</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
# best R Squared values
result_xgb_reg.loc[result_xgb_reg['R Squared'].idxmax()]
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
      <th>21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No of Estimators</th>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>Learning Rate</th>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>Max Depth</th>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>RMSE</th>
      <td>13.230000</td>
    </tr>
    <tr>
      <th>R Squared</th>
      <td>0.577738</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



**Observation**

From above we can observe that the paramenters with best RMSE and R Squared are same (n_estimators=200, Learnign Rate=0.05, max_depth=5)


```python
# XGBoost Regressor with optimal parameters
xgb_reg = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_reg.fit(X_train, y_train)

evaluate_model('XGBoost Regressor', xgb_reg.predict(X_test))
```

    RMSE of XGBoost Regressor: 13.23
    R Squared of XGBoost Regressor: 0.58


# **CatBoost Regressor**


```python
# Initialize CatBoost Regressor
catboost_reg = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_state=42)

# Training the model
catboost_reg.fit(X_train, y_train)

# predicting on test data
y_pred = catboost_reg.predict(X_test)

# Evaluate the model
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
```

    RMSE: 13.25
    R²: 0.58



```python
# hyperparameter tuninig
result_catboost_reg = []
for iterations in [100, 200, 300]:
    for learning_rate in [0.01, 0.05, 0.1, 0.2]:
        for depth in [4, 6, 8, 10]:
                catboost_reg = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0, random_state=42)
                catboost_reg.fit(X_train, y_train)
                pred_catboost_reg = catboost_reg.predict(X_test)

                result_catboost_reg.append(
                    {
                        'Iterations': iterations,
                        'Learning Rate': learning_rate,
                        'Depth': depth,
                        'RMSE': round(root_mean_squared_error(y_test, pred_catboost_reg), 2),
                        'R Squared': r2_score(y_test, pred_catboost_reg)
                    }
                )
print(f"Training Done!")
```

    Training Done!



```python
# best RMSE values
result_catboost_reg = pd.DataFrame(result_catboost_reg)
result_catboost_reg.loc[result_catboost_reg['RMSE'].idxmin()]
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
      <th>38</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iterations</th>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>Learning Rate</th>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>Depth</th>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>RMSE</th>
      <td>13.110000</td>
    </tr>
    <tr>
      <th>R Squared</th>
      <td>0.585384</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>




```python
# best R Squared values
result_catboost_reg.loc[result_catboost_reg['R Squared'].idxmax()]
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
      <th>38</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iterations</th>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>Learning Rate</th>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>Depth</th>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>RMSE</th>
      <td>13.110000</td>
    </tr>
    <tr>
      <th>R Squared</th>
      <td>0.585384</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>



**Observation**

From above we can observe that the paramenters with best RMSE and R Squared are same (iterations=300, Learnign Rate=0.05, depth=8)


```python
# CatBoost Regressor with optimal parameters
catboost_reg = CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05, verbose=0, random_state=42)
catboost_reg.fit(X_train, y_train)

evaluate_model('CatBoost Regressor', catboost_reg.predict(X_test))
```

    RMSE of CatBoost Regressor: 13.11
    R Squared of CatBoost Regressor: 0.59


# **Feedforward Neural Network**


```python
model = Sequential([
    Input(shape=(scaled_X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">3,136</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,249</span> (20.50 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,249</span> (20.50 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(scaled_X_train, y_train, epochs=30, verbose=1, validation_split=0.2)

# Predict on test data
y_pred = model.predict(scaled_X_test)

# Evaluate the model
evaluate_model('Feedforward Neural Network', y_pred)
```

    Epoch 1/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 6ms/step - loss: 954.4059 - val_loss: 237.4235
    Epoch 2/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 8ms/step - loss: 235.1667 - val_loss: 222.7542
    Epoch 3/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 222.2281 - val_loss: 216.5892
    Epoch 4/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 214.7578 - val_loss: 213.9489
    Epoch 5/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 208.7403 - val_loss: 207.6172
    Epoch 6/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 204.1488 - val_loss: 206.8962
    Epoch 7/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 201.6864 - val_loss: 202.7898
    Epoch 8/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 198.0293 - val_loss: 200.5734
    Epoch 9/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 196.4825 - val_loss: 201.1003
    Epoch 10/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 194.0689 - val_loss: 197.7098
    Epoch 11/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 192.8432 - val_loss: 205.2942
    Epoch 12/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 191.5342 - val_loss: 195.3184
    Epoch 13/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 189.9412 - val_loss: 195.2723
    Epoch 14/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 189.2857 - val_loss: 194.6075
    Epoch 15/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 188.5014 - val_loss: 194.4817
    Epoch 16/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 187.3192 - val_loss: 194.3218
    Epoch 17/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 185.9148 - val_loss: 195.5977
    Epoch 18/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 185.3235 - val_loss: 193.7781
    Epoch 19/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 185.4326 - val_loss: 193.2904
    Epoch 20/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 3ms/step - loss: 184.8078 - val_loss: 193.9355
    Epoch 21/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 184.4402 - val_loss: 192.4518
    Epoch 22/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 183.8781 - val_loss: 193.2090
    Epoch 23/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 183.2305 - val_loss: 199.1388
    Epoch 24/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 182.2159 - val_loss: 196.8798
    Epoch 25/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 181.8878 - val_loss: 192.2133
    Epoch 26/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 181.6548 - val_loss: 192.5376
    Epoch 27/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 180.9703 - val_loss: 192.7664
    Epoch 28/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 180.0427 - val_loss: 193.6093
    Epoch 29/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 180.2749 - val_loss: 193.1611
    Epoch 30/30
    [1m328/328[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 179.6483 - val_loss: 191.9061
    [1m103/103[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    RMSE of Feedforward Neural Network: 13.8
    R Squared of Feedforward Neural Network: 0.54


# **Multi-Input Neural Network**


```python
critics_consensus_df['critics_consensus'] = critics_consensus_df['critics_consensus'].fillna('Unknown')

# Tokenize the text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(critics_consensus_df['critics_consensus'])
sequences = tokenizer.texts_to_sequences(critics_consensus_df['critics_consensus'])
```


```python
# padding the sequences
padded_sequences = pad_sequences(sequences, maxlen=88, padding='post')
```


```python
# creating the feature datsets
X_text = np.array(padded_sequences)
X_other = X.drop(columns={'critics_sentiment_Negative', 'critics_sentiment_Neutral','critics_sentiment_Positive', 'critics_sentiment_Unknown',}).values
y = rt_df['audience_rating'].values

X_combined = [X_text, X_other]
```


```python
# normalizing (min-max scaling)
scaler = MinMaxScaler()
X_other_normalized = scaler.fit_transform(X_other)
```


```python
# splitting dataset to train and test datasets
X_train_text, X_test_text, X_train_other, X_test_other, y_train, y_test = train_test_split(X_text, X_other_normalized, y, test_size=0.2, random_state=42)
```


```python
# textual input branch
text_input = Input(shape=(88,), name="text_input")
embedding = Embedding(input_dim=10000, output_dim=128, input_length=88)(text_input)
lstm_out = LSTM(64)(embedding)

# numerical input branch
num_input = Input(shape=(X_other.shape[1],), name="num_input")
dense_num = Dense(64, activation='relu')(num_input)

# combining text and numerical features
merged = Concatenate()([lstm_out, dense_num])
dense = Dense(128, activation='relu')(merged)
output = Dense(1, activation='linear')(dense)

# defining the model
model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ text_input          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">88</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ embedding           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">88</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │  <span style="color: #00af00; text-decoration-color: #00af00">1,280,000</span> │ text_input[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ num_input           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">44</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">49,408</span> │ embedding[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │      <span style="color: #00af00; text-decoration-color: #00af00">2,880</span> │ num_input[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ lstm[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],       │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │                   │            │ dense_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │     <span style="color: #00af00; text-decoration-color: #00af00">16,512</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">129</span> │ dense_4[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,348,929</span> (5.15 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,348,929</span> (5.15 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# training the model
history = model.fit(
    [X_train_text, X_train_other], y_train,
    validation_data=([X_test_text, X_test_other], y_test),
    epochs=10,batch_size=32
)
```

    Epoch 1/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 45ms/step - loss: 522.3568 - mae: 16.8430 - val_loss: 222.1161 - val_mae: 11.9875
    Epoch 2/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m17s[0m 43ms/step - loss: 221.4520 - mae: 11.9081 - val_loss: 212.4980 - val_mae: 11.6843
    Epoch 3/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m19s[0m 46ms/step - loss: 213.3411 - mae: 11.6580 - val_loss: 208.6975 - val_mae: 11.6902
    Epoch 4/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m18s[0m 43ms/step - loss: 207.5067 - mae: 11.4621 - val_loss: 201.0126 - val_mae: 11.2767
    Epoch 5/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 43ms/step - loss: 202.4555 - mae: 11.2913 - val_loss: 199.1640 - val_mae: 11.2231
    Epoch 6/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 49ms/step - loss: 198.9731 - mae: 11.1656 - val_loss: 200.0201 - val_mae: 11.1262
    Epoch 7/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m18s[0m 44ms/step - loss: 197.5095 - mae: 11.1063 - val_loss: 194.6804 - val_mae: 11.1149
    Epoch 8/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m19s[0m 45ms/step - loss: 195.8876 - mae: 11.0317 - val_loss: 194.7630 - val_mae: 10.9562
    Epoch 9/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m18s[0m 44ms/step - loss: 194.0204 - mae: 10.9916 - val_loss: 193.7209 - val_mae: 10.9733
    Epoch 10/10
    [1m410/410[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m19s[0m 45ms/step - loss: 192.9033 - mae: 10.9468 - val_loss: 194.6994 - val_mae: 10.9930



```python
# predicting on test dataset
y_pred = model.predict([X_test_text, X_test_other])
```

    [1m103/103[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step



```python
# Evaluate the model
evaluate_model('Multi-Input Neural Network', y_pred)
```

    RMSE of Multi-Input Neural Network: 13.95
    R Squared of Multi-Input Neural Network: 0.53


# Model Comparision


```python
pd.DataFrame(model_perf)
```





  <div id="df-6bbacf85-771b-428b-aaab-b2fc1fe4f5f1" class="colab-df-container">
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
      <th>Model</th>
      <th>RMSE</th>
      <th>R Squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>13.99</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge Linear Regression</td>
      <td>13.99</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso Linear Regression</td>
      <td>14.32</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest Regression</td>
      <td>13.52</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XGBoost Regressor</td>
      <td>13.23</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>XGBoost Regressor</td>
      <td>13.28</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost Regressor</td>
      <td>13.23</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CatBoost Regressor</td>
      <td>13.11</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Feedforward Neural Network</td>
      <td>13.80</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Multi-Input Neural Network</td>
      <td>13.95</td>
      <td>0.53</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6bbacf85-771b-428b-aaab-b2fc1fe4f5f1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6bbacf85-771b-428b-aaab-b2fc1fe4f5f1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6bbacf85-771b-428b-aaab-b2fc1fe4f5f1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
# Visualizing the results
model_perf = pd.DataFrame(model_perf)

fig, ax1 = plt.subplots(figsize=(12, 8))

# plot RMSE
sns.lineplot(data=model_perf, x="Model", y="RMSE", marker="o", color="blue", ax=ax1, label="RMSE")
ax1.set_ylabel("Root Mean Squared Error", fontsize=12)
ax1.set_xlabel("Model", fontsize=12)
ax1.legend(loc="upper left")
ax1.set_ylim(0, max(model_perf['RMSE']) + 5)

# highlighting the model with minimum RMSE
min_rmse_idx = model_perf['RMSE'].idxmin()
min_rmse_value = model_perf.loc[min_rmse_idx, 'RMSE']
min_rmse_model = model_perf.loc[min_rmse_idx, 'Model']
ax1.scatter(x=[min_rmse_model], y=[min_rmse_value], color="red", s=100, label="Min RMSE")

# plot R²
ax2 = ax1.twinx()
sns.lineplot(data=model_perf, x="Model", y="R Squared", marker="o", color="green", ax=ax2, label="R²")
ax2.set_ylabel("R Squared", fontsize=12)
ax2.legend(loc="upper right")
ax2.set_ylim(0, 1)

# highlighting the model with maximum R²
max_r2_idx = model_perf['R Squared'].idxmax()
max_r2_value = model_perf.loc[max_r2_idx, 'R Squared']
max_r2_model = model_perf.loc[max_r2_idx, 'Model']
ax2.scatter(x=[max_r2_model], y=[max_r2_value], color="red", s=100, label="Max R²")

# Rotate x-axis values explicitly
ax1.set_xticklabels(model_perf["Model"], rotation=45, ha='right')

# Add title and adjust layout
fig.suptitle('Model Performance', fontsize=16)
fig.tight_layout()
plt.show()

```


    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_140_0.png)
    


# **Conclusion**


```python
# best performing model
model_perf.loc[model_perf['Model'] == 'CatBoost Regressor']
```





  <div id="df-8b8fa942-a78b-4cb9-b4e4-f026a2278ab1" class="colab-df-container">
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
      <th>Model</th>
      <th>RMSE</th>
      <th>R Squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>CatBoost Regressor</td>
      <td>13.11</td>
      <td>0.59</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8b8fa942-a78b-4cb9-b4e4-f026a2278ab1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8b8fa942-a78b-4cb9-b4e4-f026a2278ab1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8b8fa942-a78b-4cb9-b4e4-f026a2278ab1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




**CatBoost Regression** has the hightest model performance compared to other models with **Root Mean Squared Error: 13.11** and **R Squared: 0.59**.


```python
# Plotting Original vs Predicted values (CatBoost Regressor)
df = pd.DataFrame({
    'Index': np.arange(len(y_test)),
    'Original': y_test,
    'Predicted': catboost_reg.predict(X_test)
})

df['Bucket'] = (df['Index'] // 50) * 50
bucketed_df = df.groupby('Bucket').agg({'Original': 'mean', 'Predicted': 'mean'}).reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=bucketed_df, x='Bucket', y='Original', label='Original', color='blue', linewidth=2)
sns.lineplot(data=bucketed_df, x='Bucket', y='Predicted', label='Predicted', color='green', linestyle='--', linewidth=2)

plt.xlabel('Index')
plt.ylabel('Audience Rating')
plt.title('Original Audience Rating vs Predicted Audience Rating')
plt.legend()

plt.show()
```


    
![png](Rotten_Tomatoes_files/Rotten_Tomatoes_144_0.png)
    

