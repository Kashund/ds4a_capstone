# Team_25_DS4A_Linear_Regression_Model_Years_of_Life_Lost


```python
#Install Requirements via Text File
!pip install -r requirements.txt

#Import the packages for cleaning, analysis, and visualization 

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import pandas_profiling as pp
from ydata_profiling import ProfileReport
from pandasql import sqldf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from statsmodels.formula.api import ols
import statsmodels.api as smf
```

    Collecting https://github.com/pandas-profiling/pandas-profiling/archive/master.zip (from -r requirements.txt (line 8))
      Using cached https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: pandas in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 1)) (1.4.4)
    Requirement already satisfied: numpy in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 2)) (1.21.5)
    Requirement already satisfied: scipy in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.9.1)
    Requirement already satisfied: matplotlib in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 4)) (3.5.2)
    Requirement already satisfied: seaborn==v0.11.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 5)) (0.11.2)
    Requirement already satisfied: pingouin in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 6)) (0.5.3)
    Requirement already satisfied: pivottablejs in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 7)) (0.9.0)
    Requirement already satisfied: sqlalchemy==1.4.46 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 9)) (1.4.46)
    Requirement already satisfied: ipython-sql in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 10)) (0.4.1)
    Requirement already satisfied: pandasql in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from -r requirements.txt (line 11)) (0.7.3)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from sqlalchemy==1.4.46->-r requirements.txt (line 9)) (1.1.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pandas->-r requirements.txt (line 1)) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pandas->-r requirements.txt (line 1)) (2022.1)
    Requirement already satisfied: pillow>=6.2.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (9.2.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (1.4.2)
    Requirement already satisfied: cycler>=0.10 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (4.25.0)
    Requirement already satisfied: pyparsing>=2.2.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (3.0.9)
    Requirement already satisfied: packaging>=20.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (21.3)
    Requirement already satisfied: tabulate in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.8.10)
    Requirement already satisfied: outdated in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.2.2)
    Requirement already satisfied: scikit-learn in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (1.0.2)
    Requirement already satisfied: statsmodels>=0.13 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.13.2)
    Requirement already satisfied: pandas-flavor>=0.2.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.5.0)
    Requirement already satisfied: pydantic<1.11,>=1.8.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.10.4)
    Requirement already satisfied: PyYAML<6.1,>=5.0.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (6.0)
    Requirement already satisfied: jinja2<3.2,>=2.11.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.11.3)
    Requirement already satisfied: visions[type_image_path]==0.7.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (0.7.5)
    Requirement already satisfied: htmlmin==0.1.12 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (0.1.12)
    Requirement already satisfied: phik<0.13,>=0.11.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (0.12.3)
    Requirement already satisfied: requests<2.29,>=2.24.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.28.1)
    Requirement already satisfied: tqdm<4.65,>=4.48.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (4.64.1)
    Requirement already satisfied: multimethod<1.10,>=1.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.9.1)
    Requirement already satisfied: typeguard<2.14,>=2.13.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.13.3)
    Requirement already satisfied: attrs>=19.3.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (21.4.0)
    Requirement already satisfied: networkx>=2.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.8.4)
    Requirement already satisfied: tangled-up-in-unicode>=0.0.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (0.2.0)
    Requirement already satisfied: imagehash in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (4.3.1)
    Requirement already satisfied: six in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (1.16.0)
    Requirement already satisfied: ipython-genutils>=0.1.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (0.2.0)
    Requirement already satisfied: ipython>=1.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (7.31.1)
    Requirement already satisfied: prettytable<1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (0.7.2)
    Requirement already satisfied: sqlparse in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (0.4.3)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (3.0.20)
    Requirement already satisfied: setuptools>=18.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (63.4.1)
    Requirement already satisfied: pygments in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (2.11.2)
    Requirement already satisfied: decorator in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.18.1)
    Requirement already satisfied: pickleshare in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.7.5)
    Requirement already satisfied: appnope in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.1.2)
    Requirement already satisfied: pexpect>4.3 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (4.8.0)
    Requirement already satisfied: backcall in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.2.0)
    Requirement already satisfied: traitlets>=4.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (5.1.1)
    Requirement already satisfied: matplotlib-inline in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.1.6)
    Requirement already satisfied: MarkupSafe>=0.23 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jinja2<3.2,>=2.11.1->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.0.1)
    Requirement already satisfied: xarray in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pandas-flavor>=0.2.0->pingouin->-r requirements.txt (line 6)) (0.20.1)
    Requirement already satisfied: lazy-loader>=0.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pandas-flavor>=0.2.0->pingouin->-r requirements.txt (line 6)) (0.1)
    Requirement already satisfied: joblib>=0.14.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from phik<0.13,>=0.11.1->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.1.0)
    Requirement already satisfied: typing-extensions>=4.2.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pydantic<1.11,>=1.8.1->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (4.3.0)
    Requirement already satisfied: idna<4,>=2.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.26.11)
    Requirement already satisfied: charset-normalizer<3,>=2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2022.9.24)
    Requirement already satisfied: patsy>=0.5.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from statsmodels>=0.13->pingouin->-r requirements.txt (line 6)) (0.5.2)
    Requirement already satisfied: littleutils in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from outdated->pingouin->-r requirements.txt (line 6)) (0.2.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn->pingouin->-r requirements.txt (line 6)) (2.2.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jedi>=0.16->ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pexpect>4.3->ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.7.0)
    Requirement already satisfied: wcwidth in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.2.5)
    Requirement already satisfied: PyWavelets in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from imagehash->visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.3.0)


    /var/folders/f0/cswv940s1pl9lnmr89n7jvkm0000gn/T/ipykernel_68724/1615218538.py:12: DeprecationWarning: `import pandas_profiling` is going to be deprecated by April 1st. Please use `import ydata_profiling` instead.
      import pandas_profiling as pp



```python
#Read the data file into a dataframe
with open('data/mass_killing_incidents_public.csv') as f:
    mki=pd.read_csv(f, delimiter=',')
with open('data/mass_killing_offenders_public.csv') as f:
    mko=pd.read_csv(f, delimiter=',')
with open('data/mass_killing_victims_public.csv') as f:
    mkv=pd.read_csv(f, delimiter=',')
with open('data/mass_killing_weapons_public.csv') as f:
    mkw=pd.read_csv(f, delimiter=',')

# Source and Definitions: https://data.world/associatedpress/mass-killings-public/workspace/file?filename=MKA+Public+Data+Codebook.pdf

```


```python
#Merging Mass Killing Incident Table + Victim Table + Offender Table

mki_mkv_mko_q = """ 
            SELECT mki.incident_id, date as incidentdate, city, state, num_offenders,
            num_victims_killed, num_victims_injured, firstcod, secondcod,
            type, situation_type, location_type, location, longitude,
            latitude, narrative, victim_id, mkv.age as v_age, mkv.race as v_race,
            mkv.sex as v_sex, vorelationship, offender_id,
              CASE 
                WHEN mko.age < 0 THEN NULL
                ELSE mko.age
                END AS o_age, 
              mko.race as o_race, mko.sex as o_sex, suicide, outcome
            FROM mki
              LEFT JOIN mkv
              ON mki.incident_id = mkv.incident_id
              LEFT JOIN mko
              ON mki.incident_id = mko.incident_id          
        """
# Query execution and convert to dataframe
mki_mkv_mko = sqldf(mki_mkv_mko_q).copy()

#Cleaning Dataframe
#mki_mkv_mkw['incidentdate'] = pd.to_datetime(mki_mkv_mkw['date'])
#mki_mkv_mkw = mki_mkv_mkw.drop(columns=['date'])
#mki_mkv_mkw['num_victims_injured'] = mki_mkv_mkw['num_victims_injured'].astype(int)
mki_mkv_mko['incident_id'] = mki_mkv_mko['incident_id'].astype(str)
mki_mkv_mko['victim_id'] = mki_mkv_mko['victim_id'].astype(str)
mki_mkv_mko['offender_id'] = mki_mkv_mko['offender_id'].astype(str)
#mki_mkv_mkw['victim_age'] = mki_mkv_mkw['victim_age'].astype(int)
mki_mkv_mko = mki_mkv_mko.drop_duplicates()
mki_mkv_mko.sample(3)
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
      <th>incident_id</th>
      <th>incidentdate</th>
      <th>city</th>
      <th>state</th>
      <th>num_offenders</th>
      <th>num_victims_killed</th>
      <th>num_victims_injured</th>
      <th>firstcod</th>
      <th>secondcod</th>
      <th>type</th>
      <th>...</th>
      <th>v_age</th>
      <th>v_race</th>
      <th>v_sex</th>
      <th>vorelationship</th>
      <th>offender_id</th>
      <th>o_age</th>
      <th>o_race</th>
      <th>o_sex</th>
      <th>suicide</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2977</th>
      <td>72</td>
      <td>2008-01-14</td>
      <td>Indianapolis</td>
      <td>IN</td>
      <td>5</td>
      <td>4</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Felony</td>
      <td>...</td>
      <td>24.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Undetermined</td>
      <td>100.0</td>
      <td>30.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Prison sentence</td>
    </tr>
    <tr>
      <th>3274</th>
      <td>110</td>
      <td>2006-07-04</td>
      <td>Gustine</td>
      <td>CA</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>...</td>
      <td>10.0</td>
      <td>White</td>
      <td>Male</td>
      <td>Child or stepchild</td>
      <td>146.0</td>
      <td>38.0</td>
      <td>White</td>
      <td>Male</td>
      <td>1.0</td>
      <td>Suicide</td>
    </tr>
    <tr>
      <th>1731</th>
      <td>271</td>
      <td>2015-07-15</td>
      <td>Holly Hill</td>
      <td>SC</td>
      <td>4</td>
      <td>4</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Felony</td>
      <td>...</td>
      <td>17.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Undetermined</td>
      <td>514.0</td>
      <td>27.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Prison sentence</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 27 columns</p>
</div>




```python
# Importing Life Expectancy Data for the U.S. by Gender
with open('data/le/U.S._State_Life_Expectancy_by_Sex__2020.csv') as f:
    le=pd.read_csv(f, delimiter=',')
```


```python
# US State to Abbreviations
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

abbv_to_states = dict(zip(us_state_to_abbrev.values(), us_state_to_abbrev.keys()))

states_to_regions = {
    'Washington': 'West', 'Oregon': 'West', 'California': 'West', 'Nevada': 'West',
    'Idaho': 'West', 'Montana': 'West', 'Wyoming': 'West', 'Utah': 'West',
    'Colorado': 'West', 'Alaska': 'West', 'Hawaii': 'West', 'Maine': 'Northeast',
    'Vermont': 'Northeast', 'New York': 'Northeast', 'New Hampshire': 'Northeast',
    'Massachusetts': 'Northeast', 'Rhode Island': 'Northeast', 'Connecticut': 'Northeast',
    'New Jersey': 'Northeast', 'Pennsylvania': 'Northeast', 'North Dakota': 'Midwest',
    'South Dakota': 'Midwest', 'Nebraska': 'Midwest', 'Kansas': 'Midwest',
    'Minnesota': 'Midwest', 'Iowa': 'Midwest', 'Missouri': 'Midwest', 'Wisconsin': 'Midwest',
    'Illinois': 'Midwest', 'Michigan': 'Midwest', 'Indiana': 'Midwest', 'Ohio': 'Midwest',
    'West Virginia': 'South', 'District of Columbia': 'South', 'Maryland': 'South',
    'Virginia': 'South', 'Kentucky': 'South', 'Tennessee': 'South', 'North Carolina': 'South',
    'Mississippi': 'South', 'Arkansas': 'South', 'Louisiana': 'South', 'Alabama': 'South',
    'Georgia': 'South', 'South Carolina': 'South', 'Florida': 'South', 'Delaware': 'South',
    'Arizona': 'Southwest', 'New Mexico': 'Southwest', 'Oklahoma': 'Southwest',
    'Texas': 'Southwest'}

regions_to_states = dict(zip(states_to_regions.values(), states_to_regions.keys()))

month_season = {
    1:'winter', 2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',
    7:'summer',8:'summer',9:'fall',10:'fall',11:'fall',12:'winter'
}
```


```python
#2020 Life Expectancy by State and Year
le_state = le.copy()
le_state['region'] = le_state['State'].map( states_to_regions)
pd.set_option('display.max_rows', 200)
le_state = le_state.replace({"State": us_state_to_abbrev})
le_state.head(5)
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
      <th>State</th>
      <th>Sex</th>
      <th>LE</th>
      <th>SE</th>
      <th>Quartile</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>Total</td>
      <td>73.2</td>
      <td>0.067</td>
      <td>71.9 - 75.3</td>
      <td>South</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>Total</td>
      <td>76.6</td>
      <td>0.176</td>
      <td>75.4 - 76.8</td>
      <td>West</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AZ</td>
      <td>Total</td>
      <td>76.3</td>
      <td>0.055</td>
      <td>75.4 - 76.8</td>
      <td>Southwest</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AR</td>
      <td>Total</td>
      <td>73.8</td>
      <td>0.086</td>
      <td>71.9 - 75.3</td>
      <td>South</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>Total</td>
      <td>79.0</td>
      <td>0.022</td>
      <td>78.1 - 80.7</td>
      <td>West</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging of 2020 State Life Expectancy and Mass Killing and Incident Tables
# Query definition
mk_vo_le_q = """ 
            SELECT *,le.le - v_age as yrs_life_lost
            FROM mki_mkv_mko
              LEFT JOIN le_state as le
              ON mki_mkv_mko.state = le.state and mki_mkv_mko.v_sex = le.sex
       
        """
# Query execution and convert to dataframe
mk_vo_le = sqldf(mk_vo_le_q).copy()
mk_vo_le.replace(to_replace=[None], value=np.nan, inplace=True)
mk_vo_le['incidentdate'] = pd.to_datetime(mk_vo_le['incidentdate'],format="%Y-%m-%d")
mk_vo_le['month'] = mk_vo_le['incidentdate'].dt.month.astype('category')
mk_vo_le['year'] = mk_vo_le['incidentdate'].dt.year
mk_vo_le['season'] = mk_vo_le['month'].map(month_season)
mk_vo_le['secondcod'] = mk_vo_le['secondcod'].fillna('None')
mk_vo_le['first_second_cod'] = mk_vo_le['firstcod'] + '-' + mk_vo_le['secondcod']
mk_vo_le = mk_vo_le[mk_vo_le['year'] <= 2022].copy()
mk_vo_le['season_month'] = mk_vo_le['season'] + '-' + mk_vo_le['month'].astype(str)
mk_vo_le['num_offenders'] = mk_vo_le['num_offenders'].astype(int)
mk_vo_le['region_state'] = mk_vo_le['region'] + '-' + mk_vo_le['state']
mk_vo_le['o_race_sex'] = mk_vo_le['o_race'] + '-' + mk_vo_le['o_sex']
mk_vo_le['v_race_sex'] = mk_vo_le['v_race'] + '-' + mk_vo_le['v_sex']
mk_vo_le['city_state'] = mk_vo_le['city'] + '-' + mk_vo_le['state']
mk_vo_le = mk_vo_le.drop_duplicates()
mk_vo_le = mk_vo_le.dropna().reset_index()
mk_vo_le = mk_vo_le.drop(columns=["State","Sex","Quartile","SE","LE","index","year","latitude","longitude","narrative",]) #"firstcod","secondcod"


# function for age range of offender
def age_range(x):
    if x < 21:
        return 'Under 21'
    elif x >= 21 and x <= 30:
        return '21-30 Years'
    elif x >= 31 and x <= 50:
        return '31-50 Years'
    else:
        return '51+ Years'
mk_vo_le['o_age_range'] = mk_vo_le['o_age'].apply(age_range)


pd.set_option('display.max_columns', None)
mk_vo_le.sample(3)
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
      <th>incident_id</th>
      <th>incidentdate</th>
      <th>city</th>
      <th>state</th>
      <th>num_offenders</th>
      <th>num_victims_killed</th>
      <th>num_victims_injured</th>
      <th>firstcod</th>
      <th>secondcod</th>
      <th>type</th>
      <th>situation_type</th>
      <th>location_type</th>
      <th>location</th>
      <th>victim_id</th>
      <th>v_age</th>
      <th>v_race</th>
      <th>v_sex</th>
      <th>vorelationship</th>
      <th>offender_id</th>
      <th>o_age</th>
      <th>o_race</th>
      <th>o_sex</th>
      <th>suicide</th>
      <th>outcome</th>
      <th>region</th>
      <th>yrs_life_lost</th>
      <th>month</th>
      <th>season</th>
      <th>first_second_cod</th>
      <th>season_month</th>
      <th>region_state</th>
      <th>o_race_sex</th>
      <th>v_race_sex</th>
      <th>city_state</th>
      <th>o_age_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2098</th>
      <td>115</td>
      <td>2009-09-17</td>
      <td>Naples</td>
      <td>FL</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>599</td>
      <td>32.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Spouse</td>
      <td>155.0</td>
      <td>33.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Death sentence</td>
      <td>South</td>
      <td>48.5</td>
      <td>9</td>
      <td>fall</td>
      <td>Stabbing-None</td>
      <td>fall-9</td>
      <td>South-FL</td>
      <td>Black-Male</td>
      <td>Black-Female</td>
      <td>Naples-FL</td>
      <td>31-50 Years</td>
    </tr>
    <tr>
      <th>2768</th>
      <td>103</td>
      <td>2006-05-21</td>
      <td>Baton Rouge</td>
      <td>LA</td>
      <td>1</td>
      <td>5</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>House of Worship</td>
      <td>House of worship</td>
      <td>544</td>
      <td>68.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Other familial relationship</td>
      <td>139.0</td>
      <td>25.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Death sentence</td>
      <td>South</td>
      <td>8.4</td>
      <td>5</td>
      <td>spring</td>
      <td>Shooting-None</td>
      <td>spring-5</td>
      <td>South-LA</td>
      <td>Black-Male</td>
      <td>Black-Female</td>
      <td>Baton Rouge-LA</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>2414</th>
      <td>68</td>
      <td>2008-02-23</td>
      <td>Yorba Linda</td>
      <td>CA</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>359</td>
      <td>8.0</td>
      <td>Asian/Pacific Islander</td>
      <td>Female</td>
      <td>Child or stepchild</td>
      <td>94.0</td>
      <td>41.0</td>
      <td>Asian/Pacific Islander</td>
      <td>Male</td>
      <td>1.0</td>
      <td>Suicide</td>
      <td>West</td>
      <td>74.0</td>
      <td>2</td>
      <td>winter</td>
      <td>Shooting-None</td>
      <td>winter-2</td>
      <td>West-CA</td>
      <td>Asian/Pacific Islander-Male</td>
      <td>Asian/Pacific Islander-Female</td>
      <td>Yorba Linda-CA</td>
      <td>31-50 Years</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Columns for Easy Viewing
mk_vo_le.columns
```




    Index(['incident_id', 'incidentdate', 'city', 'state', 'num_offenders',
           'num_victims_killed', 'num_victims_injured', 'firstcod', 'secondcod',
           'type', 'situation_type', 'location_type', 'location', 'victim_id',
           'v_age', 'v_race', 'v_sex', 'vorelationship', 'offender_id', 'o_age',
           'o_race', 'o_sex', 'suicide', 'outcome', 'region', 'yrs_life_lost',
           'month', 'season', 'first_second_cod', 'season_month', 'region_state',
           'o_race_sex', 'v_race_sex', 'city_state', 'o_age_range'],
          dtype='object')




```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from statsmodels.formula.api import ols

# Load the data
df_reg = mk_vo_le.copy()

skip_columns = ['incident_id', 'incidentdate', 'num_offenders',
       'num_victims_killed', 'num_victims_injured','victim_id','offender_id','yrs_life_lost', 'year']
encoders = {}

for column in df_reg.select_dtypes(include=['object']).columns:
    if column in skip_columns:
        continue
    if df_reg[column].dtype == 'object':
        le = LabelEncoder()
        df_reg[column] = le.fit_transform(df_reg[column])
        encoders[column] = le

inverse_mapping = {}

for column, le in encoders.items():
    inverse_mapping[column] = dict(zip(le.transform(le.classes_), le.classes_))
```


```python
mapped_model_values = pd.DataFrame.from_dict(inverse_mapping, orient='columns').reset_index()
#mapped_model_values.columns
mapped_model_values_pivot = mapped_model_values.melt(id_vars=['index'],value_vars=['index', 'type', 'situation_type', 'vorelationship','season_month','o_race_sex',
       'v_race_sex', 'o_age_range','city_state'])
mapped_model_values_pivot.head(3)
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
      <th>index</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>type</td>
      <td>Family</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>type</td>
      <td>Felony</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>type</td>
      <td>Other</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_reg 
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
      <th>incident_id</th>
      <th>incidentdate</th>
      <th>city</th>
      <th>state</th>
      <th>num_offenders</th>
      <th>num_victims_killed</th>
      <th>num_victims_injured</th>
      <th>firstcod</th>
      <th>secondcod</th>
      <th>type</th>
      <th>situation_type</th>
      <th>location_type</th>
      <th>location</th>
      <th>victim_id</th>
      <th>v_age</th>
      <th>v_race</th>
      <th>v_sex</th>
      <th>vorelationship</th>
      <th>offender_id</th>
      <th>o_age</th>
      <th>o_race</th>
      <th>o_sex</th>
      <th>suicide</th>
      <th>outcome</th>
      <th>region</th>
      <th>yrs_life_lost</th>
      <th>month</th>
      <th>season</th>
      <th>first_second_cod</th>
      <th>season_month</th>
      <th>region_state</th>
      <th>o_race_sex</th>
      <th>v_race_sex</th>
      <th>city_state</th>
      <th>o_age_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>69</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2805</td>
      <td>28.0</td>
      <td>5</td>
      <td>1</td>
      <td>23</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>47.8</td>
      <td>11</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>40</td>
      <td>11</td>
      <td>11</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>69</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2806</td>
      <td>38.0</td>
      <td>5</td>
      <td>1</td>
      <td>23</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>37.8</td>
      <td>11</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>40</td>
      <td>11</td>
      <td>11</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>69</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2807</td>
      <td>35.0</td>
      <td>5</td>
      <td>0</td>
      <td>23</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>45.9</td>
      <td>11</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>40</td>
      <td>11</td>
      <td>10</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>69</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2808</td>
      <td>22.0</td>
      <td>2</td>
      <td>1</td>
      <td>23</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>53.8</td>
      <td>11</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>40</td>
      <td>11</td>
      <td>5</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>69</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2809</td>
      <td>40.0</td>
      <td>5</td>
      <td>0</td>
      <td>23</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>40.9</td>
      <td>11</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>40</td>
      <td>11</td>
      <td>10</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2832</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>264</td>
      <td>42</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>9</td>
      <td>507</td>
      <td>21.0</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>2</td>
      <td>54.1</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>9</td>
      <td>32</td>
      <td>5</td>
      <td>5</td>
      <td>273</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2833</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>264</td>
      <td>42</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>9</td>
      <td>508</td>
      <td>46.0</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>4</td>
      <td>2</td>
      <td>29.1</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>9</td>
      <td>32</td>
      <td>5</td>
      <td>5</td>
      <td>273</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2834</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>264</td>
      <td>42</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>9</td>
      <td>508</td>
      <td>46.0</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>2</td>
      <td>29.1</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>9</td>
      <td>32</td>
      <td>5</td>
      <td>5</td>
      <td>273</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>264</td>
      <td>42</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>9</td>
      <td>509</td>
      <td>55.0</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>4</td>
      <td>2</td>
      <td>25.1</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>9</td>
      <td>32</td>
      <td>5</td>
      <td>4</td>
      <td>273</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2836</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>264</td>
      <td>42</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>9</td>
      <td>509</td>
      <td>55.0</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>11</td>
      <td>2</td>
      <td>25.1</td>
      <td>1</td>
      <td>3</td>
      <td>23</td>
      <td>9</td>
      <td>32</td>
      <td>5</td>
      <td>4</td>
      <td>273</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2837 rows × 35 columns</p>
</div>




```python
# Perform linear regression

# forumula without encoded variables
formula1 = 'yrs_life_lost ~ city_state + vorelationship + season_month + v_race_sex + situation_type + type + o_age_range'
model1 = ols(formula1, mk_vo_le).fit()

# formula with encoded variables
formula = 'yrs_life_lost ~ C(city_state) + C(vorelationship) + C(season_month) + C(v_race_sex) + C(situation_type) + C(type) + C(o_age_range)'
model = ols(formula, df_reg).fit()


```


```python
# Execute ANOVA Analysis
aov_table = smf.stats.anova_lm(model, typ=2)
aov_table_df = pd.DataFrame(aov_table)

# Print summary of ANOVA Analysis
aov_table_df['Significant']=aov_table_df['PR(>F)'] < 0.05
aov_table_df
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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
      <th>Significant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(city_state)</th>
      <td>291711.637512</td>
      <td>365.0</td>
      <td>4.908847</td>
      <td>2.476104e-126</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C(vorelationship)</th>
      <td>157397.059172</td>
      <td>28.0</td>
      <td>34.526868</td>
      <td>7.008495e-154</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C(season_month)</th>
      <td>8425.756643</td>
      <td>11.0</td>
      <td>4.704731</td>
      <td>3.653194e-07</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C(v_race_sex)</th>
      <td>20096.747466</td>
      <td>11.0</td>
      <td>11.221520</td>
      <td>1.378981e-20</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C(situation_type)</th>
      <td>10458.535187</td>
      <td>13.0</td>
      <td>4.941356</td>
      <td>1.213847e-08</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C(type)</th>
      <td>3906.560401</td>
      <td>6.0</td>
      <td>3.999097</td>
      <td>5.459384e-04</td>
      <td>True</td>
    </tr>
    <tr>
      <th>C(o_age_range)</th>
      <td>2218.782849</td>
      <td>3.0</td>
      <td>4.542680</td>
      <td>3.516043e-03</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>390581.483153</td>
      <td>2399.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create summary of regeression model
cond = model.condition_number
r2 = model.rsquared
r2a = model.rsquared_adj
f1 = model.fvalue
f1_p = model.f_pvalue
summary = model.summary2()
mse = model.mse_resid

# Print summary of regression model
print("Condition No:", cond)
print("R-Squared:",r2)
print("Adjusted R-Squared:",r2a)
print("F-Statistic:",f1)
print("F-Statistic P-Value:",f1_p)
print("Mean Squared Error:", mse )
#print("Model Variables:",model.params)

#print(summary)
```

    Condition No: 710.3868145950091
    R-Squared: 0.6795165811004162
    Adjusted R-Squared: 0.6211375673200419
    F-Statistic: 11.639740672166916
    F-Statistic P-Value: 0.0
    Mean Squared Error: 162.81012219786038



```python
#Model summary
print(summary)
```

                         Results: Ordinary least squares
    ==========================================================================
    Model:                 OLS                Adj. R-squared:       0.621     
    Dependent Variable:    yrs_life_lost      AIC:                  22898.9686
    Date:                  2023-02-11 11:29   BIC:                  25505.2887
    No. Observations:      2837               Log-Likelihood:       -11011.   
    Df Model:              437                F-statistic:          11.64     
    Df Residuals:          2399               Prob (F-statistic):   0.00      
    R-squared:             0.680              Scale:                162.81    
    --------------------------------------------------------------------------
                             Coef.   Std.Err.    t    P>|t|   [0.025   0.975] 
    --------------------------------------------------------------------------
    Intercept                47.0767  13.5936  3.4632 0.0005  20.4204  73.7331
    C(city_state)[T.1]        1.0243   9.6799  0.1058 0.9157 -17.9575  20.0061
    C(city_state)[T.2]      -56.2917  10.3516 -5.4380 0.0000 -76.5907 -35.9928
    C(city_state)[T.3]      -10.3151   8.4458 -1.2213 0.2221 -26.8770   6.2468
    C(city_state)[T.4]      -20.5277  10.7236 -1.9143 0.0557 -41.5562   0.5008
    C(city_state)[T.5]       -8.5639   8.2375 -1.0396 0.2986 -24.7173   7.5896
    C(city_state)[T.6]      -13.3438   9.5280 -1.4005 0.1615 -32.0278   5.3401
    C(city_state)[T.7]      -17.6843  10.9456 -1.6157 0.1063 -39.1481   3.7794
    C(city_state)[T.8]       -0.4273  13.0835 -0.0327 0.9739 -26.0834  25.2288
    C(city_state)[T.9]       -5.2944  10.0389 -0.5274 0.5980 -24.9802  14.3915
    C(city_state)[T.10]     -11.4501   9.9161 -1.1547 0.2483 -30.8951   7.9949
    C(city_state)[T.11]     -25.6125   9.0304 -2.8363 0.0046 -43.3207  -7.9043
    C(city_state)[T.12]      13.6674   8.7819  1.5563 0.1198  -3.5535  30.8884
    C(city_state)[T.13]       1.2147   9.7449  0.1247 0.9008 -17.8945  20.3240
    C(city_state)[T.14]      -5.3790   9.2273 -0.5829 0.5600 -23.4733  12.7153
    C(city_state)[T.15]     -38.1038   9.8831 -3.8554 0.0001 -57.4841 -18.7235
    C(city_state)[T.16]      -2.2492   9.3809 -0.2398 0.8105 -20.6446  16.1462
    C(city_state)[T.17]       2.3984   8.0729  0.2971 0.7664 -13.4321  18.2289
    C(city_state)[T.18]     -12.7076  11.8287 -1.0743 0.2828 -35.9031  10.4878
    C(city_state)[T.19]     -25.9824  10.7160 -2.4246 0.0154 -46.9960  -4.9687
    C(city_state)[T.20]       1.4408  11.4535  0.1258 0.8999 -21.0191  23.9006
    C(city_state)[T.21]     -34.4912  11.4356 -3.0161 0.0026 -56.9159 -12.0664
    C(city_state)[T.22]     -20.8331   9.2672 -2.2481 0.0247 -39.0057  -2.6606
    C(city_state)[T.23]      -5.0744  10.1340 -0.5007 0.6166 -24.9467  14.7980
    C(city_state)[T.24]     -36.7038  10.0454 -3.6538 0.0003 -56.4024 -17.0052
    C(city_state)[T.25]     -10.7234   8.5431 -1.2552 0.2095 -27.4759   6.0292
    C(city_state)[T.26]     -19.0243   8.4532 -2.2505 0.0245 -35.6006  -2.4480
    C(city_state)[T.27]       5.8108   8.0023  0.7261 0.4678  -9.8814  21.5030
    C(city_state)[T.28]     -22.6591   8.7818 -2.5802 0.0099 -39.8799  -5.4384
    C(city_state)[T.29]      -4.7021  19.7840 -0.2377 0.8122 -43.4977  34.0934
    C(city_state)[T.30]      -0.8515  10.2430 -0.0831 0.9338 -20.9375  19.2345
    C(city_state)[T.31]      -8.6567  10.4556 -0.8279 0.4078 -29.1596  11.8462
    C(city_state)[T.32]       6.5083   7.0314  0.9256 0.3547  -7.2800  20.2966
    C(city_state)[T.33]     -14.4330   8.8138 -1.6375 0.1016 -31.7164   2.8505
    C(city_state)[T.34]     -29.8502  12.0723 -2.4726 0.0135 -53.5234  -6.1769
    C(city_state)[T.35]     -17.0210   9.4280 -1.8054 0.0711 -35.5089   1.4669
    C(city_state)[T.36]      -8.5823  11.1876 -0.7671 0.4431 -30.5207  13.3562
    C(city_state)[T.37]      -9.4850   9.9916 -0.9493 0.3426 -29.0782  10.1081
    C(city_state)[T.38]     -10.3790   9.3944 -1.1048 0.2694 -28.8009   8.0429
    C(city_state)[T.39]     -20.6165   8.8955 -2.3176 0.0206 -38.0601  -3.1728
    C(city_state)[T.40]     -68.2000  12.1228 -5.6258 0.0000 -91.9722 -44.4279
    C(city_state)[T.41]     -19.5125  10.2873 -1.8967 0.0580 -39.6855   0.6605
    C(city_state)[T.42]     -18.3414   9.3479 -1.9621 0.0499 -36.6722  -0.0107
    C(city_state)[T.43]     -18.9323   8.6153 -2.1975 0.0281 -35.8266  -2.0380
    C(city_state)[T.44]     -15.9772  10.5420 -1.5156 0.1298 -36.6496   4.6952
    C(city_state)[T.45]     -25.9592   8.9592 -2.8975 0.0038 -43.5278  -8.3907
    C(city_state)[T.46]      -2.8120   8.4388 -0.3332 0.7390 -19.3601  13.7362
    C(city_state)[T.47]      -2.3031   9.3015 -0.2476 0.8045 -20.5430  15.9368
    C(city_state)[T.48]     -52.8736   9.4939 -5.5692 0.0000 -71.4906 -34.2565
    C(city_state)[T.49]       0.9743   8.4035  0.1159 0.9077 -15.5046  17.4532
    C(city_state)[T.50]      -6.1074  10.2409 -0.5964 0.5510 -26.1894  13.9746
    C(city_state)[T.51]       2.0270   9.3844  0.2160 0.8290 -16.3754  20.4295
    C(city_state)[T.52]       2.2225   9.1677  0.2424 0.8085 -15.7550  20.2000
    C(city_state)[T.53]      -1.2000   9.8475 -0.1219 0.9030 -20.5104  18.1105
    C(city_state)[T.54]     -49.6020  10.2557 -4.8365 0.0000 -69.7130 -29.4910
    C(city_state)[T.55]     -12.9937  11.0996 -1.1706 0.2419 -34.7596   8.7721
    C(city_state)[T.56]      -2.0538  10.9354 -0.1878 0.8510 -23.4977  19.3901
    C(city_state)[T.57]     -20.1956  10.3271 -1.9556 0.0506 -40.4466   0.0554
    C(city_state)[T.58]     -14.8535   8.6602 -1.7151 0.0864 -31.8357   2.1288
    C(city_state)[T.59]     -11.6516  10.1252 -1.1508 0.2499 -31.5065   8.2034
    C(city_state)[T.60]     -21.9905   8.8531 -2.4839 0.0131 -39.3509  -4.6300
    C(city_state)[T.61]     -29.7368  11.7722 -2.5260 0.0116 -52.8216  -6.6520
    C(city_state)[T.62]      -9.6051   6.7982 -1.4129 0.1578 -22.9360   3.7259
    C(city_state)[T.63]      18.1726  17.5824  1.0336 0.3014 -16.3057  52.6509
    C(city_state)[T.64]     -20.1759  10.2501 -1.9684 0.0491 -40.2760  -0.0759
    C(city_state)[T.65]     -26.2463  10.9051 -2.4068 0.0162 -47.6308  -4.8619
    C(city_state)[T.66]       0.3782  10.9015  0.0347 0.9723 -20.9990  21.7555
    C(city_state)[T.67]     -26.8951  10.3396 -2.6012 0.0093 -47.1706  -6.6195
    C(city_state)[T.68]       4.1930   7.1367  0.5875 0.5569  -9.8017  18.1877
    C(city_state)[T.69]       5.9488  10.5647  0.5631 0.5734 -14.7681  26.6656
    C(city_state)[T.70]      -8.9307  10.6401 -0.8393 0.4014 -29.7955  11.9341
    C(city_state)[T.71]     -18.5846   8.7090 -2.1340 0.0329 -35.6625  -1.5068
    C(city_state)[T.72]     -21.5523  20.1590 -1.0691 0.2851 -61.0832  17.9785
    C(city_state)[T.73]     -14.6914  10.7855 -1.3621 0.1733 -35.8413   6.4585
    C(city_state)[T.74]     -21.9439  10.5544 -2.0791 0.0377 -42.6405  -1.2473
    C(city_state)[T.75]      -5.6270   8.0940 -0.6952 0.4870 -21.4989  10.2449
    C(city_state)[T.76]      -7.8194   9.2768 -0.8429 0.3994 -26.0107  10.3720
    C(city_state)[T.77]     -23.1642   9.1193 -2.5401 0.0111 -41.0468  -5.2816
    C(city_state)[T.78]     -24.1178  10.8550 -2.2218 0.0264 -45.4041  -2.8316
    C(city_state)[T.79]     -21.7512   9.3741 -2.3203 0.0204 -40.1334  -3.3689
    C(city_state)[T.80]       5.5654   7.7849  0.7149 0.4747  -9.7003  20.8312
    C(city_state)[T.81]       8.3843   9.3345  0.8982 0.3692  -9.9203  26.6888
    C(city_state)[T.82]     -23.3219   9.4107 -2.4782 0.0133 -41.7757  -4.8680
    C(city_state)[T.83]     -14.9067   8.8115 -1.6917 0.0908 -32.1857   2.3723
    C(city_state)[T.84]     -10.6973  11.7680 -0.9090 0.3634 -33.7739  12.3793
    C(city_state)[T.85]     -31.9527   8.7858 -3.6368 0.0003 -49.1813 -14.7241
    C(city_state)[T.86]     -11.2273   9.6329 -1.1655 0.2439 -30.1171   7.6624
    C(city_state)[T.87]     -27.7911   8.6839 -3.2003 0.0014 -44.8198 -10.7624
    C(city_state)[T.88]      -7.8457   8.2479 -0.9512 0.3416 -24.0195   8.3281
    C(city_state)[T.89]      -4.1141  10.1703 -0.4045 0.6859 -24.0577  15.8295
    C(city_state)[T.90]      14.5588   9.6083  1.5152 0.1298  -4.2827  33.4003
    C(city_state)[T.91]       6.5792   8.9620  0.7341 0.4629 -10.9948  24.1533
    C(city_state)[T.92]     -14.1849   7.5636 -1.8754 0.0609 -29.0169   0.6470
    C(city_state)[T.93]      -8.6116   8.2734 -1.0409 0.2980 -24.8354   7.6122
    C(city_state)[T.94]     -12.9959  13.1904 -0.9853 0.3246 -38.8616  12.8698
    C(city_state)[T.95]      -5.0001   9.1556 -0.5461 0.5850 -22.9539  12.9537
    C(city_state)[T.96]      -5.8008   9.7257 -0.5964 0.5509 -24.8725  13.2709
    C(city_state)[T.97]      -9.0856  10.4475 -0.8696 0.3846 -29.5727  11.4015
    C(city_state)[T.98]     -71.5631  10.6966 -6.6903 0.0000 -92.5386 -50.5876
    C(city_state)[T.99]       0.9386   9.2862  0.1011 0.9195 -17.2711  19.1484
    C(city_state)[T.100]    -17.3874   9.4439 -1.8411 0.0657 -35.9065   1.1317
    C(city_state)[T.101]     -6.3268  10.0054 -0.6323 0.5272 -25.9469  13.2933
    C(city_state)[T.102]     10.0160   9.0904  1.1018 0.2706  -7.8098  27.8417
    C(city_state)[T.103]      4.5770   9.0367  0.5065 0.6126 -13.1435  22.2976
    C(city_state)[T.104]      8.8209  10.9643  0.8045 0.4212 -12.6796  30.3214
    C(city_state)[T.105]     -4.1515  10.3057 -0.4028 0.6871 -24.3604  16.0575
    C(city_state)[T.106]     -4.5927  10.9832 -0.4182 0.6759 -26.1301  16.9448
    C(city_state)[T.107]     -4.9101  20.0682 -0.2447 0.8067 -44.2630  34.4428
    C(city_state)[T.108]    -20.8302  10.4852 -1.9866 0.0471 -41.3911  -0.2692
    C(city_state)[T.109]     -8.3545   9.5309 -0.8766 0.3808 -27.0442  10.3352
    C(city_state)[T.110]    -12.7240   9.0056 -1.4129 0.1578 -30.3836   4.9356
    C(city_state)[T.111]    -40.9006   9.9481 -4.1114 0.0000 -60.4083 -21.3929
    C(city_state)[T.112]     23.1906   9.9155  2.3388 0.0194   3.7467  42.6345
    C(city_state)[T.113]      2.4573   9.4463  0.2601 0.7948 -16.0664  20.9810
    C(city_state)[T.114]     -9.8873   9.1408 -1.0817 0.2795 -27.8120   8.0374
    C(city_state)[T.115]    -46.7290  10.1603 -4.5992 0.0000 -66.6529 -26.8052
    C(city_state)[T.116]     -2.5622   9.9614 -0.2572 0.7970 -22.0961  16.9717
    C(city_state)[T.117]     -5.6959   9.9874 -0.5703 0.5685 -25.2807  13.8888
    C(city_state)[T.118]    -30.6955  10.1494 -3.0244 0.0025 -50.5980 -10.7930
    C(city_state)[T.119]    -16.2386   8.5750 -1.8937 0.0584 -33.0537   0.5764
    C(city_state)[T.120]    -15.5227   9.2715 -1.6742 0.0942 -33.7037   2.6584
    C(city_state)[T.121]     -1.8650  11.7751 -0.1584 0.8742 -24.9554  21.2254
    C(city_state)[T.122]     -6.9706   8.8173 -0.7906 0.4293 -24.2608  10.3196
    C(city_state)[T.123]      4.4298  10.2657  0.4315 0.6661 -15.7007  24.5603
    C(city_state)[T.124]    -13.7800  10.1088 -1.3632 0.1730 -33.6029   6.0428
    C(city_state)[T.125]      5.2081  10.6307  0.4899 0.6242 -15.6381  26.0544
    C(city_state)[T.126]    -15.6560  10.2931 -1.5210 0.1284 -35.8404   4.5283
    C(city_state)[T.127]    -11.2963   9.6207 -1.1742 0.2404 -30.1622   7.5695
    C(city_state)[T.128]     -0.0614   9.0789 -0.0068 0.9946 -17.8646  17.7418
    C(city_state)[T.129]    -19.0891   9.4615 -2.0176 0.0437 -37.6426  -0.5356
    C(city_state)[T.130]     -8.1071   9.0194 -0.8988 0.3688 -25.7938   9.5796
    C(city_state)[T.131]     -4.0784   8.9895 -0.4537 0.6501 -21.7064  13.5496
    C(city_state)[T.132]    -16.2148  10.4862 -1.5463 0.1222 -36.7777   4.3482
    C(city_state)[T.133]     -7.6864  10.1317 -0.7586 0.4481 -27.5542  12.1814
    C(city_state)[T.134]    -46.7948  11.5049 -4.0674 0.0000 -69.3555 -24.2341
    C(city_state)[T.135]      2.0136   9.6982  0.2076 0.8355 -17.0040  21.0313
    C(city_state)[T.136]    -54.3223  14.9947 -3.6228 0.0003 -83.7262 -24.9184
    C(city_state)[T.137]    -12.0994   9.1726 -1.3191 0.1873 -30.0864   5.8875
    C(city_state)[T.138]      5.3839  10.1073  0.5327 0.5943 -14.4361  25.2039
    C(city_state)[T.139]    -21.6480  10.3761 -2.0863 0.0371 -41.9951  -1.3009
    C(city_state)[T.140]     -5.7583   8.7765 -0.6561 0.5118 -22.9687  11.4521
    C(city_state)[T.141]    -24.4413   9.3023 -2.6274 0.0087 -42.6827  -6.1999
    C(city_state)[T.142]     -3.2715   8.3448 -0.3920 0.6951 -19.6353  13.0922
    C(city_state)[T.143]    -23.6375   9.7696 -2.4195 0.0156 -42.7953  -4.4797
    C(city_state)[T.144]    -20.9038   7.4711 -2.7980 0.0052 -35.5542  -6.2534
    C(city_state)[T.145]      0.3284  10.2639  0.0320 0.9745 -19.7986  20.4553
    C(city_state)[T.146]    -16.3869   8.6484 -1.8948 0.0582 -33.3461   0.5723
    C(city_state)[T.147]    -15.2083  10.8262 -1.4048 0.1602 -36.4381   6.0214
    C(city_state)[T.148]     -7.6467   7.0809 -1.0799 0.2803 -21.5320   6.2387
    C(city_state)[T.149]    -18.9644  10.7217 -1.7688 0.0771 -39.9891   2.0602
    C(city_state)[T.150]    -10.2249   9.4875 -1.0777 0.2813 -28.8295   8.3797
    C(city_state)[T.151]      1.3568   7.7612  0.1748 0.8612 -13.8625  16.5761
    C(city_state)[T.152]    -21.8808  13.6242 -1.6060 0.1084 -48.5974   4.8357
    C(city_state)[T.153]      1.5814   8.6951  0.1819 0.8557 -15.4692  18.6321
    C(city_state)[T.154]     -8.0268   7.9907 -1.0045 0.3152 -23.6962   7.6425
    C(city_state)[T.155]    -52.7419   7.6266 -6.9155 0.0000 -67.6974 -37.7865
    C(city_state)[T.156]    -24.6645  10.3101 -2.3923 0.0168 -44.8822  -4.4469
    C(city_state)[T.157]    -35.4292   9.6565 -3.6690 0.0002 -54.3652 -16.4933
    C(city_state)[T.158]    -23.2609  10.7991 -2.1540 0.0313 -44.4374  -2.0844
    C(city_state)[T.159]    -15.1796   9.3959 -1.6156 0.1063 -33.6046   3.2454
    C(city_state)[T.160]     -9.0949   7.0422 -1.2915 0.1967 -22.9044   4.7146
    C(city_state)[T.161]    -39.4817   7.5575 -5.2242 0.0000 -54.3016 -24.6619
    C(city_state)[T.162]    -10.3537  10.6062 -0.9762 0.3291 -31.1520  10.4446
    C(city_state)[T.163]     -3.9524   9.4854 -0.4167 0.6769 -22.5528  14.6480
    C(city_state)[T.164]     15.8764   9.9090  1.6022 0.1092  -3.5547  35.3074
    C(city_state)[T.165]    -11.1776  11.4179 -0.9790 0.3277 -33.5677  11.2124
    C(city_state)[T.166]    -15.3543   9.1413 -1.6797 0.0932 -33.2800   2.5714
    C(city_state)[T.167]    -24.3544   9.8552 -2.4712 0.0135 -43.6800  -5.0288
    C(city_state)[T.168]     -2.7145  11.8372 -0.2293 0.8186 -25.9267  20.4977
    C(city_state)[T.169]      7.5896  11.4477  0.6630 0.5074 -14.8587  30.0380
    C(city_state)[T.170]    -17.1695   9.5271 -1.8022 0.0716 -35.8518   1.5127
    C(city_state)[T.171]     -3.8166   7.5111 -0.5081 0.6114 -18.5455  10.9123
    C(city_state)[T.172]    -13.8057   9.7709 -1.4129 0.1578 -32.9660   5.3547
    C(city_state)[T.173]    -17.5941  10.9872 -1.6013 0.1094 -39.1394   3.9512
    C(city_state)[T.174]      3.7425  10.3924  0.3601 0.7188 -16.6366  24.1216
    C(city_state)[T.175]      6.9363   7.7058  0.9001 0.3681  -8.1745  22.0471
    C(city_state)[T.176]    -17.7889   8.4784 -2.0981 0.0360 -34.4146  -1.1631
    C(city_state)[T.177]     25.4917  12.8529  1.9833 0.0474   0.2877  50.6957
    C(city_state)[T.178]    -14.5118   9.5879 -1.5136 0.1303 -33.3133   4.2896
    C(city_state)[T.179]     -3.3651   8.8414 -0.3806 0.7035 -20.7027  13.9724
    C(city_state)[T.180]    -24.5698  19.8305 -1.2390 0.2155 -63.4564  14.3168
    C(city_state)[T.181]     -7.9022   9.6835 -0.8160 0.4146 -26.8911  11.0868
    C(city_state)[T.182]     -3.0764   9.5558 -0.3219 0.7475 -21.8149  15.6620
    C(city_state)[T.183]      0.1811  11.1722  0.0162 0.9871 -21.7270  22.0892
    C(city_state)[T.184]     -8.7005   7.2609 -1.1983 0.2309 -22.9389   5.5379
    C(city_state)[T.185]     -1.3642   9.0532 -0.1507 0.8802 -19.1171  16.3886
    C(city_state)[T.186]     -0.5943  10.2399 -0.0580 0.9537 -20.6743  19.4857
    C(city_state)[T.187]    -28.3966   9.8775 -2.8749 0.0041 -47.7658  -9.0273
    C(city_state)[T.188]     -4.0225   9.0719 -0.4434 0.6575 -21.8120  13.7670
    C(city_state)[T.189]      0.7844  11.2056  0.0700 0.9442 -21.1893  22.7581
    C(city_state)[T.190]     22.6407  11.7781  1.9223 0.0547  -0.4556  45.7371
    C(city_state)[T.191]     -9.5765   8.9887 -1.0654 0.2868 -27.2029   8.0500
    C(city_state)[T.192]    -14.9801   9.1324 -1.6403 0.1011 -32.8883   2.9280
    C(city_state)[T.193]      3.3698   9.1353  0.3689 0.7122 -14.5441  21.2837
    C(city_state)[T.194]    -16.8580   8.5752 -1.9659 0.0494 -33.6735  -0.0425
    C(city_state)[T.195]     -3.5599  10.3485 -0.3440 0.7309 -23.8527  16.7330
    C(city_state)[T.196]     -4.5882  10.4272 -0.4400 0.6600 -25.0354  15.8589
    C(city_state)[T.197]     -5.1539   8.5572 -0.6023 0.5470 -21.9342  11.6264
    C(city_state)[T.198]     27.8094  12.2678  2.2669 0.0235   3.7528  51.8661
    C(city_state)[T.199]    -46.0850  11.8842 -3.8778 0.0001 -69.3892 -22.7807
    C(city_state)[T.200]     -7.6015   7.3705 -1.0313 0.3025 -22.0547   6.8517
    C(city_state)[T.201]     -2.1542   9.2677 -0.2324 0.8162 -20.3276  16.0193
    C(city_state)[T.202]     -5.5091   7.9660 -0.6916 0.4893 -21.1301  10.1119
    C(city_state)[T.203]     -7.3193   8.7124 -0.8401 0.4009 -24.4039   9.7653
    C(city_state)[T.204]     -1.0258  12.4902 -0.0821 0.9345 -25.5185  23.4668
    C(city_state)[T.205]     -3.4368  10.7606 -0.3194 0.7495 -24.5379  17.6643
    C(city_state)[T.206]     10.3170  10.6939  0.9648 0.3348 -10.6533  31.2872
    C(city_state)[T.207]    -10.1481  10.1822 -0.9967 0.3190 -30.1150   9.8188
    C(city_state)[T.208]     -5.5530   9.3371 -0.5947 0.5521 -23.8626  12.7565
    C(city_state)[T.209]      4.0556  10.4574  0.3878 0.6982 -16.4508  24.5621
    C(city_state)[T.210]     -6.0475   9.3119 -0.6494 0.5161 -24.3078  12.2127
    C(city_state)[T.211]     -9.7120  10.7095 -0.9069 0.3646 -30.7128  11.2889
    C(city_state)[T.212]     12.2027  10.7196  1.1384 0.2551  -8.8179  33.2234
    C(city_state)[T.213]      0.0490  10.9374  0.0045 0.9964 -21.3988  21.4968
    C(city_state)[T.214]     12.7541   9.7750  1.3048 0.1921  -6.4143  31.9225
    C(city_state)[T.215]     -9.8585  14.5616 -0.6770 0.4985 -38.4132  18.6962
    C(city_state)[T.216]      2.9153   9.8864  0.2949 0.7681 -16.4714  22.3020
    C(city_state)[T.217]    -17.2365   8.6463 -1.9935 0.0463 -34.1914  -0.2816
    C(city_state)[T.218]     -3.6820   7.2451 -0.5082 0.6114 -17.8893  10.5254
    C(city_state)[T.219]     15.9661  11.3206  1.4104 0.1586  -6.2331  38.1654
    C(city_state)[T.220]     -2.7085   7.6147 -0.3557 0.7221 -17.6406  12.2236
    C(city_state)[T.221]    -24.4630   9.4183 -2.5974 0.0095 -42.9319  -5.9941
    C(city_state)[T.222]    -14.8288  11.8963 -1.2465 0.2127 -38.1570   8.4994
    C(city_state)[T.223]     20.0431   8.5279  2.3503 0.0188   3.3202  36.7660
    C(city_state)[T.224]     35.3991   9.4565  3.7433 0.0002  16.8553  53.9429
    C(city_state)[T.225]    -25.7250  10.1473 -2.5351 0.0113 -45.6234  -5.8265
    C(city_state)[T.226]    -19.7679  15.3262 -1.2898 0.1972 -49.8219  10.2861
    C(city_state)[T.227]    -23.0720   8.8329 -2.6121 0.0091 -40.3928  -5.7511
    C(city_state)[T.228]    -31.2745   9.1338 -3.4241 0.0006 -49.1854 -13.3636
    C(city_state)[T.229]    -29.7911   9.6089 -3.1004 0.0020 -48.6336 -10.9485
    C(city_state)[T.230]    -10.6707  10.3374 -1.0322 0.3021 -30.9419   9.6006
    C(city_state)[T.231]    -61.7796  11.4602 -5.3908 0.0000 -84.2525 -39.3068
    C(city_state)[T.232]     -1.8542   9.2677 -0.2001 0.8414 -20.0276  16.3193
    C(city_state)[T.233]      1.5309   8.9174  0.1717 0.8637 -15.9557  19.0176
    C(city_state)[T.234]    -32.2811  11.0337 -2.9257 0.0035 -53.9176 -10.6446
    C(city_state)[T.235]    -12.9869   9.5448 -1.3606 0.1738 -31.7038   5.7301
    C(city_state)[T.236]     -8.7913   7.9328 -1.1082 0.2679 -24.3470   6.7645
    C(city_state)[T.237]    -14.7263   8.5791 -1.7165 0.0862 -31.5494   2.0968
    C(city_state)[T.238]      2.5199   8.1137  0.3106 0.7562 -13.3908  18.4306
    C(city_state)[T.239]    -23.0268  10.2281 -2.2513 0.0245 -43.0836  -2.9700
    C(city_state)[T.240]      0.8937  10.6845  0.0836 0.9333 -20.0581  21.8455
    C(city_state)[T.241]      4.9451   9.8045  0.5044 0.6140 -14.2812  24.1714
    C(city_state)[T.242]     24.6728  12.9405  1.9066 0.0567  -0.7029  50.0485
    C(city_state)[T.243]      3.2434   8.7818  0.3693 0.7119 -13.9773  20.4640
    C(city_state)[T.244]     -5.6371  10.3588 -0.5442 0.5864 -25.9503  14.6761
    C(city_state)[T.245]      6.8460  12.3945  0.5523 0.5808 -17.4591  31.1512
    C(city_state)[T.246]      8.0475   9.3931  0.8567 0.3917 -10.3719  26.4669
    C(city_state)[T.247]    -27.6168   8.0830 -3.4167 0.0006 -43.4672 -11.7665
    C(city_state)[T.248]     10.2531  10.0076  1.0245 0.3057  -9.3714  29.8776
    C(city_state)[T.249]      4.5726   9.3046  0.4914 0.6232 -13.6732  22.8185
    C(city_state)[T.250]      7.0621  11.6300  0.6072 0.5438 -15.7438  29.8680
    C(city_state)[T.251]    -16.2546   9.3043 -1.7470 0.0808 -34.4999   1.9907
    C(city_state)[T.252]     -5.8911   7.4773 -0.7879 0.4309 -20.5537   8.7715
    C(city_state)[T.253]      1.0556   8.1507  0.1295 0.8970 -14.9275  17.0386
    C(city_state)[T.254]    -16.9284   9.2524 -1.8296 0.0674 -35.0719   1.2152
    C(city_state)[T.255]    -56.6759   9.8362 -5.7619 0.0000 -75.9643 -37.3874
    C(city_state)[T.256]    -14.3922   9.0744 -1.5860 0.1129 -32.1867   3.4023
    C(city_state)[T.257]     -4.9882   7.8497 -0.6355 0.5252 -20.3812  10.4047
    C(city_state)[T.258]     -9.7705   9.8400 -0.9929 0.3208 -29.0663   9.5254
    C(city_state)[T.259]     14.7282  10.9636  1.3434 0.1793  -6.7708  36.2273
    C(city_state)[T.260]    -11.0897   9.4536 -1.1731 0.2409 -29.6277   7.4484
    C(city_state)[T.261]    -22.5033  10.2797 -2.1891 0.0287 -42.6612  -2.3453
    C(city_state)[T.262]      2.3681   8.1795  0.2895 0.7722 -13.6715  18.4078
    C(city_state)[T.263]     -4.2540  10.4781 -0.4060 0.6848 -24.8011  16.2931
    C(city_state)[T.264]     -0.2751   9.9554 -0.0276 0.9780 -19.7971  19.2469
    C(city_state)[T.265]    -11.7356   9.9451 -1.1800 0.2381 -31.2374   7.7663
    C(city_state)[T.266]     26.4936   9.5501  2.7742 0.0056   7.7662  45.2210
    C(city_state)[T.267]     -4.1703   9.6415 -0.4325 0.6654 -23.0768  14.7362
    C(city_state)[T.268]     -5.5028   9.1868 -0.5990 0.5492 -23.5176  12.5121
    C(city_state)[T.269]    -11.2750   8.8642 -1.2720 0.2035 -28.6572   6.1072
    C(city_state)[T.270]      0.0183   8.5572  0.0021 0.9983 -16.7621  16.7987
    C(city_state)[T.271]     -3.4858  11.1273 -0.3133 0.7541 -25.3059  18.3342
    C(city_state)[T.272]    -16.2329   9.4986 -1.7090 0.0876 -34.8592   2.3933
    C(city_state)[T.273]    -24.5394   8.1212 -3.0216 0.0025 -40.4647  -8.6141
    C(city_state)[T.274]      1.8984   9.4211  0.2015 0.8403 -16.5759  20.3727
    C(city_state)[T.275]     -3.4770  12.1514 -0.2861 0.7748 -27.3054  20.3515
    C(city_state)[T.276]     -6.5107   7.7608 -0.8389 0.4016 -21.7293   8.7078
    C(city_state)[T.277]    -14.0532  10.6707 -1.3170 0.1880 -34.9779   6.8715
    C(city_state)[T.278]    -15.2587  17.3808 -0.8779 0.3801 -49.3417  18.8242
    C(city_state)[T.279]    -19.0995   9.2740 -2.0595 0.0396 -37.2854  -0.9136
    C(city_state)[T.280]      3.9090   9.6757  0.4040 0.6862 -15.0646  22.8826
    C(city_state)[T.281]    -28.2473  10.7682 -2.6232 0.0088 -49.3633  -7.1313
    C(city_state)[T.282]      3.1860  10.0653  0.3165 0.7516 -16.5516  22.9236
    C(city_state)[T.283]     -4.0427   8.9174 -0.4533 0.6503 -21.5292  13.4438
    C(city_state)[T.284]    -23.4797  10.4234 -2.2526 0.0244 -43.9195  -3.0399
    C(city_state)[T.285]     -0.1751   9.9554 -0.0176 0.9860 -19.6971  19.3469
    C(city_state)[T.286]    -15.5416   8.5250 -1.8231 0.0684 -32.2587   1.1756
    C(city_state)[T.287]     10.4200  11.8016  0.8829 0.3774 -12.7224  33.5625
    C(city_state)[T.288]      5.3878  10.0847  0.5343 0.5932 -14.3879  25.1634
    C(city_state)[T.289]      7.6593   9.6404  0.7945 0.4270 -11.2451  26.5638
    C(city_state)[T.290]    -16.9411   8.9901 -1.8844 0.0596 -34.5702   0.6881
    C(city_state)[T.291]      1.5724   9.4336  0.1667 0.8676 -16.9263  20.0712
    C(city_state)[T.292]      0.0942   9.3360  0.0101 0.9920 -18.2133  18.4016
    C(city_state)[T.293]      2.1865   9.4108  0.2323 0.8163 -16.2676  20.6406
    C(city_state)[T.294]      2.3535   9.3482  0.2518 0.8012 -15.9778  20.6848
    C(city_state)[T.295]    -14.7984   9.8488 -1.5026 0.1331 -34.1115   4.5146
    C(city_state)[T.296]    -23.8949   9.3182 -2.5643 0.0104 -42.1674  -5.6223
    C(city_state)[T.297]      8.5148  10.3732  0.8209 0.4118 -11.8265  28.8561
    C(city_state)[T.298]      7.7853   8.9955  0.8655 0.3869  -9.8544  25.4251
    C(city_state)[T.299]    -23.3171  11.4053 -2.0444 0.0410 -45.6824  -0.9517
    C(city_state)[T.300]    -21.9311   8.5866 -2.5541 0.0107 -38.7690  -5.0932
    C(city_state)[T.301]     -5.4347   9.9088 -0.5485 0.5834 -24.8653  13.9959
    C(city_state)[T.302]    -35.0330  11.2052 -3.1265 0.0018 -57.0058 -13.0601
    C(city_state)[T.303]    -12.9030   9.3671 -1.3775 0.1685 -31.2714   5.4655
    C(city_state)[T.304]    -15.6763   9.3575 -1.6753 0.0940 -34.0259   2.6734
    C(city_state)[T.305]    -21.2194  10.3402 -2.0521 0.0403 -41.4959  -0.9428
    C(city_state)[T.306]    -12.2442  12.7050 -0.9637 0.3353 -37.1580  12.6697
    C(city_state)[T.307]      6.5768   9.8934  0.6648 0.5063 -12.8237  25.9774
    C(city_state)[T.308]    -26.4069   9.0455 -2.9194 0.0035 -44.1446  -8.6692
    C(city_state)[T.309]     -5.8978   9.7577 -0.6044 0.5456 -25.0322  13.2366
    C(city_state)[T.310]     -5.9572  10.4064 -0.5725 0.5671 -26.3636  14.4492
    C(city_state)[T.311]     -4.1039  10.7381 -0.3822 0.7024 -25.1608  16.9531
    C(city_state)[T.312]     -5.0467   8.9143 -0.5661 0.5714 -22.5272  12.4338
    C(city_state)[T.313]    -13.1096   9.7225 -1.3484 0.1777 -32.1750   5.9557
    C(city_state)[T.314]    -19.6710  11.9581 -1.6450 0.1001 -43.1204   3.7783
    C(city_state)[T.315]    -10.4667   9.4675 -1.1055 0.2690 -29.0320   8.0986
    C(city_state)[T.316]     -6.6021   8.5693 -0.7704 0.4411 -23.4061  10.2018
    C(city_state)[T.317]     13.5123  10.0345  1.3466 0.1782  -6.1648  33.1894
    C(city_state)[T.318]     -7.4183   9.2768 -0.7997 0.4240 -25.6096  10.7731
    C(city_state)[T.319]      6.4007  11.0699  0.5782 0.5632 -15.3069  28.1083
    C(city_state)[T.320]    -23.7306  12.5166 -1.8959 0.0581 -48.2750   0.8137
    C(city_state)[T.321]      3.7461   9.2678  0.4042 0.6861 -14.4277  21.9198
    C(city_state)[T.322]    -19.6635  10.4635 -1.8792 0.0603 -40.1820   0.8550
    C(city_state)[T.323]     -2.1657   8.1867 -0.2645 0.7914 -18.2194  13.8880
    C(city_state)[T.324]     14.3640   9.9935  1.4373 0.1508  -5.2327  33.9607
    C(city_state)[T.325]     -2.5905  10.4274 -0.2484 0.8038 -23.0382  17.8572
    C(city_state)[T.326]     -5.0878   8.4433 -0.6026 0.5468 -21.6448  11.4691
    C(city_state)[T.327]      9.1176   9.2481  0.9859 0.3243  -9.0175  27.2527
    C(city_state)[T.328]      6.1900   9.0888  0.6811 0.4959 -11.6327  24.0128
    C(city_state)[T.329]     -8.5520   8.4819 -1.0083 0.3134 -25.1845   8.0805
    C(city_state)[T.330]      2.7170  10.1561  0.2675 0.7891 -17.1987  22.6327
    C(city_state)[T.331]     12.9597   8.6270  1.5022 0.1332  -3.9573  29.8768
    C(city_state)[T.332]     -1.2839   9.3539 -0.1373 0.8908 -19.6265  17.0587
    C(city_state)[T.333]    -15.2291   9.3704 -1.6252 0.1042 -33.6040   3.1458
    C(city_state)[T.334]    -10.0043  11.6738 -0.8570 0.3915 -32.8961  12.8875
    C(city_state)[T.335]     -6.1139   8.9402 -0.6839 0.4941 -23.6453  11.4175
    C(city_state)[T.336]    -26.8997   8.8096 -3.0535 0.0023 -44.1749  -9.6246
    C(city_state)[T.337]      0.6496  12.6599  0.0513 0.9591 -24.1758  25.4750
    C(city_state)[T.338]    -19.6336   8.8843 -2.2099 0.0272 -37.0553  -2.2119
    C(city_state)[T.339]     -3.9792   9.5674 -0.4159 0.6775 -22.7404  14.7820
    C(city_state)[T.340]     -3.5509   9.3328 -0.3805 0.7036 -21.8522  14.7504
    C(city_state)[T.341]      8.0552   9.1907  0.8765 0.3809  -9.9673  26.0777
    C(city_state)[T.342]     -9.3974   8.9574 -1.0491 0.2942 -26.9624   8.1675
    C(city_state)[T.343]     -7.2987   9.7170 -0.7511 0.4526 -26.3533  11.7559
    C(city_state)[T.344]     -8.6035  10.2700 -0.8377 0.4023 -28.7425  11.5356
    C(city_state)[T.345]    -22.8860   9.2997 -2.4609 0.0139 -41.1224  -4.6497
    C(city_state)[T.346]    -13.6461  11.0103 -1.2394 0.2153 -35.2367   7.9445
    C(city_state)[T.347]    -11.4781  10.3822 -1.1056 0.2690 -31.8371   8.8808
    C(city_state)[T.348]    -11.3094   6.7179 -1.6835 0.0924 -24.4829   1.8640
    C(city_state)[T.349]    -31.7077  10.9593 -2.8932 0.0038 -53.1984 -10.2171
    C(city_state)[T.350]     -8.5294  10.5380 -0.8094 0.4184 -29.1939  12.1350
    C(city_state)[T.351]    -16.2092  10.1868 -1.5912 0.1117 -36.1851   3.7667
    C(city_state)[T.352]    -31.2765   9.8982 -3.1598 0.0016 -50.6864 -11.8666
    C(city_state)[T.353]     -6.7695   8.9187 -0.7590 0.4479 -24.2586  10.7195
    C(city_state)[T.354]     -6.6223   9.7676 -0.6780 0.4978 -25.7761  12.5314
    C(city_state)[T.355]    -75.3826   8.5523 -8.8143 0.0000 -92.1533 -58.6118
    C(city_state)[T.356]     -8.1747   9.8665 -0.8285 0.4074 -27.5224  11.1730
    C(city_state)[T.357]    -17.1501  10.3565 -1.6560 0.0979 -37.4586   3.1585
    C(city_state)[T.358]      0.0363   9.3746  0.0039 0.9969 -18.3469  18.4195
    C(city_state)[T.359]      2.5434   9.6084  0.2647 0.7913 -16.2983  21.3851
    C(city_state)[T.360]    -17.5421  15.4224 -1.1374 0.2555 -47.7847  12.7004
    C(city_state)[T.361]     27.8923  11.6335  2.3976 0.0166   5.0796  50.7050
    C(city_state)[T.362]    -11.0200  10.3298 -1.0668 0.2862 -31.2762   9.2362
    C(city_state)[T.363]     -2.0134   9.4012 -0.2142 0.8304 -20.4487  16.4219
    C(city_state)[T.364]     -2.9484   9.5095 -0.3100 0.7566 -21.5960  15.6992
    C(city_state)[T.365]    -31.5265  11.9904 -2.6293 0.0086 -55.0390  -8.0140
    C(vorelationship)[T.1]  -14.0857   3.8513 -3.6574 0.0003 -21.6380  -6.5334
    C(vorelationship)[T.2]   27.1871   2.9283  9.2842 0.0000  21.4447  32.9294
    C(vorelationship)[T.3]   28.0395   7.3817  3.7985 0.0001  13.5642  42.5147
    C(vorelationship)[T.4]   -9.8987   3.9422 -2.5110 0.0121 -17.6291  -2.1683
    C(vorelationship)[T.5]    9.6742   3.9416  2.4544 0.0142   1.9449  17.4036
    C(vorelationship)[T.6]   -1.7288   3.0317 -0.5702 0.5686  -7.6738   4.2163
    C(vorelationship)[T.7]    0.1526   3.5500  0.0430 0.9657  -6.8088   7.1139
    C(vorelationship)[T.8]    6.9521   3.2373  2.1475 0.0319   0.6040  13.3003
    C(vorelationship)[T.9]    0.3183   3.9040  0.0815 0.9350  -7.3372   7.9738
    C(vorelationship)[T.10]   2.8725   4.1820  0.6869 0.4922  -5.3282  11.0733
    C(vorelationship)[T.11]   3.4217   4.7623  0.7185 0.4725  -5.9168  12.7603
    C(vorelationship)[T.12]   1.0425   3.9483  0.2640 0.7918  -6.7000   8.7851
    C(vorelationship)[T.13]  40.8385   9.0173  4.5289 0.0000  23.1559  58.5210
    C(vorelationship)[T.14] -42.1728   5.0879 -8.2889 0.0000 -52.1498 -32.1957
    C(vorelationship)[T.15] -12.2348   3.5193 -3.4765 0.0005 -19.1359  -5.3337
    C(vorelationship)[T.16]   0.7414   2.7574  0.2689 0.7880  -4.6656   6.1485
    C(vorelationship)[T.17]  -4.0388   3.1475 -1.2832 0.1996 -10.2110   2.1333
    C(vorelationship)[T.18]  25.9453   3.4721  7.4725 0.0000  19.1366  32.7540
    C(vorelationship)[T.19]  29.0182  14.3636  2.0203 0.0435   0.8518  57.1846
    C(vorelationship)[T.20]  -4.0277   3.7785 -1.0659 0.2866 -11.4372   3.3818
    C(vorelationship)[T.21] -10.0636   4.7323 -2.1266 0.0336 -19.3434  -0.7839
    C(vorelationship)[T.22] -19.7829   3.0202 -6.5502 0.0000 -25.7054 -13.8605
    C(vorelationship)[T.23]   1.3644   2.5802  0.5288 0.5970  -3.6953   6.4240
    C(vorelationship)[T.24]   8.4931   2.6398  3.2174 0.0013   3.3166  13.6696
    C(vorelationship)[T.25]   2.7692   8.0911  0.3423 0.7322 -13.0971  18.6356
    C(vorelationship)[T.26]  10.0724   3.1654  3.1820 0.0015   3.8651  16.2797
    C(vorelationship)[T.27]   0.7897   3.2712  0.2414 0.8093  -5.6250   7.2043
    C(vorelationship)[T.28]  -4.1948   2.8500 -1.4719 0.1412  -9.7836   1.3939
    C(season_month)[T.1]      2.6739   3.5540  0.7524 0.4519  -4.2953   9.6431
    C(season_month)[T.2]      2.9064   3.7347  0.7782 0.4365  -4.4172  10.2300
    C(season_month)[T.3]     10.1362   3.2383  3.1301 0.0018   3.7861  16.4863
    C(season_month)[T.4]      8.4093   3.3645  2.4994 0.0125   1.8116  15.0070
    C(season_month)[T.5]     15.1053   5.0460  2.9935 0.0028   5.2104  25.0003
    C(season_month)[T.6]      1.3316   3.1981  0.4164 0.6772  -4.9397   7.6029
    C(season_month)[T.7]      2.9921   3.9472  0.7580 0.4485  -4.7481  10.7323
    C(season_month)[T.8]     22.9525   4.2650  5.3815 0.0000  14.5890  31.3161
    C(season_month)[T.9]     14.2346   3.4444  4.1327 0.0000   7.4803  20.9890
    C(season_month)[T.10]     5.6744   4.3205  1.3134 0.1892  -2.7978  14.1466
    C(season_month)[T.11]     3.6689   3.2568  1.1265 0.2601  -2.7175  10.0554
    C(v_race_sex)[T.1]       -7.7649   5.6967 -1.3631 0.1730 -18.9359   3.4061
    C(v_race_sex)[T.2]        8.2076   8.4848  0.9673 0.3335  -8.4307  24.8459
    C(v_race_sex)[T.3]        2.2031   8.5692  0.2571 0.7971 -14.6007  19.0069
    C(v_race_sex)[T.4]        9.0313   8.3743  1.0785 0.2809  -7.3903  25.4530
    C(v_race_sex)[T.5]        3.3136   8.3700  0.3959 0.6922 -13.0996  19.7268
    C(v_race_sex)[T.6]        8.4532   8.3586  1.0113 0.3120  -7.9377  24.8441
    C(v_race_sex)[T.7]        2.0951   8.3139  0.2520 0.8011 -14.2080  18.3982
    C(v_race_sex)[T.8]       18.0403   9.5411  1.8908 0.0588  -0.6694  36.7500
    C(v_race_sex)[T.9]       12.1899   9.7192  1.2542 0.2099  -6.8690  31.2487
    C(v_race_sex)[T.10]       5.4922   8.2787  0.6634 0.5071 -10.7420  21.7264
    C(v_race_sex)[T.11]      -0.3785   8.2885 -0.0457 0.9636 -16.6318  15.8749
    C(situation_type)[T.1]   -5.7850   7.2084 -0.8025 0.4223 -19.9204   8.3504
    C(situation_type)[T.2]   11.7803  10.2678  1.1473 0.2514  -8.3545  31.9151
    C(situation_type)[T.3]   -2.3968   8.8210 -0.2717 0.7859 -19.6943  14.9008
    C(situation_type)[T.4]    7.4703   7.9873  0.9353 0.3497  -8.1925  23.1331
    C(situation_type)[T.5]    7.3626   9.8194  0.7498 0.4534 -11.8928  26.6181
    C(situation_type)[T.6]   14.7732  10.4560  1.4129 0.1578  -5.7305  35.2769
    C(situation_type)[T.7]  -12.8986   7.3604 -1.7524 0.0798 -27.3320   1.5347
    C(situation_type)[T.8]   -6.7731   6.6988 -1.0111 0.3121 -19.9091   6.3628
    C(situation_type)[T.9]  -15.7496   7.4099 -2.1255 0.0336 -30.2801  -1.2190
    C(situation_type)[T.10]   5.0370  17.5376  0.2872 0.7740 -29.3534  39.4274
    C(situation_type)[T.11]   9.4254  10.1801  0.9259 0.3546 -10.5372  29.3880
    C(situation_type)[T.12]  -6.9287   8.4075 -0.8241 0.4100 -23.4153   9.5580
    C(situation_type)[T.13] -18.5310   7.4055 -2.5023 0.0124 -33.0528  -4.0093
    C(type)[T.1]             -7.5934   8.3494 -0.9095 0.3632 -23.9661   8.7793
    C(type)[T.2]              5.3648   3.3659  1.5939 0.1111  -1.2356  11.9651
    C(type)[T.3]             -2.2805   4.3929 -0.5191 0.6037 -10.8948   6.3339
    C(type)[T.4]             13.8924   5.4949  2.5282 0.0115   3.1171  24.6678
    C(type)[T.5]             20.7507  10.4117  1.9930 0.0464   0.3337  41.1676
    C(type)[T.6]             33.3574  10.5021  3.1762 0.0015  12.7632  53.9516
    C(o_age_range)[T.1]      -3.9533   1.2838 -3.0794 0.0021  -6.4708  -1.4359
    C(o_age_range)[T.2]      -6.0717   3.1603 -1.9212 0.0548 -12.2689   0.1255
    C(o_age_range)[T.3]       1.5058   1.5799  0.9531 0.3406  -1.5923   4.6039
    --------------------------------------------------------------------------
    Omnibus:                 154.610         Durbin-Watson:            1.577  
    Prob(Omnibus):           0.000           Jarque-Bera (JB):         330.434
    Skew:                    -0.360          Prob(JB):                 0.000  
    Kurtosis:                4.509           Condition No.:            710    
    ==========================================================================
    



```python
# Return p-values and coefficients values from linear regression model
df_reg_output = pd.DataFrame({'coef': model.params, 'pval': model.pvalues, 'r_squared': model.rsquared}).reset_index()
df_reg_output['index'] = df_reg_output['index'].str.replace('T.',':').str.replace(']','').str.replace(')','').str.replace('[','')
df_reg_output[['variable','enc_value']] = df_reg_output['index'].str.split(':',expand=True)
df_reg_output = df_reg_output.drop(columns=['index'])
df_reg_output[['delete','variable1']] = df_reg_output['variable'].str.split('(',expand=True)
df_reg_output = df_reg_output.drop(columns=['variable','delete'])
df_reg_output
```

    /var/folders/f0/cswv940s1pl9lnmr89n7jvkm0000gn/T/ipykernel_68724/582375765.py:3: FutureWarning: The default value of regex will change from True to False in a future version.
      df_reg_output['index'] = df_reg_output['index'].str.replace('T.',':').str.replace(']','').str.replace(')','').str.replace('[','')
    /var/folders/f0/cswv940s1pl9lnmr89n7jvkm0000gn/T/ipykernel_68724/582375765.py:3: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.
      df_reg_output['index'] = df_reg_output['index'].str.replace('T.',':').str.replace(']','').str.replace(')','').str.replace('[','')





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
      <th>coef</th>
      <th>pval</th>
      <th>r_squared</th>
      <th>enc_value</th>
      <th>variable1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47.076746</td>
      <td>5.432125e-04</td>
      <td>0.679517</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.024337</td>
      <td>9.157331e-01</td>
      <td>0.679517</td>
      <td>1</td>
      <td>city_state</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-56.291722</td>
      <td>5.933975e-08</td>
      <td>0.679517</td>
      <td>2</td>
      <td>city_state</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-10.315072</td>
      <td>2.220851e-01</td>
      <td>0.679517</td>
      <td>3</td>
      <td>city_state</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-20.527702</td>
      <td>5.570651e-02</td>
      <td>0.679517</td>
      <td>4</td>
      <td>city_state</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>433</th>
      <td>20.750667</td>
      <td>4.637371e-02</td>
      <td>0.679517</td>
      <td>5</td>
      <td>type</td>
    </tr>
    <tr>
      <th>434</th>
      <td>33.357430</td>
      <td>1.510875e-03</td>
      <td>0.679517</td>
      <td>6</td>
      <td>type</td>
    </tr>
    <tr>
      <th>435</th>
      <td>-3.953346</td>
      <td>2.097574e-03</td>
      <td>0.679517</td>
      <td>1</td>
      <td>o_age_range</td>
    </tr>
    <tr>
      <th>436</th>
      <td>-6.071714</td>
      <td>5.481890e-02</td>
      <td>0.679517</td>
      <td>2</td>
      <td>o_age_range</td>
    </tr>
    <tr>
      <th>437</th>
      <td>1.505787</td>
      <td>3.406385e-01</td>
      <td>0.679517</td>
      <td>3</td>
      <td>o_age_range</td>
    </tr>
  </tbody>
</table>
<p>438 rows × 5 columns</p>
</div>




```python
# Export data model for import in to Tableau for visualization

with open('data/yrs_lost_pvalue_encoded_GS.csv', 'w') as file:
     df_reg_output.to_csv(file, index=False)

with open('data/yrs_lost_model_GS.csv', 'w') as file:
     df_reg.to_csv(file, index=False)
```
