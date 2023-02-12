# Team_25_DS4A_MassKilling_Data_Model

# Tool References


```python
# pandasql base code structure | Uncomment code (#) to see full structure

# Query definition
#query = """ SELECT st.Students, st.Gender, st.Email, st.Age, tat.Department
            #FROM students_df st INNER JOIN teaching_assistant_df tat 
            #ON st.Email = tat.Email
        # """
# Query execution

#name_email = sqldf(query)
#name_email

# Source: https://towardsdatascience.com/how-to-run-sql-queries-on-your-pandas-dataframes-with-python-4237ffecc43b
```


```python
# Pandas Profiling

# General Code Structure
# profile = ProfileReport(df, title="XYZ Report")
# profile.to_notebook_iframe()
```


```python
# Distributions Example from Sy

# Link: https://colab.research.google.com/drive/1G2Wl2TFGjCTV8IZwvFDJIOC_kr1teNDe?usp=sharing
```

# Import Data for Local Jupyter Notebook


```python
#Install Requirements via Text File
!pip install -r requirements.txt
!pip install nbconvert
!pip install pyppeteer --allow-chromium-download


#Import the packages for cleaning, analysis, and visualization 

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import ydata_profiling as pp
from ydata_profiling import ProfileReport
from pandasql import sqldf
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
    Requirement already satisfied: packaging>=20.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (21.3)
    Requirement already satisfied: pillow>=6.2.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (9.2.0)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (4.25.0)
    Requirement already satisfied: cycler>=0.10 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (1.4.2)
    Requirement already satisfied: pyparsing>=2.2.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from matplotlib->-r requirements.txt (line 4)) (3.0.9)
    Requirement already satisfied: pandas-flavor>=0.2.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.5.0)
    Requirement already satisfied: tabulate in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.8.10)
    Requirement already satisfied: outdated in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.2.2)
    Requirement already satisfied: statsmodels>=0.13 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (0.13.2)
    Requirement already satisfied: scikit-learn in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pingouin->-r requirements.txt (line 6)) (1.0.2)
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
    Requirement already satisfied: networkx>=2.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.8.4)
    Requirement already satisfied: attrs>=19.3.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (21.4.0)
    Requirement already satisfied: tangled-up-in-unicode>=0.0.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (0.2.0)
    Requirement already satisfied: imagehash in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (4.3.1)
    Requirement already satisfied: sqlparse in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (0.4.3)
    Requirement already satisfied: six in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (1.16.0)
    Requirement already satisfied: prettytable<1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (0.7.2)
    Requirement already satisfied: ipython-genutils>=0.1.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (0.2.0)
    Requirement already satisfied: ipython>=1.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython-sql->-r requirements.txt (line 10)) (7.31.1)
    Requirement already satisfied: decorator in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (5.1.1)
    Requirement already satisfied: pygments in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (2.11.2)
    Requirement already satisfied: matplotlib-inline in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.1.6)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (3.0.20)
    Requirement already satisfied: jedi>=0.16 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.18.1)
    Requirement already satisfied: appnope in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.1.2)
    Requirement already satisfied: pickleshare in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.7.5)
    Requirement already satisfied: backcall in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.2.0)
    Requirement already satisfied: traitlets>=4.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (5.1.1)
    Requirement already satisfied: pexpect>4.3 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (4.8.0)
    Requirement already satisfied: setuptools>=18.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (63.4.1)
    Requirement already satisfied: MarkupSafe>=0.23 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jinja2<3.2,>=2.11.1->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.0.1)
    Requirement already satisfied: lazy-loader>=0.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pandas-flavor>=0.2.0->pingouin->-r requirements.txt (line 6)) (0.1)
    Requirement already satisfied: xarray in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pandas-flavor>=0.2.0->pingouin->-r requirements.txt (line 6)) (0.20.1)
    Requirement already satisfied: joblib>=0.14.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from phik<0.13,>=0.11.1->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.1.0)
    Requirement already satisfied: typing-extensions>=4.2.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pydantic<1.11,>=1.8.1->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (4.3.0)
    Requirement already satisfied: idna<4,>=2.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (3.3)
    Requirement already satisfied: charset-normalizer<3,>=2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from requests<2.29,>=2.24.0->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (2022.9.24)
    Requirement already satisfied: patsy>=0.5.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from statsmodels>=0.13->pingouin->-r requirements.txt (line 6)) (0.5.2)
    Requirement already satisfied: littleutils in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from outdated->pingouin->-r requirements.txt (line 6)) (0.2.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from scikit-learn->pingouin->-r requirements.txt (line 6)) (2.2.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jedi>=0.16->ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from pexpect>4.3->ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.7.0)
    Requirement already satisfied: wcwidth in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=1.0->ipython-sql->-r requirements.txt (line 10)) (0.2.5)
    Requirement already satisfied: PyWavelets in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from imagehash->visions[type_image_path]==0.7.5->ydata-profiling==0.0.dev0->-r requirements.txt (line 8)) (1.3.0)
    Requirement already satisfied: nbconvert in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (6.4.4)
    Requirement already satisfied: beautifulsoup4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (4.11.1)
    Requirement already satisfied: jupyter-core in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (4.11.1)
    Requirement already satisfied: testpath in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (0.6.0)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (0.5.13)
    Requirement already satisfied: nbformat>=4.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (5.5.0)
    Requirement already satisfied: entrypoints>=0.2.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (0.4)
    Requirement already satisfied: jupyterlab-pygments in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (0.1.2)
    Requirement already satisfied: bleach in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (4.1.0)
    Requirement already satisfied: mistune<2,>=0.8.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (0.8.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (1.5.0)
    Requirement already satisfied: jinja2>=2.4 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (2.11.3)
    Requirement already satisfied: defusedxml in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (0.7.1)
    Requirement already satisfied: traitlets>=5.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (5.1.1)
    Requirement already satisfied: pygments>=2.4.1 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbconvert) (2.11.2)
    Requirement already satisfied: MarkupSafe>=0.23 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jinja2>=2.4->nbconvert) (2.0.1)
    Requirement already satisfied: nest-asyncio in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (1.5.5)
    Requirement already satisfied: jupyter-client>=6.1.5 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert) (7.3.4)
    Requirement already satisfied: jsonschema>=2.6 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbformat>=4.4->nbconvert) (4.16.0)
    Requirement already satisfied: fastjsonschema in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from nbformat>=4.4->nbconvert) (2.16.2)
    Requirement already satisfied: soupsieve>1.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from beautifulsoup4->nbconvert) (2.3.1)
    Requirement already satisfied: webencodings in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from bleach->nbconvert) (0.5.1)
    Requirement already satisfied: packaging in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from bleach->nbconvert) (21.3)
    Requirement already satisfied: six>=1.9.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from bleach->nbconvert) (1.16.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (0.18.0)
    Requirement already satisfied: attrs>=17.4.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (21.4.0)
    Requirement already satisfied: tornado>=6.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (6.1)
    Requirement already satisfied: pyzmq>=23.0 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (23.2.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert) (2.8.2)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /Users/kashundavis/opt/anaconda3/lib/python3.9/site-packages (from packaging->bleach->nbconvert) (3.0.9)
    
    Usage:   
      pip install [options] <requirement specifier> [package-index-options] ...
      pip install [options] -r <requirements file> [package-index-options] ...
      pip install [options] [-e] <vcs project url> ...
      pip install [options] [-e] <local project path> ...
      pip install [options] <archive url/path> ...
    
    no such option: --allow-chromium-download



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

# Data Cleaning & Transformation


```python
# Incident Table Only | No Filters
# Query definition
mki_q = """ 
            SELECT *
            FROM mki   
        """
# Query execution and convert to dataframe
mki = sqldf(mki_q).copy()

#Cleaning Dataframe
mki['incidentdate'] = mki['date']
#mki = mki.drop(columns=['date'])
#mki_mkv['num_victims_injured'] = mki_mkv['num_victims_injured'].astype(int)
mki['incident_id'] = mki['incident_id'].astype(str)
#mki_mkv['victim_id'] = mki_mkv['victim_id'].astype(str)
#mki_mkv['victim_age'] = mki_mkv['victim_age'].astype(int)
mki.sample(3)
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
      <th>date</th>
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
      <th>longitude</th>
      <th>latitude</th>
      <th>narrative</th>
      <th>incidentdate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>496</td>
      <td>2021-12-27</td>
      <td>Denver</td>
      <td>CO</td>
      <td>1</td>
      <td>5</td>
      <td>2.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Public</td>
      <td>Interpersonal conflict</td>
      <td>Multiple</td>
      <td>Multiple</td>
      <td>-104.992361</td>
      <td>39.739317</td>
      <td>Lyndon McLeod, 47, went on a shooting spree th...</td>
      <td>2021-12-27</td>
    </tr>
    <tr>
      <th>329</th>
      <td>123</td>
      <td>2012-10-17</td>
      <td>Denver</td>
      <td>CO</td>
      <td>3</td>
      <td>5</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>None</td>
      <td>Felony</td>
      <td>Robbery</td>
      <td>Commercial/Retail/Entertainment</td>
      <td>Bar/Club/Restaurant</td>
      <td>-104.984703</td>
      <td>39.739154</td>
      <td>Firefighters responding to a blaze at Fero's B...</td>
      <td>2012-10-17</td>
    </tr>
    <tr>
      <th>435</th>
      <td>158</td>
      <td>2008-11-02</td>
      <td>Long Beach</td>
      <td>CA</td>
      <td>2</td>
      <td>5</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Felony</td>
      <td>Drug trade</td>
      <td>Residence/Other shelter</td>
      <td>Shelter/Drug house</td>
      <td>-118.189235</td>
      <td>33.766962</td>
      <td>Five men in a homeless camp were killed in a d...</td>
      <td>2008-11-02</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging of Incidents and Victim Tables
# Query definition
mki_mkv_q = """ 
            SELECT mki.incident_id, incidentdate, city, state, num_offenders, num_victims_killed,
              num_victims_injured, firstcod, secondcod, type,
             situation_type, location_type, location, longitude, latitude,
             narrative, victim_id, cast (age as int) as v_age, race as v_race, sex as v_sex, vorelationship
            FROM mki
              LEFT JOIN mkv
              ON mki.incident_id = mkv.incident_id
       
        """
# Query execution and convert to dataframe
mki_mkv = sqldf(mki_mkv_q).copy()

#Cleaning Dataframe
#mki_mkv['incidentdate'] = pd.to_datetime(mki_mkv['date'])
#mki_mkv = mki_mkv.drop(columns=['date'])
#mki_mkv['num_victims_injured'] = mki_mkv['num_victims_injured'].astype(int)
mki_mkv['incident_id'] = mki_mkv['incident_id'].astype(str)
mki_mkv['victim_id'] = mki_mkv['victim_id'].astype(str)
mki_mkv['incidentdate'] = pd.to_datetime(mki_mkv['incidentdate'],format="%Y-%m-%d")
mki_mkv['month'] = mki_mkv['incidentdate'].dt.month.astype('category')
mki_mkv['year'] = mki_mkv['incidentdate'].dt.year
mki_mkv = mki_mkv.drop_duplicates().reset_index()
mki_mkv = mki_mkv.drop(columns="index")
mki_mkv.head(3)
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
      <th>longitude</th>
      <th>latitude</th>
      <th>narrative</th>
      <th>victim_id</th>
      <th>v_age</th>
      <th>v_race</th>
      <th>v_sex</th>
      <th>vorelationship</th>
      <th>month</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>-119.4214</td>
      <td>36.34629</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2848</td>
      <td>72.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>-119.4214</td>
      <td>36.34629</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2849</td>
      <td>52.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>None</td>
      <td>1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>-119.4214</td>
      <td>36.34629</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2850</td>
      <td>19.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>None</td>
      <td>1</td>
      <td>2023</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>




```python
# Merging of Incidents and Weapon Tables
# Query definition
mki_mkw_q = """ 
            SELECT 
              mki.incident_id
              ,incidentdate
              ,city
              ,state
              ,num_offenders
              ,num_victims_killed
              ,num_victims_injured
              ,firstcod
              ,secondcod
              ,type
              ,situation_type
              ,location_type
              ,location
              ,longitude
              ,latitude
              ,narrative
              ,weapon_id
              ,weapon_type
              ,gun_class
              ,gun_type
            FROM mki
              LEFT JOIN mkw
              ON mki.incident_id = mkw.incident_id
         
        """
# Query execution
mki_mkw = sqldf(mki_mkw_q).copy()
#mki_mkw['incidentdate'] = pd.to_datetime(mki_mkw['date'])
#mki_mkw = mki_mkw.drop(columns=['date'])
#mki_mkw['num_victims_injured'] = mki_mkw['num_victims_injured'].astype(int)
mki_mkw['incident_id'] = mki_mkw['incident_id'].astype(str)
mki_mkw['weapon_id'] = mki_mkw['weapon_id'].astype(str)
mki_mkw = mki_mkw.drop_duplicates()

mki_mkw.head(5)
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
      <th>longitude</th>
      <th>latitude</th>
      <th>narrative</th>
      <th>weapon_id</th>
      <th>weapon_type</th>
      <th>gun_class</th>
      <th>gun_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>Drug trade</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-119.42140</td>
      <td>36.34629</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>852.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>549</td>
      <td>2023-01-13</td>
      <td>Cleveland</td>
      <td>OH</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>None</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-81.70978</td>
      <td>41.45313</td>
      <td>Martin Muniz was arrested ad charged with kill...</td>
      <td>851.0</td>
      <td>gun</td>
      <td>HG</td>
      <td>semiautomatic handgun</td>
    </tr>
    <tr>
      <th>2</th>
      <td>548</td>
      <td>2023-01-07</td>
      <td>High Point</td>
      <td>NC</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>Despondency</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-79.97645</td>
      <td>36.00943</td>
      <td>Responding to cries for help, the police broke...</td>
      <td>850.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>547</td>
      <td>2023-01-04</td>
      <td>Enoch</td>
      <td>UT</td>
      <td>1</td>
      <td>7</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>37.76798</td>
      <td>-113.01698</td>
      <td>Carrying out a welfare check, the police disco...</td>
      <td>849.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>544</td>
      <td>2022-11-30</td>
      <td>Buffalo Grove</td>
      <td>IL</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>None</td>
      <td>Family</td>
      <td>Family issue</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-87.97934</td>
      <td>42.19935</td>
      <td>A man killed his wife, who had filed for divor...</td>
      <td>846.0</td>
      <td>knife</td>
      <td>NG</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging of Incidents and Offender Tables
# Query definition

mki_mko_q = """ 
            SELECT mki.incident_id, incidentdate, city, state, num_offenders,
              num_victims_killed, num_victims_injured, firstcod, secondcod,
              type, situation_type, location_type, location, longitude,
              latitude, narrative, offender_id, firstname,
              middlename, lastname, suffix, 
              CASE 
                WHEN age < 0 THEN NULL
                ELSE age
                END AS o_age, 
              race as o_race, sex as o_sex, suicide,
              deathcause, outcome, criminal_justice_process, sentence_type,
              sentence_details
            FROM mki
              LEFT JOIN mko
              ON mki.incident_id = mko.incident_id
           
        """
# Query execution
mki_mko = sqldf(mki_mko_q).copy()
#mki_mko['incidentdate'] = pd.to_datetime(mki_mko['date'])
#mki_mko = mki_mko.drop(columns=['date'])
#mki_mko['num_victims_injured'] = mki_mko['num_victims_injured'].astype(int)
mki_mko['incident_id'] = mki_mko['incident_id'].astype(str)
mki_mko['offender_id'] = mki_mko['offender_id'].astype(str)
mki_mko = mki_mko.drop_duplicates()
mki_mko
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
      <th>suffix</th>
      <th>o_age</th>
      <th>o_race</th>
      <th>o_sex</th>
      <th>suicide</th>
      <th>deathcause</th>
      <th>outcome</th>
      <th>criminal_justice_process</th>
      <th>sentence_type</th>
      <th>sentence_details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>549</td>
      <td>2023-01-13</td>
      <td>Cleveland</td>
      <td>OH</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>...</td>
      <td>None</td>
      <td>41.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Arrested/Pending trial</td>
      <td>Arrested/Pending trial</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>548</td>
      <td>2023-01-07</td>
      <td>High Point</td>
      <td>NC</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>...</td>
      <td>Jr.</td>
      <td>45.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>Suicide</td>
      <td>Not applicable</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>547</td>
      <td>2023-01-04</td>
      <td>Enoch</td>
      <td>UT</td>
      <td>1</td>
      <td>7</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>...</td>
      <td>None</td>
      <td>42.0</td>
      <td>White</td>
      <td>Male</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>Suicide</td>
      <td>Not applicable</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>544</td>
      <td>2022-11-30</td>
      <td>Buffalo Grove</td>
      <td>IL</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>None</td>
      <td>Family</td>
      <td>...</td>
      <td>None</td>
      <td>39.0</td>
      <td>None</td>
      <td>Male</td>
      <td>1.0</td>
      <td>None</td>
      <td>Suicide</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
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
    </tr>
    <tr>
      <th>668</th>
      <td>232</td>
      <td>2006-02-24</td>
      <td>Brooklyn</td>
      <td>NY</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>Smoke inhalation &amp; burns</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>None</td>
      <td>32.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Charges dropped</td>
      <td>Charges dropped</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>669</th>
      <td>97</td>
      <td>2006-02-21</td>
      <td>Mesa</td>
      <td>AZ</td>
      <td>1</td>
      <td>5</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Felony</td>
      <td>...</td>
      <td>None</td>
      <td>28.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Death sentence</td>
      <td>Trial</td>
      <td>Death sentence</td>
      <td>Sentenced to death on five counts of first deg...</td>
    </tr>
    <tr>
      <th>670</th>
      <td>109</td>
      <td>2006-01-30</td>
      <td>Goleta</td>
      <td>CA</td>
      <td>1</td>
      <td>7</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Public</td>
      <td>...</td>
      <td>None</td>
      <td>44.0</td>
      <td>White</td>
      <td>Female</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>Suicide</td>
      <td>Not applicable</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>671</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>None</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Death sentence</td>
      <td>Trial</td>
      <td>Death sentence</td>
      <td>None</td>
    </tr>
    <tr>
      <th>672</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>None</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Life without parole</td>
      <td>Plea</td>
      <td>Life sentence without parole</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>673 rows × 30 columns</p>
</div>




```python
# Merging of Incidents, Victim, and Weapon Tables
# Query definition
mki_mkv_mkw_q = """ 
            SELECT mki.incident_id, incidentdate, city, state, num_offenders,
            num_victims_killed, num_victims_injured, firstcod, secondcod,
            type, situation_type, location_type, location, longitude,
            latitude, narrative, victim_id, age as v_age, race as v_race,
            sex as v_sex, vorelationship, weapon_id, weapon_type,
            gun_class, gun_type
            FROM mki
              LEFT JOIN mkv
              ON mki.incident_id = mkv.incident_id
              LEFT JOIN mkw
              ON mki.incident_id = mkw.incident_id
                 
        """
# Query execution and convert to dataframe
mki_mkv_mkw = sqldf(mki_mkv_mkw_q).copy()

#Cleaning Dataframe
#mki_mkv_mkw['incidentdate'] = pd.to_datetime(mki_mkv_mkw['date'])
#mki_mkv_mkw = mki_mkv_mkw.drop(columns=['date'])
#mki_mkv_mkw['num_victims_injured'] = mki_mkv_mkw['num_victims_injured'].astype(int)
mki_mkv_mkw['incident_id'] = mki_mkv_mkw['incident_id'].astype(str)
mki_mkv_mkw['victim_id'] = mki_mkv_mkw['victim_id'].astype(str)
mki_mkv_mkw['weapon_id'] = mki_mkv_mkw['weapon_id'].astype(str)
#mki_mkv_mkw['victim_age'] = mki_mkv_mkw['victim_age'].astype(int)
mki_mkv_mkw = mki_mkv_mkw.drop_duplicates()
mki_mkv_mkw
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
      <th>narrative</th>
      <th>victim_id</th>
      <th>v_age</th>
      <th>v_race</th>
      <th>v_sex</th>
      <th>vorelationship</th>
      <th>weapon_id</th>
      <th>weapon_type</th>
      <th>gun_class</th>
      <th>gun_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2848</td>
      <td>72.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>852.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2849</td>
      <td>52.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>None</td>
      <td>852.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2850</td>
      <td>19.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>None</td>
      <td>852.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2851</td>
      <td>49.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>852.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>Six people were fatally shot inside and outsid...</td>
      <td>2852</td>
      <td>16.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>852.0</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
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
    </tr>
    <tr>
      <th>5977</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>Ricky Gray, 28, was convicted of murdering a f...</td>
      <td>507</td>
      <td>21.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Criminal associate</td>
      <td>159.0</td>
      <td>blunt object</td>
      <td>NG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5978</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>Ricky Gray, 28, was convicted of murdering a f...</td>
      <td>508</td>
      <td>46.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Relative of a known person</td>
      <td>158.0</td>
      <td>knife</td>
      <td>NG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5979</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>Ricky Gray, 28, was convicted of murdering a f...</td>
      <td>508</td>
      <td>46.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Relative of a known person</td>
      <td>159.0</td>
      <td>blunt object</td>
      <td>NG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5980</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>Ricky Gray, 28, was convicted of murdering a f...</td>
      <td>509</td>
      <td>55.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Relative of a known person</td>
      <td>158.0</td>
      <td>knife</td>
      <td>NG</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5981</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>Ricky Gray, 28, was convicted of murdering a f...</td>
      <td>509</td>
      <td>55.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Relative of a known person</td>
      <td>159.0</td>
      <td>blunt object</td>
      <td>NG</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>5982 rows × 25 columns</p>
</div>




```python
# Merging of Incidents, Victim, Weapon, and Offender Tables
# Query definition
mki_all_q = """ 
            SELECT 
              mki.incident_id, incidentdate, city, state, num_offenders,
              num_victims_killed, num_victims_injured, firstcod, secondcod,
              type, situation_type, location_type, location, longitude,
              latitude, narrative, victim_id, mkv.age as v_age, mkv.race as v_race,
              mkv.sex as v_sex, vorelationship, weapon_id, weapon_type,
              gun_class, gun_type, offender_id, firstname,
              middlename, lastname, suffix, mko.age as o_age, mko.race as o_race, mko.sex as o_sex, suicide,
              deathcause, outcome, criminal_justice_process, sentence_type,
              sentence_details
            FROM mki
              LEFT JOIN mkv
              ON mki.incident_id = mkv.incident_id
              LEFT JOIN mkw
              ON mki.incident_id = mkw.incident_id
              LEFT JOIN mko
              ON mki.incident_id = mko.incident_id
            
        """
# Query execution and convert to dataframe
mki_all = sqldf(mki_all_q).copy()

#Cleaning Dataframe
#mki_all['incidentdate'] = pd.to_datetime(mki_all['date'])
#mki_all= mki_all.drop(columns=['date'])
#mki_all['num_victims_injured'] = mki_all['num_victims_injured'].astype(int)
mki_all['incident_id'] = mki_all['incident_id'].astype(str)
mki_all['victim_id'] = mki_all['victim_id'].astype(str)
mki_all['weapon_id'] = mki_all['weapon_id'].astype(str)
mki_all['offender_id'] = mki_all['offender_id'].astype(str)
mki_all.astype({'num_victims_killed':'int'})
mki_all.astype({'num_offenders':'int'})
mki_all = mki_all.drop(columns= ['firstname','middlename','lastname','suffix','criminal_justice_process','sentence_type','sentence_details','weapon_id'])
mki_all = mki_all.drop_duplicates()
mki_all

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
      <th>weapon_type</th>
      <th>gun_class</th>
      <th>gun_type</th>
      <th>offender_id</th>
      <th>o_age</th>
      <th>o_race</th>
      <th>o_sex</th>
      <th>suicide</th>
      <th>deathcause</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>gun</td>
      <td>UG</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
      <td>None</td>
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
    </tr>
    <tr>
      <th>7354</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>blunt object</td>
      <td>NG</td>
      <td>None</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Life without parole</td>
    </tr>
    <tr>
      <th>7355</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>knife</td>
      <td>NG</td>
      <td>None</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Death sentence</td>
    </tr>
    <tr>
      <th>7356</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>knife</td>
      <td>NG</td>
      <td>None</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Life without parole</td>
    </tr>
    <tr>
      <th>7357</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>blunt object</td>
      <td>NG</td>
      <td>None</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Death sentence</td>
    </tr>
    <tr>
      <th>7358</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>blunt object</td>
      <td>NG</td>
      <td>None</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>None</td>
      <td>Life without parole</td>
    </tr>
  </tbody>
</table>
<p>4961 rows × 31 columns</p>
</div>




```python
# Merging of Incidents, Victim, and Offender Tables
# Query definition
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
mki_mkv_mko
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
      <th>0</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>72.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>52.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>19.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>49.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550</td>
      <td>2023-01-16</td>
      <td>Goshen</td>
      <td>CA</td>
      <td>1</td>
      <td>6</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Suspected felony</td>
      <td>...</td>
      <td>16.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>None</td>
      <td>704.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>None</td>
      <td>0.0</td>
      <td>None</td>
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
    </tr>
    <tr>
      <th>3399</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>21.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Criminal associate</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Life without parole</td>
    </tr>
    <tr>
      <th>3400</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>46.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Relative of a known person</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Death sentence</td>
    </tr>
    <tr>
      <th>3401</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>46.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Relative of a known person</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Life without parole</td>
    </tr>
    <tr>
      <th>3402</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>55.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Relative of a known person</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Death sentence</td>
    </tr>
    <tr>
      <th>3403</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>Strangulation</td>
      <td>Felony</td>
      <td>...</td>
      <td>55.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Relative of a known person</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Life without parole</td>
    </tr>
  </tbody>
</table>
<p>3404 rows × 27 columns</p>
</div>




```python
#Pandas Profiling Template -- Uncomment the profile you want to see. This uses a lot a memory and may take a while.

#Incident Profile
#profile_mk = ProfileReport(mk, title="Incident Profiling Report")
#profile_mk.to_notebook_iframe()

# Incident and Victim Profile
#profile_mki_mkv = ProfileReport(mki_mkv, title="Incident and Victims Profiling Report")
#profile_mki_mkv.to_notebook_iframe()

# Incident and Offender Profile
#profile_mki_mko = ProfileReport(mki_mko, title="Incident and Offender Profiling Report")
#profile_mki_mko.to_notebook_iframe()

# Incident and Weapon Profile
#profile_mki_mko = ProfileReport(mki_mko, title="Incident and Offender Profiling Report")
#profile_mki_mko.to_notebook_iframe()

# Incident, Victim, and Weapon Profile
#profile_mki_mkv_mkw = ProfileReport(mki_mkv_mkw, title="Incident, Victim, and Weapon Profiling Report")
#profile_mki_mkv_mkw.to_notebook_iframe()

# Incident, Victim, Weapon, and Offender Profile
#profile_mki_all = ProfileReport(mki_all, title="Incident, Victim, Weapon, Offender Profiling Report")
#profile_mki_all.to_notebook_iframe()

# Incident, Victim, and Offender Profile
#profile_mki_mkv_mko = ProfileReport(mki_mkv_mko, title="Incident, Victim, Offender Profiling Report")
#profile_mki_mkv_mko.to_notebook_iframe()

```

# EDA - Mass Killing Data | No Filters


```python
# Incident, Victim, Weapon, and Offender Profile
#profile_mki_all = ProfileReport(mki_all, title="Incident, Victim, Weapon, Offender Profiling Report")
#profile_mki_all.to_notebook_iframe()

```


```python
#Columns in the dataset
mki_all.columns
```




    Index(['incident_id', 'incidentdate', 'city', 'state', 'num_offenders',
           'num_victims_killed', 'num_victims_injured', 'firstcod', 'secondcod',
           'type', 'situation_type', 'location_type', 'location', 'longitude',
           'latitude', 'narrative', 'victim_id', 'v_age', 'v_race', 'v_sex',
           'vorelationship', 'weapon_type', 'gun_class', 'gun_type', 'offender_id',
           'o_age', 'o_race', 'o_sex', 'suicide', 'deathcause', 'outcome'],
          dtype='object')




```python
# Data types in the dataset
mki_all.dtypes
```




    incident_id             object
    incidentdate            object
    city                    object
    state                   object
    num_offenders            int64
    num_victims_killed       int64
    num_victims_injured    float64
    firstcod                object
    secondcod               object
    type                    object
    situation_type          object
    location_type           object
    location                object
    longitude              float64
    latitude               float64
    narrative               object
    victim_id               object
    v_age                  float64
    v_race                  object
    v_sex                   object
    vorelationship          object
    weapon_type             object
    gun_class               object
    gun_type                object
    offender_id             object
    o_age                  float64
    o_race                  object
    o_sex                   object
    suicide                float64
    deathcause              object
    outcome                 object
    dtype: object




```python
#How many incidents per state?
inc_per_state = mki.groupby('state')['incident_id'].count().sort_values(ascending = False)
inc_per_state.head()
```




    state
    CA    58
    TX    44
    IL    31
    FL    28
    OH    26
    Name: incident_id, dtype: int64




```python
#How many victims per state?
v_per_state = mki.groupby('state')['num_victims_killed'].sum().sort_values(ascending = False)
v_per_state.head()
```




    state
    CA    297
    TX    280
    FL    183
    IL    149
    OH    123
    Name: num_victims_killed, dtype: int64




```python
# Top states for incidents?
top5_state = mki.groupby('state')['incident_id'].count().sort_values(ascending = False).to_frame()
top5_state.head(5)
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
    </tr>
    <tr>
      <th>state</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CA</th>
      <td>58</td>
    </tr>
    <tr>
      <th>TX</th>
      <td>44</td>
    </tr>
    <tr>
      <th>IL</th>
      <td>31</td>
    </tr>
    <tr>
      <th>FL</th>
      <td>28</td>
    </tr>
    <tr>
      <th>OH</th>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top states for incidents?
top5_v_state = mki.groupby('state')['num_victims_killed'].sum().sort_values(ascending = False).to_frame()
top5_v_state.head(5)
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
      <th>num_victims_killed</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CA</th>
      <td>297</td>
    </tr>
    <tr>
      <th>TX</th>
      <td>280</td>
    </tr>
    <tr>
      <th>FL</th>
      <td>183</td>
    </tr>
    <tr>
      <th>IL</th>
      <td>149</td>
    </tr>
    <tr>
      <th>OH</th>
      <td>123</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Average of Age of Victims
mki_all.describe()

# Average age is approximately 33 across all vicimts in the dataset
# It's interesting to look into the 9yr old offender and the 0yr old victim age. (Gaffar comment)
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
      <th>num_offenders</th>
      <th>num_victims_killed</th>
      <th>num_victims_injured</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>v_age</th>
      <th>o_age</th>
      <th>suicide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4961.000000</td>
      <td>4961.000000</td>
      <td>4945.000000</td>
      <td>4961.000000</td>
      <td>4961.000000</td>
      <td>4928.000000</td>
      <td>4771.000000</td>
      <td>4948.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.666599</td>
      <td>9.249143</td>
      <td>35.953286</td>
      <td>-92.938705</td>
      <td>36.923645</td>
      <td>31.503044</td>
      <td>32.888074</td>
      <td>0.286580</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.263680</td>
      <td>12.453381</td>
      <td>161.910718</td>
      <td>16.541662</td>
      <td>7.458446</td>
      <td>19.948645</td>
      <td>12.114913</td>
      <td>0.452209</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>-149.108573</td>
      <td>-113.016980</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>-104.831919</td>
      <td>33.776069</td>
      <td>17.000000</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>-87.753945</td>
      <td>37.257717</td>
      <td>28.000000</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>-81.218693</td>
      <td>40.707002</td>
      <td>45.000000</td>
      <td>40.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>60.000000</td>
      <td>867.000000</td>
      <td>37.767980</td>
      <td>61.609945</td>
      <td>98.000000</td>
      <td>73.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#What was the first cause of deaths for each individual victim in mass killings? (Assuming that first cause of death may not be the same for each victim)
FirstCOD = mki_mkv.groupby('firstcod')['incident_id'].count().sort_values(ascending = False).to_frame()
FirstCOD

# Looks like certain causes of death might need to be filtered out. 
# My initial thoughts are vehicle crash, smoke inhalation & burns and unknown. Need to look at the narratives to understand how these are being categorized
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
    </tr>
    <tr>
      <th>firstcod</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Shooting</th>
      <td>2240</td>
    </tr>
    <tr>
      <th>Smoke inhalation &amp; burns</th>
      <td>169</td>
    </tr>
    <tr>
      <th>Stabbing</th>
      <td>167</td>
    </tr>
    <tr>
      <th>Blunt force</th>
      <td>127</td>
    </tr>
    <tr>
      <th>Vehicle crash</th>
      <td>33</td>
    </tr>
    <tr>
      <th>Asphyxiation</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Strangulation</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Drowning</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Pushing/Jumping</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Where did these incidents happen location wise?

loco = mki.groupby(['location'])['incident_id'].count().sort_values(ascending = False).to_frame()
loco

# Looks like certain locations need to be filtered out like vehicle

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
    </tr>
    <tr>
      <th>location</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Residence</th>
      <td>361</td>
    </tr>
    <tr>
      <th>Open space</th>
      <td>35</td>
    </tr>
    <tr>
      <th>Commercial/Retail</th>
      <td>34</td>
    </tr>
    <tr>
      <th>Multiple</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Bar/Club/Restaurant</th>
      <td>19</td>
    </tr>
    <tr>
      <th>Vehicle</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Government/Transit</th>
      <td>10</td>
    </tr>
    <tr>
      <th>House of worship</th>
      <td>7</td>
    </tr>
    <tr>
      <th>School</th>
      <td>7</td>
    </tr>
    <tr>
      <th>College</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Shelter/Drug house</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Hotel/Motel</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Medical facility</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Where did these incidents happen by situation type?

sit_type = mki.groupby(['type'])['incident_id'].count().sort_values(ascending = False).to_frame()
sit_type

# Looks 
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
    </tr>
    <tr>
      <th>type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Family</th>
      <td>253</td>
    </tr>
    <tr>
      <th>Public</th>
      <td>100</td>
    </tr>
    <tr>
      <th>Felony</th>
      <td>97</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>41</td>
    </tr>
    <tr>
      <th>Unsolved</th>
      <td>21</td>
    </tr>
    <tr>
      <th>Suspected felony</th>
      <td>18</td>
    </tr>
    <tr>
      <th>Undetermined</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What types of weapons were used?

wea_use = mki_mkw.groupby(['weapon_type'])['incident_id'].nunique().sort_values(ascending = False).to_frame()
wea_use

# Guns are the leading weapons used
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
    </tr>
    <tr>
      <th>weapon_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gun</th>
      <td>416</td>
    </tr>
    <tr>
      <th>fire</th>
      <td>53</td>
    </tr>
    <tr>
      <th>sharp object</th>
      <td>39</td>
    </tr>
    <tr>
      <th>blunt object</th>
      <td>37</td>
    </tr>
    <tr>
      <th>knife</th>
      <td>36</td>
    </tr>
    <tr>
      <th>strangulation</th>
      <td>15</td>
    </tr>
    <tr>
      <th>other</th>
      <td>9</td>
    </tr>
    <tr>
      <th>vehicle</th>
      <td>9</td>
    </tr>
    <tr>
      <th>explosive</th>
      <td>4</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What types of guns were used?

gun_use = mki_mkw.groupby(['gun_type'])['incident_id'].nunique().sort_values(ascending = False).to_frame()
gun_use

# Handguns, Semiautomatc, rifles, and shotguns are the lead respectively
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
    </tr>
    <tr>
      <th>gun_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>semiautomatic handgun</th>
      <td>99</td>
    </tr>
    <tr>
      <th>handgun</th>
      <td>98</td>
    </tr>
    <tr>
      <th>rifle</th>
      <td>47</td>
    </tr>
    <tr>
      <th>shotgun</th>
      <td>42</td>
    </tr>
    <tr>
      <th>revolver</th>
      <td>34</td>
    </tr>
    <tr>
      <th>semiautomatic rifle</th>
      <td>32</td>
    </tr>
    <tr>
      <th>pistol</th>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Avg Age of Offender?

off_age = mko['age'].mean()
off_age

# Handguns, Semiautomatc, rifles, and shotguns are the lead respectively
```




    32.29606299212598




```python
# Offender Sex

off_sex = mki_mko.groupby(['o_sex'])['incident_id'].nunique().sort_values(ascending = False).to_frame()
off_sex

# Heavily skewed to males
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
    </tr>
    <tr>
      <th>o_sex</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>475</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Race of the Offender?

off_race = mki_mko.groupby(['o_race'])['incident_id'].nunique().sort_values(ascending = False).to_frame()
off_race

# Top 3 White, Black, Hispanic/Latino respectively
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
    </tr>
    <tr>
      <th>o_race</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>227</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>160</td>
    </tr>
    <tr>
      <th>Hispanic/Latino</th>
      <td>63</td>
    </tr>
    <tr>
      <th>Asian/Pacific Islander</th>
      <td>27</td>
    </tr>
    <tr>
      <th>American Indian</th>
      <td>9</td>
    </tr>
    <tr>
      <th>White</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Race of the Offender and vicimts?

off_race = mki_mko.groupby(['o_race'])['num_victims_killed'].sum().sort_values(ascending = False).to_frame()
off_race

# Top 3 White, Black, Hispanic/Latino respectively
# Need to do this again as there is likely more than one offender for some incidents
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
      <th>num_victims_killed</th>
    </tr>
    <tr>
      <th>o_race</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>1390</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>991</td>
    </tr>
    <tr>
      <th>Hispanic/Latino</th>
      <td>373</td>
    </tr>
    <tr>
      <th>Asian/Pacific Islander</th>
      <td>255</td>
    </tr>
    <tr>
      <th>American Indian</th>
      <td>54</td>
    </tr>
    <tr>
      <th>White</th>
      <td>17</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Did Offenders use more than one weapon?
off_wea = mki_mko.groupby(['o_race'])['num_victims_killed'].sum().sort_values(ascending = False).to_frame()
off_race
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
      <th>num_victims_killed</th>
    </tr>
    <tr>
      <th>o_race</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>White</th>
      <td>1390</td>
    </tr>
    <tr>
      <th>Black</th>
      <td>991</td>
    </tr>
    <tr>
      <th>Hispanic/Latino</th>
      <td>373</td>
    </tr>
    <tr>
      <th>Asian/Pacific Islander</th>
      <td>255</td>
    </tr>
    <tr>
      <th>American Indian</th>
      <td>54</td>
    </tr>
    <tr>
      <th>White</th>
      <td>17</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Was there more than one offender per event?

```

# EDA - Life Expectancy


```python
# Importing Life Expectancy Data for the U.S. by Gender
with open('data/le/U.S._State_Life_Expectancy_by_Sex__2020.csv') as f:
    le=pd.read_csv(f, delimiter=',')

# Want to plot Life Expectancy for US for 2020
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
# Created for exporting for any cleaning work in Excel
state_name_export = pd.DataFrame.from_dict(us_state_to_abbrev, orient='index').reset_index()

state_name_export.columns = ['state', 'state_abb']

state_name_export
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
      <th>state</th>
      <th>state_abb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>CA</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>CO</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>CT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>DE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Florida</td>
      <td>FL</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Georgia</td>
      <td>GA</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hawaii</td>
      <td>HI</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Idaho</td>
      <td>ID</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Illinois</td>
      <td>IL</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Indiana</td>
      <td>IN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Iowa</td>
      <td>IA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Kansas</td>
      <td>KS</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Kentucky</td>
      <td>KY</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Louisiana</td>
      <td>LA</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Maine</td>
      <td>ME</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Maryland</td>
      <td>MD</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Massachusetts</td>
      <td>MA</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Michigan</td>
      <td>MI</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Minnesota</td>
      <td>MN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Mississippi</td>
      <td>MS</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Missouri</td>
      <td>MO</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Montana</td>
      <td>MT</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Nebraska</td>
      <td>NE</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nevada</td>
      <td>NV</td>
    </tr>
    <tr>
      <th>28</th>
      <td>New Hampshire</td>
      <td>NH</td>
    </tr>
    <tr>
      <th>29</th>
      <td>New Jersey</td>
      <td>NJ</td>
    </tr>
    <tr>
      <th>30</th>
      <td>New Mexico</td>
      <td>NM</td>
    </tr>
    <tr>
      <th>31</th>
      <td>New York</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>32</th>
      <td>North Carolina</td>
      <td>NC</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Dakota</td>
      <td>ND</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Ohio</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Oklahoma</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oregon</td>
      <td>OR</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Pennsylvania</td>
      <td>PA</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Rhode Island</td>
      <td>RI</td>
    </tr>
    <tr>
      <th>39</th>
      <td>South Carolina</td>
      <td>SC</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Dakota</td>
      <td>SD</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Tennessee</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Texas</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Utah</td>
      <td>UT</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Vermont</td>
      <td>VT</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Virginia</td>
      <td>VA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Washington</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>47</th>
      <td>West Virginia</td>
      <td>WV</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Wisconsin</td>
      <td>WI</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wyoming</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>50</th>
      <td>District of Columbia</td>
      <td>DC</td>
    </tr>
    <tr>
      <th>51</th>
      <td>American Samoa</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Guam</td>
      <td>GU</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Northern Mariana Islands</td>
      <td>MP</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Puerto Rico</td>
      <td>PR</td>
    </tr>
    <tr>
      <th>55</th>
      <td>United States Minor Outlying Islands</td>
      <td>UM</td>
    </tr>
    <tr>
      <th>56</th>
      <td>U.S. Virgin Islands</td>
      <td>VI</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Cleaning Life Exepctancy Data
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
mki_all_le_q = """ 
            SELECT *,le.le - v_age as yrs_life_lost
            FROM mki_all
              LEFT JOIN le_state as le
              ON mki_all.state = le.state and mki_all.v_sex = le.sex
       
        """
# Query execution and convert to dataframe
mki_all_le = sqldf(mki_all_le_q).copy()
mki_all_le = mki_all_le.drop(columns=["State","Sex","Quartile"])
mki_all_le.replace(to_replace=[None], value=np.nan, inplace=True)
mki_all_le['incidentdate'] = pd.to_datetime(mki_all_le['incidentdate'],format="%Y-%m-%d")
mki_all_le['month'] = mki_all_le['incidentdate'].dt.month.astype('category')
mki_all_le['year'] = mki_all_le['incidentdate'].dt.year
mki_all_le['season'] = mki_all_le['month'].map(month_season)
mki_all_le['secondcod'] = mki_all_le['secondcod'].fillna('None')
mki_all_le['first_second_cod'] = mki_all_le['firstcod'] + '-' + mki_all_le['secondcod']
mki_all_le = mki_all_le[mki_all_le['year'] <= 2022].copy()
mki_all_le['season_month'] = mki_all_le['season'] + '-' + mki_all_le['month'].astype(str)
mki_all_le['num_offenders'] = mki_all_le['num_offenders'].astype(int)
mki_all_le['region_state'] = mki_all_le['region'] + '-' + mki_all_le['state']
mki_all_le['o_race_sex'] = mki_all_le['o_race'] + '-' + mki_all_le['o_sex']
mki_all_le['v_race_sex'] = mki_all_le['v_race'] + '-' + mki_all_le['v_sex']



# function for age range of offender
def age_range(x):
    if x < 33:
        return 'Below Avg Age (33)'
    else:
        return 'Above Avg Age (33)'
mki_all_le['o_age_range'] = mki_all_le['o_age'].apply(age_range)

# function for weapon type label
def weapon_gun(x):
    if x == 'gun':
        return 'Gun'
    else:
        return 'Not A Gun'

mki_all_le['gun_or_not'] = mki_all_le['weapon_type'].apply(weapon_gun)

# One-hot encode the 'region' variable
# mki_all_le = pd.get_dummies(mki_all_le, columns=['season'])

pd.set_option('display.max_columns', None)
mki_all_le.sample(3)
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
      <th>longitude</th>
      <th>latitude</th>
      <th>narrative</th>
      <th>victim_id</th>
      <th>v_age</th>
      <th>v_race</th>
      <th>v_sex</th>
      <th>vorelationship</th>
      <th>weapon_type</th>
      <th>gun_class</th>
      <th>gun_type</th>
      <th>offender_id</th>
      <th>o_age</th>
      <th>o_race</th>
      <th>o_sex</th>
      <th>suicide</th>
      <th>deathcause</th>
      <th>outcome</th>
      <th>LE</th>
      <th>SE</th>
      <th>region</th>
      <th>yrs_life_lost</th>
      <th>month</th>
      <th>year</th>
      <th>season</th>
      <th>first_second_cod</th>
      <th>season_month</th>
      <th>region_state</th>
      <th>o_race_sex</th>
      <th>v_race_sex</th>
      <th>o_age_range</th>
      <th>gun_or_not</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2613</th>
      <td>258</td>
      <td>2015-01-12</td>
      <td>Port Lavaca</td>
      <td>TX</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Stabbing</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-96.654733</td>
      <td>28.612962</td>
      <td>A fire set to a one-room trailer concealed the...</td>
      <td>1289</td>
      <td>0.0</td>
      <td>Hispanic/Latino</td>
      <td>Female</td>
      <td>Child or stepchild</td>
      <td>sharp object</td>
      <td>NG</td>
      <td>NaN</td>
      <td>323.0</td>
      <td>23.0</td>
      <td>Hispanic/Latino</td>
      <td>Male</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Life without parole</td>
      <td>79.3</td>
      <td>0.035</td>
      <td>Southwest</td>
      <td>79.3</td>
      <td>1</td>
      <td>2015</td>
      <td>winter</td>
      <td>Stabbing-None</td>
      <td>winter-1</td>
      <td>Southwest-TX</td>
      <td>Hispanic/Latino-Male</td>
      <td>Hispanic/Latino-Female</td>
      <td>Below Avg Age (33)</td>
      <td>Not A Gun</td>
    </tr>
    <tr>
      <th>3447</th>
      <td>305</td>
      <td>2011-05-11</td>
      <td>Ammon</td>
      <td>ID</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-111.947723</td>
      <td>43.487589</td>
      <td>Gaylin Leirmoe killed his sons, their mother, ...</td>
      <td>1579</td>
      <td>1.0</td>
      <td>White</td>
      <td>Male</td>
      <td>Child or stepchild</td>
      <td>gun</td>
      <td>UG</td>
      <td>NaN</td>
      <td>376.0</td>
      <td>26.0</td>
      <td>White</td>
      <td>Male</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>Suicide</td>
      <td>76.1</td>
      <td>0.150</td>
      <td>West</td>
      <td>75.1</td>
      <td>5</td>
      <td>2011</td>
      <td>spring</td>
      <td>Shooting-None</td>
      <td>spring-5</td>
      <td>West-ID</td>
      <td>White-Male</td>
      <td>White-Male</td>
      <td>Below Avg Age (33)</td>
      <td>Gun</td>
    </tr>
    <tr>
      <th>2724</th>
      <td>241</td>
      <td>2014-07-09</td>
      <td>Spring</td>
      <td>TX</td>
      <td>1</td>
      <td>6</td>
      <td>1.0</td>
      <td>Shooting</td>
      <td>None</td>
      <td>Family</td>
      <td>Interpersonal conflict</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>-95.442238</td>
      <td>30.041007</td>
      <td>A man bound and fatally shot six members of a ...</td>
      <td>1209</td>
      <td>4.0</td>
      <td>White</td>
      <td>Male</td>
      <td>Former relative/in-law</td>
      <td>gun</td>
      <td>HG</td>
      <td>semiautomatic handgun</td>
      <td>306.0</td>
      <td>33.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Death sentence</td>
      <td>73.7</td>
      <td>0.038</td>
      <td>Southwest</td>
      <td>69.7</td>
      <td>7</td>
      <td>2014</td>
      <td>summer</td>
      <td>Shooting-None</td>
      <td>summer-7</td>
      <td>Southwest-TX</td>
      <td>White-Male</td>
      <td>White-Male</td>
      <td>Above Avg Age (33)</td>
      <td>Gun</td>
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
mk_vo_le = mk_vo_le.drop(columns=["State","Sex","Quartile","SE","LE","index","year","latitude","longitude","firstcod","secondcod","narrative",])


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


# One-hot encode the 'region' variable
# mki_all_le = pd.get_dummies(mki_all_le, columns=['season'])

pd.set_option('display.max_columns', None)
mk_vo_le
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
      <td>Colorado Springs</td>
      <td>CO</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>Public</td>
      <td>Hate</td>
      <td>Commercial/Retail/Entertainment</td>
      <td>Bar/Club/Restaurant</td>
      <td>2805</td>
      <td>28.0</td>
      <td>White</td>
      <td>Male</td>
      <td>Random bystander/stranger</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Arrested/Pending trial</td>
      <td>West</td>
      <td>47.8</td>
      <td>11</td>
      <td>fall</td>
      <td>Shooting-None</td>
      <td>fall-11</td>
      <td>West-CO</td>
      <td>White-Male</td>
      <td>White-Male</td>
      <td>Colorado Springs-CO</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>1</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>Colorado Springs</td>
      <td>CO</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>Public</td>
      <td>Hate</td>
      <td>Commercial/Retail/Entertainment</td>
      <td>Bar/Club/Restaurant</td>
      <td>2806</td>
      <td>38.0</td>
      <td>White</td>
      <td>Male</td>
      <td>Random bystander/stranger</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Arrested/Pending trial</td>
      <td>West</td>
      <td>37.8</td>
      <td>11</td>
      <td>fall</td>
      <td>Shooting-None</td>
      <td>fall-11</td>
      <td>West-CO</td>
      <td>White-Male</td>
      <td>White-Male</td>
      <td>Colorado Springs-CO</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>2</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>Colorado Springs</td>
      <td>CO</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>Public</td>
      <td>Hate</td>
      <td>Commercial/Retail/Entertainment</td>
      <td>Bar/Club/Restaurant</td>
      <td>2807</td>
      <td>35.0</td>
      <td>White</td>
      <td>Female</td>
      <td>Random bystander/stranger</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Arrested/Pending trial</td>
      <td>West</td>
      <td>45.9</td>
      <td>11</td>
      <td>fall</td>
      <td>Shooting-None</td>
      <td>fall-11</td>
      <td>West-CO</td>
      <td>White-Male</td>
      <td>White-Female</td>
      <td>Colorado Springs-CO</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>3</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>Colorado Springs</td>
      <td>CO</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>Public</td>
      <td>Hate</td>
      <td>Commercial/Retail/Entertainment</td>
      <td>Bar/Club/Restaurant</td>
      <td>2808</td>
      <td>22.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Random bystander/stranger</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Arrested/Pending trial</td>
      <td>West</td>
      <td>53.8</td>
      <td>11</td>
      <td>fall</td>
      <td>Shooting-None</td>
      <td>fall-11</td>
      <td>West-CO</td>
      <td>White-Male</td>
      <td>Black-Male</td>
      <td>Colorado Springs-CO</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>4</th>
      <td>541</td>
      <td>2022-11-20</td>
      <td>Colorado Springs</td>
      <td>CO</td>
      <td>1</td>
      <td>5</td>
      <td>17.0</td>
      <td>Public</td>
      <td>Hate</td>
      <td>Commercial/Retail/Entertainment</td>
      <td>Bar/Club/Restaurant</td>
      <td>2809</td>
      <td>40.0</td>
      <td>White</td>
      <td>Female</td>
      <td>Random bystander/stranger</td>
      <td>693.0</td>
      <td>22.0</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Arrested/Pending trial</td>
      <td>West</td>
      <td>40.9</td>
      <td>11</td>
      <td>fall</td>
      <td>Shooting-None</td>
      <td>fall-11</td>
      <td>West-CO</td>
      <td>White-Male</td>
      <td>White-Female</td>
      <td>Colorado Springs-CO</td>
      <td>21-30 Years</td>
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
    </tr>
    <tr>
      <th>2832</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Felony</td>
      <td>Robbery</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>507</td>
      <td>21.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Criminal associate</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Life without parole</td>
      <td>South</td>
      <td>54.1</td>
      <td>1</td>
      <td>winter</td>
      <td>Stabbing-Strangulation</td>
      <td>winter-1</td>
      <td>South-VA</td>
      <td>Black-Male</td>
      <td>Black-Male</td>
      <td>Richmond-VA</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>2833</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Felony</td>
      <td>Robbery</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>508</td>
      <td>46.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Relative of a known person</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Death sentence</td>
      <td>South</td>
      <td>29.1</td>
      <td>1</td>
      <td>winter</td>
      <td>Stabbing-Strangulation</td>
      <td>winter-1</td>
      <td>South-VA</td>
      <td>Black-Male</td>
      <td>Black-Male</td>
      <td>Richmond-VA</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>2834</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Felony</td>
      <td>Robbery</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>508</td>
      <td>46.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>Relative of a known person</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Life without parole</td>
      <td>South</td>
      <td>29.1</td>
      <td>1</td>
      <td>winter</td>
      <td>Stabbing-Strangulation</td>
      <td>winter-1</td>
      <td>South-VA</td>
      <td>Black-Male</td>
      <td>Black-Male</td>
      <td>Richmond-VA</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Felony</td>
      <td>Robbery</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>509</td>
      <td>55.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Relative of a known person</td>
      <td>133.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Death sentence</td>
      <td>South</td>
      <td>25.1</td>
      <td>1</td>
      <td>winter</td>
      <td>Stabbing-Strangulation</td>
      <td>winter-1</td>
      <td>South-VA</td>
      <td>Black-Male</td>
      <td>Black-Female</td>
      <td>Richmond-VA</td>
      <td>21-30 Years</td>
    </tr>
    <tr>
      <th>2836</th>
      <td>98</td>
      <td>2006-01-01</td>
      <td>Richmond</td>
      <td>VA</td>
      <td>2</td>
      <td>7</td>
      <td>0.0</td>
      <td>Felony</td>
      <td>Robbery</td>
      <td>Residence/Other shelter</td>
      <td>Residence</td>
      <td>509</td>
      <td>55.0</td>
      <td>Black</td>
      <td>Female</td>
      <td>Relative of a known person</td>
      <td>134.0</td>
      <td>28.0</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>Life without parole</td>
      <td>South</td>
      <td>25.1</td>
      <td>1</td>
      <td>winter</td>
      <td>Stabbing-Strangulation</td>
      <td>winter-1</td>
      <td>South-VA</td>
      <td>Black-Male</td>
      <td>Black-Female</td>
      <td>Richmond-VA</td>
      <td>21-30 Years</td>
    </tr>
  </tbody>
</table>
<p>2837 rows × 33 columns</p>
</div>




```python
with open('data/mass_killing_yrs_lost.csv', 'w') as file:
     mki_all_le.to_csv(file, index=False)
```


```python
mk_vo_le.columns
```




    Index(['incident_id', 'incidentdate', 'city', 'state', 'num_offenders',
           'num_victims_killed', 'num_victims_injured', 'type', 'situation_type',
           'location_type', 'location', 'victim_id', 'v_age', 'v_race', 'v_sex',
           'vorelationship', 'offender_id', 'o_age', 'o_race', 'o_sex', 'suicide',
           'outcome', 'region', 'yrs_life_lost', 'month', 'season',
           'first_second_cod', 'season_month', 'region_state', 'o_race_sex',
           'v_race_sex', 'city_state', 'o_age_range'],
          dtype='object')




```python
"""
temp_df = pd.DataFrame(mki_all_le['vorelationship'].unique())
temp_df.style.hide_index()
"""
```




    "\ntemp_df = pd.DataFrame(mki_all_le['vorelationship'].unique())\ntemp_df.style.hide_index()\n"


