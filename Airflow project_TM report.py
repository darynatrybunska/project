
# coding: utf-8

# In[323]:


import pandas as pd
import numpy as np

df = pd.read_csv('/Users/darynatrybunska/Downloads/2_taxi_nyc.csv')


# In[290]:


def to_csv(df):
    path = sys.path
    df.to_csv('/Users/darynatrybunska/airflow/output.csv')


# In[291]:


to_csv(df)


# In[5]:


import pandas as pd
import numpy as np

ad = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vR-ti6Su94955DZ4Tky8EbwifpgZf_dTjpBdiVH0Ukhsq94jZdqoHuUytZsFZKfwpXEUCKRFteJRc9P/pub?gid=889004448&single=true&output=csv')

ad_metrics = ad     .groupby(['date','event'], as_index=False)     .agg({'ad_id':'count'})     .pivot(index ='date', columns='event', values='ad_id')     .reset_index()
    
    
ad_metrics = ad_metrics.assign(CTR = round(ad_metrics.click/ad_metrics.view *100,2),
                               money_spent = int(ad.ad_cost.unique())/1000 * ad_metrics.view)

ad_metrics = ad_metrics.transpose().reset_index()

ad_metrics.columns = ad_metrics.iloc[0]
ad_metrics = ad_metrics.drop([0], axis=0)

ad_metrics = ad_metrics.assign(diff = 100*((ad_metrics.iloc[: ,2]-ad_metrics.iloc[: ,1])/ ad_metrics.iloc[: ,1]))


ad_report = ad_metrics             .replace(['click','view','money_spent'],['Клики','Показы','Траты'])             .set_index('date')

column = ad_report.columns[1]
header = "Отчет по объявлению 121288 за {} \n".format(column)

lines = ["{}: {:.2f} ({:+.2f}%) \n".format(index, row [column], row['diff']) for index, row in ad_report.iterrows()]
lines = [header] + lines

output_file = open("report.txt","w") 
output_file.writelines(lines)
output_file.close()


# In[6]:


for i in lines:
    print(i)


# In[7]:


import requests
import json
from urllib.parse import urlencode

token = '1094615560:AAGSDuV80JeE43T4jA5kCX6ghaAWHTnDLR4'
chat_id = 434223784  # your chat id
message = 'Report is ready'
params = {'chat_id': chat_id, 'text': message}
base_url = f'https://api.telegram.org/bot{token}/'


filepath = 'report.txt'
url = base_url + 'sendDocument?' + urlencode(params)
files = {'document': open(filepath, 'rb')}

resp = requests.get(url, files=files)


# In[ ]:


from airflow import DAG
from airflow.operators import BashOperator
from datetime import datetime
import sys

default_args = {
    'owner': 'dtrybunska',
    'depends_on_past': False,
    'start_date': datetime(2020, 7, 6),
    'retries': 0
}

dag = DAG('report_prep', 
    default_args=default_args,
    schedule_interval='00 12 * * mon')

    
t1 = BashOperator(
    task_id='report_generation',
    bash_command='python /home/airflow/dags/dtrybunska/report_generation.py',
    dag=dag)

t2 = BashOperator(
    task_id='report_share_telegram',
    bash_command='python /home/airflow/dags/dtrybunska/report_share_telegram.py',
    dag=dag)


t1 >> t2

