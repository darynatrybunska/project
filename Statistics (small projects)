#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandahouse as ph
import pandas as pd
from ast import literal_eval
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from datetime import datetime as dt
import matplotlib.dates as mdates
import statsmodels.stats.api as sms
import numpy as np, scipy.stats as st
from scipy import stats
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set(
    font_scale=1,
    style="whitegrid",
    rc={'figure.figsize':(20,7)}
        )


# In[2]:


x = np.arange(100)
np.std(x,ddof=1)


# # Mini Project 1

# In[3]:


df = pd.read_csv('https://stepik.org/media/attachments/lesson/383837/games.csv')


# In[4]:


df.head()
df.shape


# In[5]:


df.dtypes.sort_values()


# In[6]:


df.isnull().sum()


# In[7]:


df_nan = df.dropna()


# In[8]:


df_nan.shape


# In[9]:


df_nan['Year'].describe()


# In[10]:


df_nan['Year'].mode()


# In[11]:


df_nan['Year'].median()


# In[12]:


sns.distplot(df_nan.Year)


# In[13]:


df_nan.groupby('Platform', as_index = False)                .aggregate({'Name':'count'})                 .sort_values('Name', ascending=False )                 .head()


# In[14]:


df_nan['Platform'].value_counts(normalize=True).apply(lambda x: round(x*100,2)).head()


# In[15]:


df_nan['Platform'].value_counts(normalize=True).apply(lambda x: round(x*100,2)).index


# In[16]:


df_nan['Publisher'].value_counts().index


# In[17]:


nintendo = df_nan.query('Publisher == "Nintendo"')
nintendo


# In[18]:


nintendo.mean()


# In[19]:


nintendo.median()


# In[20]:


nintendo.std()


# In[21]:


ax = sns.boxplot(x="Genre", y="JP_Sales", data=nintendo)


# In[22]:


sns.lineplot(x="Year", y="Global_Sales", hue= "Genre",
            data=nintendo)


# In[23]:


(181 - 173)/11


# In[24]:


(192 - 173)/11


# # Mini project 2

# In[25]:


ads = pd.read_csv('https://stepik.org/media/attachments/lesson/384453/conversion.csv')


# In[26]:


ads


# In[27]:


ads.shape


# In[28]:


ads.dtypes


# In[29]:


ads.isna().sum()


# In[30]:


ads.xyz_campaign_id.unique()


# In[31]:


sns.distplot(np.log(ads.Impressions))


# In[32]:


ads['ctr']=ads.Clicks/ads.Impressions


# In[33]:


ads.ctr.describe()


# In[34]:


ads.sort_values('ctr', ascending = False)


# In[35]:


sns.distplot(ads.query("xyz_campaign_id == 936")['ctr'],bins=20)


# In[36]:


sns.distplot(ads.query("xyz_campaign_id == 916")['ctr'],bins=20)


# In[37]:


sns.distplot(ads.query("xyz_campaign_id == 1178")['ctr'],bins=20)


# In[38]:


ads['cpc'] = ads.Spent/ads.Clicks


# In[39]:


sns.distplot(ads.cpc.dropna())


# In[40]:


ads.cpc.describe()


# In[41]:


round(iqr(ads.cpc,nan_policy='omit'),2)


# In[42]:


sns.distplot(ads.query("gender == 'F'")['cpc'])
sns.distplot(ads.query("gender == 'M'")['cpc'])


# In[43]:


ads['conversion'] = (ads.Approved_Conversion/ads.Clicks)*100


# In[44]:


ads[ads['ad_id'] == 1121814]


# In[45]:


2.58*0.5


# # Mini project 3

# In[46]:


bcl = pd.read_csv('https://stepik.org/media/attachments/lesson/384464/london.csv')
bcl.timestamp = pd.to_datetime(bcl.timestamp)


# In[47]:


bcl.dtypes


# In[48]:


bcl.head()


# In[49]:


bcl.shape


# In[50]:


bcl.isna().sum()


# In[51]:


sns.lineplot(x='timestamp',y='cnt', data=bcl)


# In[52]:


bcl_time = bcl.set_index('timestamp')
bcl_time.head()


# In[53]:


sns.lineplot(data = bcl_time.resample(rule='D').cnt.sum())


# In[ ]:


#На данном шаге возьмите агрегированные данные по дням с предыдущего шага и посчитайте скользящее среднее с окном 3. 
#В качестве ответа укажите полученное число аренд за 2015-07-09, округлив значение до целого.


# In[74]:


bcl['date'] = bcl['timestamp'].dt.normalize()


# In[72]:


bcl['date']


# In[103]:


cnt_date = bcl_time.resample(rule='D').cnt.sum()
cnt_date_rol = cnt_date.rolling(window=3).mean()


# In[104]:


cnt_date_rol


# In[105]:


cnt_date_rol['2015-07-09']


# In[106]:


#Теперь посчитайте разницу между наблюдаемыми и подсчитанными значениями. 
#Далее – примените функцию для подсчета стандартного отклонения

c = cnt_date - cnt_date_rol
std = np.std(c)


# In[107]:


std


# In[108]:


#Теперь определим границы интервалов. Для этого нужно взять данные, полученные при расчете скользящего среднего, 
#и создать следующие объекты:

#upper_bound – верхняя граница; к средним прибавляем 2.576 * std
#lower_bound – нижняя граница; вычитаем 2.576 * std
#Полученные значения запишите в новые столбцы датафрейма с агрегированными данными.


# In[128]:


cnt_date_rolling = cnt_date_rol.to_frame().reset_index()


# In[129]:


cnt_date_rolling['upper_bound'] = cnt_date_rolling.cnt +(2.576*std)
cnt_date_rolling['lower_bound'] = cnt_date_rolling.cnt -(2.576*std)


# In[136]:


cnt_date_original = cnt_date.to_frame().reset_index()


# In[137]:


cnt_date_original


# In[138]:


cnt_date_rolling['cnt_original'] = cnt_date_original.cnt


# In[139]:


cnt_date_rolling


# In[ ]:


#values with results more than 99% of reliable interval


# In[143]:


cnt_date_rolling.query('cnt_original > upper_bound').sort_values('cnt_original', ascending=False)


# In[147]:


bcl.query('date == "2015-07-09"')


# In[148]:


bcl.query('date == "2015-07-08"')


# In[149]:


#values with results less than 99% of reliable interval
cnt_date_rolling.query('cnt_original < lower_bound').sort_values('cnt_original', ascending=False)


# In[150]:


bcl.query('date == "2016-09-02"')


# # Mini project 4

# In[151]:


delivery = pd.read_csv('https://stepik.org/media/attachments/lesson/385916/experiment_lesson_4.csv')


# In[152]:


delivery.head()


# In[153]:


delivery.dtypes


# In[154]:


delivery.isna().sum()


# In[155]:


delivery.shape


# In[160]:


delivery[delivery.experiment_group == 'test'].shape


# In[161]:


delivery[delivery.experiment_group == 'test'].mean()


# In[185]:


#check for normality of destribution
stats.shapiro(delivery[delivery['experiment_group'] == 'test']['delivery_time'].sample(1000))


# In[189]:


std_test = np.std(delivery[delivery['experiment_group'] == 'test'].delivery_time)
std_test


# In[169]:


sns.distplot(delivery[delivery.experiment_group == 'test'].delivery_time, bins=20, kde=False)


# In[171]:


delivery[delivery.experiment_group == 'control'].shape


# In[172]:


delivery[delivery.experiment_group == 'control'].mean()


# In[186]:


#check for normality of destribution
stats.shapiro(delivery[delivery['experiment_group'] == 'control']['delivery_time'].sample(1000))


# In[190]:


std_control = np.std(delivery[delivery['experiment_group'] == 'control'].delivery_time)
std_control


# In[170]:


sns.distplot(delivery[delivery.experiment_group == 'control'].delivery_time, bins=20, kde=False)


# In[ ]:





# In[158]:


delivery[delivery.experiment_group == 'control']


# In[187]:


# diff between test and control counts
10092-10104


# In[200]:


((39.046813 - 45.065101)/45.065101)*100


# In[191]:


#comparing mean values between two groups with t-test


# In[201]:


stats.ttest_ind(delivery[delivery.experiment_group == 'test'].delivery_time, 
                delivery[delivery.experiment_group == 'control'].delivery_time)


# In[199]:


t_result=stats.ttest_1samp(delivery[delivery.experiment_group == 'test'].delivery_time,39.046813)
print(t_result)


# In[1]:


0.05/4


# # Mini project 5

# In[4]:


df_1 = pd.read_csv('https://stepik.org/media/attachments/lesson/385920/5_task_1.csv')
df_2 = pd.read_csv('https://stepik.org/media/attachments/lesson/385920/5_task_2.csv')


# In[5]:


df_1.head()


# In[6]:


df_2.head()


# In[11]:


#checking homogenity with Levene test
stats.levene(df_1[df_1.group == 'A'].events,
                  df_1[df_1.group == 'B'].events,
                  df_1[df_1.group == 'C'].events)


# In[12]:


#checking normality of distribution with Shapiro-Wilko test

stats.shapiro(df_1[df_1.group == 'A'].events.sample(1000))


# In[13]:


stats.shapiro(df_1[df_1.group == 'B'].events.sample(1000))


# In[14]:


stats.shapiro(df_1[df_1.group == 'B'].events.sample(1000))


# In[15]:


# testing data with f_oneway
stats.f_oneway(df_1[df_1.group == 'A'].events,
                  df_1[df_1.group == 'B'].events,
                  df_1[df_1.group == 'C'].events)


# In[20]:


#applying Tukey criterium for understanding where are statistical valuable differences between the groups
MultiComp = MultiComparison(df_1.events, df_1.group)

print(MultiComp.tukeyhsd().summary())


# In[17]:


sns.boxplot(x = 'group', y = 'events', data = df_1) #,capsize = .2)
plt.title('Impact valutation of different picture sizes')
plt.xlabel('Pic size')
plt.ylabel('Number of events')


# In[26]:


#analyzing df_2 data - checking button change
sns.distplot(df_2[df_2.group == 'control'].events, bins=50, kde=False)


# In[27]:


sns.distplot(df_2[df_2.group == 'test'].events, bins=50, kde=False)


# In[32]:


df_2.query("group == 'control' & segment == 'low'").events.describe()


# In[33]:


df_2.query("group == 'control' & segment == 'high'").events.describe()


# In[34]:


df_2.query("group == 'test' & segment == 'low'").events.describe()


# In[35]:


df_2.query("group == 'test' & segment == 'high'").events.describe()


# In[37]:


#calculating ANOVA multifactor anlaysis

formula = 'events ~ segment + group + segment:group'
model = ols(formula, df_2).fit()
aov_table = anova_lm(model, typ=2)


# In[38]:


aov_table


# In[40]:


#applying Tukey criterium for understanding where are statistical valuable differences between the groups

df_2['combination'] = df_2[['group', 'segment']].agg('-'.join, axis=1)
MultiComp = MultiComparison(df_2.events, df_2.combination)

print(MultiComp.tukeyhsd().summary())


# In[43]:


sns.boxplot(x = 'segment', y = 'events', hue = 'group', data = df_2)
plt.title('Impact of the segment and button view')
plt.xlabel('Segment')
plt.ylabel('Number of events')
plt.legend(title = 'group')


# In[48]:


sns.pointplot(x = 'segment', y = 'events', hue = 'group', data = df_2,capsize = .2)
plt.title('Impact of the segment and button view')
plt.xlabel('Segment')
plt.ylabel('Number of events')
plt.legend(title = 'group')


# # Lesson 6

# In[ ]:


x = []
np.corrcoef(x, y)


# In[ ]:


7,68 + 3,66*10 + 7,62*promotion + 0,82*8 = 150
promotion = (150 - 7.68 - 3.66*10 - 0.82*8)/7.62


# In[1]:


promotion = (150 - 7.68 - 3.66*10 - 0.82*8)/7.62


# In[2]:


promotion


# In[3]:


68.7 - 0.06*50 - 0.05*80 - 0.57*90


# # Mini project 7

# In[12]:


cars = pd.read_csv('https://stepik.org/media/attachments/lesson/387691/cars.csv')


# In[4]:


cars.head()


# In[5]:


cars.dtypes


# In[6]:


cars.isna().sum()


# In[7]:


cars.shape


# In[8]:


cars.CarName.str.split(" ")


# In[17]:


cars[['company','model']] = pd.DataFrame(cars.CarName.str.split(' ',1).tolist())


# In[19]:


cars.head()


# In[28]:


cars.company.value_counts()


# In[29]:


cars['company'] = cars['company'].replace(['toyouta','porcshce','Nissan','vokswagen','maxda','vw'],
                                         ['toyota','porsche','nissan','volkswagen','mazda','volkswagen'])


# In[30]:


cars.company.value_counts()


# In[31]:


cars_clean = cars.drop(columns=['CarName','car_ID','model'])


# In[32]:


cars_clean


# In[33]:


sns.pairplot(cars_clean, kind = 'reg')


# In[37]:


cars_reg = cars_clean[['company', 'fueltype', 'aspiration','carbody', 'drivewheel', 
                 'wheelbase', 'carlength','carwidth', 'curbweight', 'enginetype', 
                 'cylindernumber', 'enginesize', 'boreratio','horsepower','price']]


# In[38]:


cars_reg


# In[39]:


cars_reg.corr()


# In[43]:


cars_reg.dtypes


# In[44]:


cars_reg_dummy = pd.get_dummies(data=cars_reg[['company','fueltype','aspiration',
                                              'carbody','drivewheel','enginetype',
                                              'cylindernumber']], drop_first = True)


# In[45]:


cars_reg_dummy


# In[47]:


car_reg_final = pd.concat([cars_reg_dummy, cars_reg[['wheelbase','carlength','carwidth','curbweight',
                                     'enginesize','boreratio','horsepower','price']]], axis=1)


# In[48]:


car_reg_final


# In[52]:


formula = 'price ~ horsepower'
results = smf.ols(formula, car_reg_final).fit()
print(results.summary())


# In[57]:


x = car_reg_final.loc[:, ~car_reg_final.columns.isin(['price'])]


# In[58]:


Y = car_reg_final['price']


# In[60]:


X = sm.add_constant(x)  # добавить константу, чтобы был свободный член
model = sm.OLS(Y, X)  # говорим модели, что у нас ЗП, а что НП
results = model.fit()  # строим регрессионную прямую
print(results.summary())


# In[76]:


x_2 = x[[i for i in x.columns if 'company' not in i]]


# In[77]:


X = sm.add_constant(x_2)  # добавить константу, чтобы был свободный член
model = sm.OLS(Y, X)  # говорим модели, что у нас ЗП, а что НП
results = model.fit()  # строим регрессионную прямую
print(results.summary())


# In[ ]:




