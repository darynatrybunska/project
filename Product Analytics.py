#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


df = pd.read_csv('https://stepik.org/media/attachments/lesson/394402/karpov_courses_%D0%94%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8_%D0%92%D1%8B%D0%B3%D1%80%D1%83%D0%B7%D0%BA%D0%B0_%D1%81%D0%B4%D0%B5%D0%BB%D0%BE%D0%BA_%D0%B7%D0%B0_%D0%B0%D0%B2%D0%B3%D1%83%D1%81%D1%82_upd.csv')


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df_cleaned = df.drop(df.columns[[5,6]], axis=1) 


# In[47]:


df_cleaned.head()


# In[9]:


df_cleaned.columns = df_cleaned.columns.str.replace('ga:','')


# In[48]:


df_cleaned.head()


# In[19]:


df_cleaned["userID"] = df_cleaned["userID"].str.replace(',','')


# In[32]:


df_cleaned["transaction_id"] = df_cleaned["transaction_id"].str.replace('-','0')


# In[33]:


df_cleaned[["clientid", "userID","transaction_id","revenue"]] = df_cleaned[["clientid", "userID","transaction_id","revenue"]].apply(pd.to_numeric)


# In[35]:


df_cleaned["date"] = pd.to_datetime(df_cleaned["date"])


# In[36]:


df_cleaned.dtypes


# In[43]:


df_cleaned[df_cleaned['transaction_id'].isna()]


# In[45]:


df_cleaned['transaction_id'] = df_cleaned['transaction_id'].fillna(0)


# In[46]:


df_cleaned.isna().sum()


# In[53]:


#Сколько всего имеется уникальных пользователей?

141000


# In[56]:


#Сколько всего имеется клиентов?
df_cleaned[df_cleaned['revenue'] !=0].clientid.nunique()


# In[86]:


#Далее посчитайте конверсию. В качестве ответа укажите значение в процентах, округленное до сотых.
conversion = (685/141000)*100


# In[87]:


conversion


# In[71]:


#Сколько рублей составляет средний чек? Ответ округлите до целого.
avarage_bill = round(df_cleaned.revenue.sum()/df_cleaned[df_cleaned['revenue'] !=0].clientid.count())


# In[72]:


#Теперь посчитайте повторные покупки. Результат округлите до двух знаков после точки.
repeat_purchases = df_cleaned[df_cleaned['revenue'] !=0].clientid.count()/df_cleaned[df_cleaned['revenue'] !=0].clientid.nunique()


# In[73]:


#Чему равен средний доход на платящего пользователя (ARPPU)? Ответ округлите до целого.
ARPPU = avarage_bill*repeat_purchases


# In[74]:


ARPPU


# In[88]:


#Посчитайте средний доход с пользователя (ARPU). Ответ округлите до целого.
ARPU = (conversion*ARPPU)/100


# In[89]:


ARPU


# # Mini project 2

# In[ ]:


с выручкой – 20 млн. рублей, 
прибыль – 6 млн. 
расходы на рекламу – 2 млн. 
заказы – 4 000
покупатели – 3 300
пользователей на сайте – 600 000


# In[46]:


conversion = 3300/600000


# In[47]:


conversion*100


# In[7]:


av_bill = 20000000/4000


# In[8]:


av_bill


# In[9]:


rp = 4000/3300


# In[10]:


rp


# In[11]:


ARPPU = av_bill*rp


# In[12]:


ARPPU


# In[15]:


ARPU = (ARPPU*conversion)/100


# In[16]:


ARPU


# In[17]:


CPaq = 2000000/600000


# In[18]:


CPaq


# In[32]:


#users∗(conv∗AOV∗margin∗RP−CPAcq)=CM
users = 600000
CM = 6000000
margin = (CM/users+CPaq)/(conversion*av_bill*rp)


# In[33]:


margin


# In[41]:


conversion_1 = 0.01
CM_1 = round(users*(conversion_1*av_bill*margin*rp - CPaq))


# In[42]:


CM_1


# In[48]:


users_1 = CM_1/(conversion*av_bill*margin*rp - CPaq)


# In[50]:


round(users_1) - users


# In[51]:


av_bill_1 = (CM_1/users+CPaq)/(conversion*margin*rp)


# In[54]:


round(av_bill_1)


# In[55]:


margin_1 = (CM_1/users+CPaq)/(conversion*av_bill*rp)


# In[58]:


round(margin_1*100)


# In[59]:


rp_1 = (CM_1/users+CPaq)/(conversion*av_bill*margin)


# In[62]:


round(rp_1,3)


# In[ ]:


В приложении зарегистрировалось 400 пользователей – это наша когорта. 
В первую неделю к нам вернулось 100 человек, 
во вторую – 80 (из них 60 человек были на первой неделе), 
в третью – 60 (10 из них не были в неделю 1 и 2), 
в четвертую – 50 (10 из них не были в недели 1, 2, 3). 

Посчитайте Retention (календарно) первой, второй, третьей недели
Посчитайте Rolling Retention первой, второй, третьей недели 


# In[70]:


classic_week_1 = 100/400
classic_week_1


# In[71]:


classic_week_2 = 80/400
classic_week_2


# In[72]:


classic_week_3 = 60/400
classic_week_3


# In[74]:


classic_week_4 = 50/400
classic_week_4


# In[81]:


rolling_week_1 = (100+20+10+10)/400
rolling_week_1


# In[82]:


rolling_week_2 = (80+10+10)/400
rolling_week_2


# In[83]:


rolling_week_3 = (60+10)/400
rolling_week_3


# # Mini project 3

# In[2]:


unit = pd.DataFrame(columns=['description','Change','users', 'conversion', 'customers','ARPPU','AOV','margin','total_purchases',
                             'RP','CA','CPAcq','CAC','ARPU','revenue'])
#Conversion = clients/users
#clients = paying users
#ARPPU = avarage revenue per paying user
#AOV = avarage bill
#margin = Маржа = Выручка — Себестоимость / Выручка х 100 margin = (CM/users+CPaq)/(conversion*av_bill*rp)
#RP = repeat purchases
#CA = customer acquisition (marketing)
#CPAcq = cost per user acquisition 
#ARPU = avarage revenue per user
#CM = contribution margin (прибыль) CM_1 = round(users*(conversion_1*av_bill*margin*rp - CPaq))
#CAC = cost per custimer acquisition


# ### Default values

# In[3]:


unit.loc[0,'description'] = 'default values'
unit.loc[0,'revenue'] = 32000000
unit.loc[0,'CA'] = 400000
unit.loc[0,'users'] = 141000
unit.loc[0,'customers'] = 830
unit.loc[0,'RP'] = 1.2
unit.loc[0,'margin'] = 0.3
unit['conversion'] = round(pd.to_numeric((unit.customers/unit.users)*100),2)
unit['total_purchases'] = round(pd.to_numeric(unit.RP*unit.customers))
unit['AOV'] = round(pd.to_numeric(unit.revenue/unit.total_purchases))
unit['ARPPU'] = round(pd.to_numeric(unit.AOV*unit.RP))
unit['ARPU'] = round(pd.to_numeric(unit.ARPPU*unit.conversion/100),2)
unit['CPAcq'] = round(pd.to_numeric(unit.CA/unit.users),2)
unit['CM_calc'] = round(pd.to_numeric(unit.users*(unit.conversion/100*unit.AOV*unit.margin*unit.RP - unit.CPAcq)))


# In[4]:


unit


# ### Conversion = 1% (↑+0.41%)

# In[5]:


unit.loc[1,'description'] = 'Conversion + 0.41'
unit.loc[1,'CA'] = 400000
unit.loc[1,'users'] = 141000
unit.loc[1,'RP'] = 1.2
unit.loc[1,'margin'] = 0.3
unit.loc[1,'conversion'] = 1.0
unit.loc[1,'customers'] = round(pd.to_numeric(unit.users[1]*unit.conversion[1]/100))
unit.loc[1,'AOV'] = 32129.0
unit['ARPPU'] = round(pd.to_numeric(unit.AOV*unit.RP))
unit.loc[1,'revenue'] = round(pd.to_numeric(float(unit.customers[1]*unit.ARPPU[1])),0)
unit.loc[1,'total_purchases'] = round(pd.to_numeric(unit.RP[1]*unit.customers[1]))
unit['ARPU'] = round(pd.to_numeric(unit.ARPPU*unit.conversion/100),2)
unit['CPAcq'] = round(pd.to_numeric(unit.CA/unit.users),2)
unit['CM_calc'] = round(pd.to_numeric(unit.users*(unit.conversion/100*unit.AOV*unit.margin*unit.RP - unit.CPAcq)))
unit.loc[1,'revenue'] = round(pd.to_numeric(unit.customers[1]*unit.ARPPU[1]))
unit.loc[1,'Change'] = 'CM ↑+ ' + str(unit.CM_calc[1]-unit.CM_calc[0])


# In[6]:


unit


# ### How to get CM ↑ + 6,69M by increasing no. of users?

# In[7]:


unit.loc[2,'description'] = 'Increase of users for CM ↑+6.69M'
unit.loc[2,'conversion'] = 0.59
unit.loc[2,'AOV'] = 32129.0
unit.loc[2,'margin'] = 0.3
unit.loc[2,'RP'] = 1.2
unit.loc[2,'CA'] = 400000
unit.loc[2,'revenue'] = float(5.43626e+07)
unit.loc[2,'CM_calc'] = 15908240.0
unit.loc[2,'CPAcq'] = 2.84
unit.loc[2,'ARPPU'] = round(pd.to_numeric(unit.AOV[2]*unit.RP[2]))
unit.loc[2,'ARPU'] = round(pd.to_numeric(unit.ARPPU[2]*unit.conversion[2]/100),2)
unit.loc[2,'users'] = round(unit.CM_calc[2]/(unit.conversion[2]/100*unit.AOV[2]*unit.margin[2]*unit.RP[2]-unit.CPAcq[2]))
unit.loc[2,'customers'] = unit.users[2]/(unit.users[0]/unit.customers[0])
unit.loc[2,'total_purchases'] = round(pd.to_numeric(unit.RP[2]*unit.customers[2]))
unit.loc[2,'Change'] = 'Users ↑+ ' + str(unit.users[2]-unit.users[1])


# In[8]:


unit


# ### How to get CM ↑ + 6,69M by increasing average bill?

# In[9]:


unit.loc[3,'description'] = 'Increase of AOV for CM ↑+6.69M'
unit.loc[3,'users'] = 141000
unit.loc[3,'conversion'] = 0.59
unit.loc[3,'customers'] = 830
unit.loc[3,'margin'] = 0.3
unit.loc[3,'RP'] = 1.2
unit.loc[3,'total_purchases'] = round(pd.to_numeric(unit.RP[3]*unit.customers[3]))
unit.loc[3,'CA'] = 400000
unit.loc[3,'revenue'] = 54362600
unit.loc[3,'CM_calc'] = 15908240.0
unit.loc[3,'CPAcq'] = 2.84

unit.loc[3,'AOV'] = round((unit.CM_calc[3]/unit.users[3]+unit.CPAcq[3])/(unit.conversion[3]/100*unit.margin[3]*unit.RP[3]))
unit.loc[3,'ARPPU'] = round(pd.to_numeric(unit.AOV[3]*unit.RP[3]))
unit.loc[3,'ARPU'] = round(pd.to_numeric(unit.ARPPU[3]*unit.conversion[3]/100),2)

unit.loc[3,'Change'] = 'AOV ↑+ ' + str(unit.AOV[3]-unit.AOV[2])


# In[10]:


unit


# ### How to get CM ↑ + 6,69M by increasing margin?

# In[11]:


unit.loc[4,'description'] = 'Increase of margin for CM ↑+6.69M'
unit.loc[4,'users'] = 141000
unit.loc[4,'conversion'] = 0.59
unit.loc[4,'customers'] = 830
unit.loc[4,'RP'] = 1.2
unit.loc[4,'total_purchases'] = round(pd.to_numeric(unit.RP[4]*unit.customers[4]))
unit.loc[4,'CA'] = 400000
unit.loc[4,'revenue'] = 32000000
unit.loc[4,'CM_calc'] = 15908240.0
unit.loc[4,'CPAcq'] = 2.84

unit.loc[4,'AOV'] = 32129.0
unit.loc[4,'ARPPU'] = round(pd.to_numeric(unit.AOV[4]*unit.RP[4]))
unit.loc[4,'ARPU'] = round(pd.to_numeric(unit.ARPPU[4]*unit.conversion[4]/100),2)

unit.loc[4,'margin'] = round((unit.CM_calc[4]/unit.users[4]+unit.CPAcq[4])/(unit.conversion[4]/100*unit.AOV[4]*unit.RP[4]),2)
unit.loc[4,'Change'] = 'margin ↑+ ' + str(round(unit.margin[4]-unit.margin[3],2))


# In[12]:


unit


# ### How to get CM ↑ + 6,69M by increasing repeat purchases?

# In[13]:


unit.loc[5,'description'] = 'Increase of RP for CM ↑+6.69M'
unit.loc[5,'users'] = 141000
unit.loc[5,'conversion'] = 0.59
unit.loc[5,'customers'] = 830
unit.loc[5,'margin'] = 0.3
unit.loc[5,'CA'] = 400000
unit.loc[5,'revenue'] = 54362600
unit.loc[5,'CM_calc'] = 15908240.0
unit.loc[5,'CPAcq'] = 2.84
unit.loc[5,'AOV'] = 32129.0

unit.loc[5,'RP'] = round((unit.CM_calc[5]/unit.users[5]+unit.CPAcq[5])/(unit.conversion[5]/100*unit.AOV[5]*unit.margin[5]),2)
unit.loc[5,'total_purchases'] = round(pd.to_numeric(unit.RP[5]*unit.customers[5]))
unit.loc[5,'ARPPU'] = round(pd.to_numeric(unit.AOV[5]*unit.RP[5]))
unit.loc[5,'ARPU'] = round(pd.to_numeric(unit.ARPPU[5]*unit.conversion[5]/100),2)

unit.loc[5,'Change'] = 'RP ↑+ ' + str(round(unit.RP[5]-unit.RP[4],2))


# In[14]:


unit


# ### How to get CM ↑ + 6,69M by decreasing costs for customer acquistion?

# In[15]:


unit.loc[6,'description'] = 'Decrease of CPAcq for CM ↑+6.69M'
unit.loc[6,'users'] = 141000
unit.loc[6,'conversion'] = 0.59
unit.loc[6,'customers'] = 830
unit.loc[6,'margin'] = 0.3
unit.loc[6,'revenue'] = 54362600
unit.loc[6,'CM_calc'] = 15908240.0
unit.loc[6,'AOV'] = 32129.0
unit.loc[6,'RP'] = 1.2
unit.loc[6,'total_purchases'] = round(pd.to_numeric(unit.RP[6]*unit.customers[6]))
unit.loc[6,'ARPPU'] = round(pd.to_numeric(unit.AOV[6]*unit.RP[6]))
unit.loc[6,'ARPU'] = round(pd.to_numeric(unit.ARPPU[6]*unit.conversion[6]/100),2)

unit.loc[6,'CPAcq'] = round((unit.conversion[6]/100*unit.AOV[6]*unit.margin[6]*unit.RP[6])-(unit.CM_calc[6]/unit.users[6]),2)
unit.loc[6,'CA'] = unit.CPAcq[6]*unit.users[6]
unit.loc[6,'Change'] = 'CPAcq ↑- ' + str(round(unit.CPAcq[5]-unit.CPAcq[6],2))


# In[16]:


unit


# ### Contribution margin changes based on the experiments effects

# In[27]:


conv_changes = pd.DataFrame(columns=['description','Change','users', 'conversion', 'customers','ARPPU','AOV','margin','total_purchases',
                             'RP','CA','CPAcq','CAC','ARPU','revenue'])


# In[28]:


conv_changes.loc[0,'description'] = 'default values'
conv_changes.loc[0,'revenue'] = 32000000
conv_changes.loc[0,'CA'] = 400000
conv_changes.loc[0,'users'] = 141000
conv_changes.loc[0,'customers'] = 830
conv_changes.loc[0,'RP'] = 1.2
conv_changes.loc[0,'margin'] = 0.3
conv_changes['conversion'] = round(pd.to_numeric((conv_changes.customers/conv_changes.users)*100),2)
conv_changes['total_purchases'] = round(pd.to_numeric(conv_changes.RP*conv_changes.customers))
conv_changes['AOV'] = round(pd.to_numeric(conv_changes.revenue/conv_changes.total_purchases))
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))


# In[29]:


conv_changes.loc[1,'description'] = 'Conversion = 1.21'
conv_changes.loc[1,'CA'] = 400000
conv_changes.loc[1,'users'] = 141000
conv_changes.loc[1,'RP'] = 1.2
conv_changes.loc[1,'margin'] = 0.3
conv_changes.loc[1,'conversion'] = 1.21
conv_changes.loc[1,'customers'] = round(pd.to_numeric(conv_changes.users[1]*conv_changes.conversion[1]/100))
conv_changes.loc[1,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[1,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[1]*conv_changes.ARPPU[1])),0)
conv_changes.loc[1,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[1]*conv_changes.customers[1]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[1,'revenue'] = round(pd.to_numeric(conv_changes.customers[1]*conv_changes.ARPPU[1]))
conv_changes.loc[1,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[1]-conv_changes.CM_calc[0])


# In[30]:


conv_changes.loc[2,'description'] = 'Conversion = 1.1'
conv_changes.loc[2,'CA'] = 400000
conv_changes.loc[2,'users'] = 141000
conv_changes.loc[2,'RP'] = 1.2
conv_changes.loc[2,'margin'] = 0.3
conv_changes.loc[2,'conversion'] = 1.1
conv_changes.loc[2,'customers'] = round(pd.to_numeric(conv_changes.users[2]*conv_changes.conversion[2]/100))
conv_changes.loc[2,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[2,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[2]*conv_changes.ARPPU[2])),0)
conv_changes.loc[2,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[2]*conv_changes.customers[2]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[2,'revenue'] = round(pd.to_numeric(conv_changes.customers[2]*conv_changes.ARPPU[2]))
conv_changes.loc[2,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[2]-conv_changes.CM_calc[0])


# In[31]:


conv_changes.loc[3,'description'] = 'Conversion = 1.03'
conv_changes.loc[3,'CA'] = 400000
conv_changes.loc[3,'users'] = 141000
conv_changes.loc[3,'RP'] = 1.2
conv_changes.loc[3,'margin'] = 0.3
conv_changes.loc[3,'conversion'] = 1.03
conv_changes.loc[3,'customers'] = round(pd.to_numeric(conv_changes.users[3]*conv_changes.conversion[3]/100))
conv_changes.loc[3,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[3,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[3]*conv_changes.ARPPU[3])),0)
conv_changes.loc[3,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[3]*conv_changes.customers[3]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[3,'revenue'] = round(pd.to_numeric(conv_changes.customers[3]*conv_changes.ARPPU[3]))
conv_changes.loc[3,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[3]-conv_changes.CM_calc[0])


# In[33]:


conv_changes.loc[4,'description'] = 'Conversion = 0.88'
conv_changes.loc[4,'CA'] = 400000
conv_changes.loc[4,'users'] = 141000
conv_changes.loc[4,'RP'] = 1.2
conv_changes.loc[4,'margin'] = 0.3
conv_changes.loc[4,'conversion'] = 0.88
conv_changes.loc[4,'customers'] = round(pd.to_numeric(conv_changes.users[4]*conv_changes.conversion[4]/100))
conv_changes.loc[4,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[4,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[4]*conv_changes.ARPPU[4])),0)
conv_changes.loc[4,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[4]*conv_changes.customers[4]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[4,'revenue'] = round(pd.to_numeric(conv_changes.customers[4]*conv_changes.ARPPU[4]))
conv_changes.loc[4,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[4]-conv_changes.CM_calc[0])


# In[35]:


conv_changes.loc[5,'description'] = 'Conversion = 0.83'
conv_changes.loc[5,'CA'] = 400000
conv_changes.loc[5,'users'] = 141000
conv_changes.loc[5,'RP'] = 1.2
conv_changes.loc[5,'margin'] = 0.3
conv_changes.loc[5,'conversion'] = 0.83
conv_changes.loc[5,'customers'] = round(pd.to_numeric(conv_changes.users[5]*conv_changes.conversion[5]/100))
conv_changes.loc[5,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[5,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[5]*conv_changes.ARPPU[5])),0)
conv_changes.loc[5,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[5]*conv_changes.customers[5]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[5,'revenue'] = round(pd.to_numeric(conv_changes.customers[5]*conv_changes.ARPPU[5]))
conv_changes.loc[5,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[5]-conv_changes.CM_calc[0])


# In[37]:


conv_changes.loc[6,'description'] = 'Conversion = 0.77'
conv_changes.loc[6,'CA'] = 400000
conv_changes.loc[6,'users'] = 141000
conv_changes.loc[6,'RP'] = 1.2
conv_changes.loc[6,'margin'] = 0.3
conv_changes.loc[6,'conversion'] = 0.77
conv_changes.loc[6,'customers'] = round(pd.to_numeric(conv_changes.users[6]*conv_changes.conversion[6]/100))
conv_changes.loc[6,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[6,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[6]*conv_changes.ARPPU[6])),0)
conv_changes.loc[6,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[6]*conv_changes.customers[6]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[6,'revenue'] = round(pd.to_numeric(conv_changes.customers[6]*conv_changes.ARPPU[6]))
conv_changes.loc[6,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[6]-conv_changes.CM_calc[0])


# In[39]:


conv_changes.loc[7,'description'] = 'Conversion = 0.72'
conv_changes.loc[7,'CA'] = 400000
conv_changes.loc[7,'users'] = 141000
conv_changes.loc[7,'RP'] = 1.2
conv_changes.loc[7,'margin'] = 0.3
conv_changes.loc[7,'conversion'] = 0.72
conv_changes.loc[7,'customers'] = round(pd.to_numeric(conv_changes.users[7]*conv_changes.conversion[7]/100))
conv_changes.loc[7,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[7,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[7]*conv_changes.ARPPU[7])),0)
conv_changes.loc[7,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[7]*conv_changes.customers[7]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[7,'revenue'] = round(pd.to_numeric(conv_changes.customers[7]*conv_changes.ARPPU[7]))
conv_changes.loc[7,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[7]-conv_changes.CM_calc[0])


# In[41]:


conv_changes.loc[8,'description'] = 'Conversion = 0.63'
conv_changes.loc[8,'CA'] = 400000
conv_changes.loc[8,'users'] = 141000
conv_changes.loc[8,'RP'] = 1.2
conv_changes.loc[8,'margin'] = 0.3
conv_changes.loc[8,'conversion'] = 0.63
conv_changes.loc[8,'customers'] = round(pd.to_numeric(conv_changes.users[8]*conv_changes.conversion[8]/100))
conv_changes.loc[8,'AOV'] = 32129.0
conv_changes['ARPPU'] = round(pd.to_numeric(conv_changes.AOV*conv_changes.RP))
conv_changes.loc[8,'revenue'] = round(pd.to_numeric(float(conv_changes.customers[8]*conv_changes.ARPPU[8])),0)
conv_changes.loc[8,'total_purchases'] = round(pd.to_numeric(conv_changes.RP[8]*conv_changes.customers[8]))
conv_changes['ARPU'] = round(pd.to_numeric(conv_changes.ARPPU*conv_changes.conversion/100),2)
conv_changes['CPAcq'] = round(pd.to_numeric(conv_changes.CA/conv_changes.users),2)
conv_changes['CM_calc'] = round(pd.to_numeric(conv_changes.users*(conv_changes.conversion/100*conv_changes.AOV*conv_changes.margin*conv_changes.RP - conv_changes.CPAcq)))
conv_changes.loc[8,'revenue'] = round(pd.to_numeric(conv_changes.customers[8]*conv_changes.ARPPU[8]))
conv_changes.loc[8,'Change'] = 'CM ↑+ ' + str(conv_changes.CM_calc[8]-conv_changes.CM_calc[0])


# In[42]:


conv_changes


# In[52]:


conv_changes.to_csv('PA_Mini_project_3_margin_changes.csv') 


# In[51]:


unit.to_csv('PA_Mini_project_3_unit_economics.csv')


# ## Final results

# Согласно полученным результатам, "узким местом" продукта является количество повторяющихся покупок:
# - при увеличении повторяющихся покупок с 1.2 до 2 онлайн-магазин сможет повысить показатели прибыли на 6.69М рублей, тем самым получить аналогичный эффект что и при увеличении конверсии с 0.59% до 1% (смю таблицу ниже).

# In[150]:


unit


# In[ ]:




