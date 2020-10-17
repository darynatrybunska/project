#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ## Problem No.1

# Retention – один из самых важных показателей в компании. 
# Ваша задача – написать функцию, которая будет считать retention игроков (по дням от даты регистрации игрока). 
# 

# In[41]:


def retention_per_day(registrations,authentication):
    reg = pd.read_csv(registrations, sep=';')
    auth = pd.read_csv(authentication, sep=';')
    reg['reg_ts_upd'] = pd.to_datetime(reg.reg_ts, unit='s')
    reg['reg_ts_day'] = reg['reg_ts_upd'].dt.date
    auth['auth_ts_upd'] = pd.to_datetime(auth.auth_ts, unit='s')
    auth['auth_ts_day'] = auth['auth_ts_upd'].dt.date
    df = reg.merge(auth,on = ['uid'], how='inner')
    df['return'] = (df.auth_ts_day - df.reg_ts_day).dt.days+1
    group = df.groupby(['reg_ts_day', 'return'])
    cohort_data = group['uid'].size()
    cohort_data = cohort_data.reset_index()
    cohort_counts = cohort_data.pivot(index='reg_ts_day', columns='return', values='uid')
    base = cohort_counts[1]
    retention = cohort_counts.divide(base, axis=0).round(3).multiply(100,axis=0).round(2).reset_index()
    retention.to_csv('retention_per_day.csv')
    print('\nFile "retention_per_day.csv" have been saved to your local folder.\nThe example of results:\n\n', 
          retention.iloc[:, 0:11].tail(10))


# In[42]:


retention_per_day('shared/problem1-reg_data.csv','shared/problem1-auth_data.csv')


# ## Problem No.2

# Имеются результаты A/B теста, в котором двум группам пользователей предлагались различные наборы акционных предложений. 
# Известно, что ARPU в тестовой группе выше на 5%, чем в контрольной. 
# При этом в контрольной группе 1928 игроков из 202103 оказались платящими, а в тестовой – 1805 из 202667.
# 
# Какой набор предложений можно считать лучшим? Какие метрики стоит проанализировать для принятия правильного решения и как?

# In[44]:


ab = pd.read_csv('https://stepik.org/media/attachments/lesson/409318/problem2.csv',sep=';')


# In[65]:


a_non_paying_users = ab.query('testgroup == "a" and revenue == 0').count().user_id
a_clients = ab.query('testgroup == "a" and revenue != 0').count().user_id
a_arppu = ab.query('testgroup == "a" and revenue != 0').mean().revenue

b_non_paying_users = ab.query('testgroup == "b" and revenue == 0').count().user_id
b_clients = ab.query('testgroup == "b" and revenue != 0').count().user_id    
b_arppu = ab.query('testgroup == "b" and revenue != 0').mean().revenue


# In[72]:


a_users = a_non_paying_users + a_clients
a_conversion = round(a_clients/a_users *100,2)
a_arpu = round(a_arppu*a_conversion /100,2)

print('Conversion for group A:',a_conversion)
print('ARPU for group A: ',a_arpu)


# In[78]:


b_users = a_non_paying_users + a_clients
b_conversion = round(b_clients/b_users *100,2)
b_arpu = round(b_arppu*b_conversion /100,2)

print('Conversion for group B:',b_conversion)
print('ARPU for group B: ',b_arpu)


# In[85]:


conversion_change = round(((b_conversion-a_conversion)/a_conversion)*100,2)
arpu_change = round(((b_arpu-a_arpu)/a_arpu)*100,2)
print('Converion is changing by: ', conversion_change,'%')
print('ARPU is changing by: ', arpu_change,'%')


# Согласно полученным результатам, проведенное тестирование приводит как к положительному изменению ARPU, так и к снижению конверсии:
# - наборы акционных предложений, предложенные тестовой группе В, положительно влияют на определенную целевую аудиторию, но не на всех пользователей, так как увеличение ARPPU при уменьшении общего количества платящих пользователей объясняется либо увеличившимся средним чеком, либо - большим числом повторных покупок.
# - Снижение конверсии на 6 процентов может являтся более глобальным показателем, который негативно сказывается на возможность внедрения тестируемых изменений.
# 
# На мой взгляд, больший показатель конверсии является более значимым показателем как для роста прибыли компании, так и других показателей unit-экономики. Однако, если целью компании является сосредоточение на определенном сегменте пользователей - результаты тестирования можно оценивать как положительными и использовать их в качетсве аргументации для внедрения тестируемых изменений в продакшн.

# ## Problem No.3

# В игре Plants & Gardens каждый месяц проводятся тематические события, ограниченные по времени. В них игроки могут получить уникальные предметы для сада и персонажей, дополнительные монеты или бонусы. Для получения награды требуется пройти ряд уровней за определенное время. 
# 
#     1.A. С помощью каких метрик можно оценить результаты последнего прошедшего события?
#     1.B. Предположим, в другом событии мы усложнили механику событий так, что при каждой неудачной попытке выполнения уровня игрок будет откатываться на несколько уровней назад. Изменится ли набор метрик оценки результата? Если да, то как?

# ### 1.A.

# Следующие метрики могут быть полезны для оценивания результатов прошедшего события:
# - среднее количество времени на прохождения каждого из уровней.
# - определение начала и окончания действия/уровня/игры.
# - подсчет выполненных действий.
# - аchievement count: подсчет достижений, полученных игроком на протяжении игры.
# - source: отражает количество монет/бонусов, которые игрок заработал в течениии игры.
# - sink: отражает в этапы, на которых пользователю необходимо потратить монеты/бонусы, чтобы двигаться вперед или конкурировать с другими игроками.
# - flow: отражает баланс монет/бонусов, которые игрок заработал и потратил в течении игры.

# ### 2.A.

# Помимо метрик, указанных в пункте 1.А., имеет смысл также считать следующие метрики, чтобы отслеживать, какой уровень окажется слишком сложным/легким для игроков:
# - start: это показатель того, сколько раз в среднем игрок переходил на новый уровень.
# - fail: отражает, сколько раз игрок запускал уровень, но не смог его завершить.
# - complete: подсчитыввает, сколько раз пользователь успешно прошел уровень.

# In[ ]:


300 gr = 35500 usd
35500/300

