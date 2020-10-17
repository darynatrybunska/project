#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
sns.set(
    font_scale=1,
    style="whitegrid",
    rc={'figure.figsize':(20,7)}
        )

plt.style.use('ggplot')


# In[14]:


# Объявим функцию, которая позволит проверять гипотезы с помощью бутстрапа


# In[48]:


def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 10000, # количество бутстрэп-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_len = max([len(data_column_1), len(data_column_2)])
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            boot_len, 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            boot_len, # чтобы сохранить дисперсию, берем такой же размер выборки
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1-samples_2)) 
    pd_boot_data = pd.DataFrame(boot_data)
    
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value,
            "pd_boot_data":pd_boot_data }


# In[49]:


#Обработка данных для использования в тестах


# In[50]:


df = pd.read_csv('https://stepik.org/media/attachments/lesson/389496/hw_bootstrap.csv', sep=';')
df = df.drop(columns='Unnamed: 0')
df.value= df.value.str.replace(',','.',regex=True).astype({"value": float})

sample_1 = df.query('experimentVariant == "Treatment"').value
sample_2 = df.query('experimentVariant != "Treatment"').value


# In[51]:


#Гипотезы для проведения тестов:
    #H0 = средние между двумя выборками равны
    #H1 = средние между двумя выборками не различаются


# In[52]:


booted_data = get_bootstrap(sample_1, sample_2) # в результате хранится разница двух распределений, ДИ и pvalue


# In[53]:


booted_data["p_value"] # альфа


# In[54]:


booted_data["quants"] # ДИ


# In[55]:


stats.mannwhitneyu(sample_1, sample_2)


# In[56]:


ax = sns.boxplot(x="experimentVariant", y="value", data=df)


# In[57]:


sns.distplot(sample_1,kde=False, bins=20)
sns.distplot(sample_2,kde=False, bins=20)


# ## Выводы
# 
# Результаты проведенных тестов показали, что, согласно полученному p value, обнаружены статистически важные различия между двумя выборками.
# Однако, бутстрап дал более низкое р-value (результаты более статистичиски значимые), что объясняется разницей в осуществлении методов:
# - при тесте использовалась выборка разных размерностей (данных из группы "Treatment" в 10 раз меньше, чем контрольной группе); 
# - при ресемплировании данных в бутстрапе использовался размер максимальной группы для сохранения дисперсии, что поспособствовало более однородному сравнению
# - тест Mann-Whitney использует подсчет рангов в двух выборках; поскольку сравниваемые выборки сущестевнно различаются по своей размерности, поэтому сумма рангов для "Treatment" была очевидно меньше, чем для второй группы, что могло повлиять на качество результатов.
# 
# К тому же, если бустрап отображает разницу в сравнениях средних значений между двумя выборками (так как используемая статистика = np.mean), то тест Mann-Whitney скорее оценивает одинаковы ли распределения двух оцениваемых выборок или нет.
# На мой взгляд, этот фактор также отражает разность в полученых результатах.

# In[ ]:




