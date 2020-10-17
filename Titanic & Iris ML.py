
# coding: utf-8

# In[139]:


from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

from IPython.display import HTML
style = "<style>svg{width:60% !important;height:60% !important;}</style>"
HTML(style)


# In[88]:


titanic_data = pd.read_csv('/Users/darynatrybunska/Downloads/titanic/train.csv')
titanic_data.head()


# In[89]:


titanic_data.isnull().sum() #deleting unnecessary data


# In[90]:


X = titanic_data.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1)
y = titanic_data.Survived

#as str valuea are not allowed by sklearn using get.dummies method and spliting all our str columns
X = pd.get_dummies(X)
#filling NaN values in age column as model is not fitted if any NaN
X = X.fillna({'Age': X.Age.median()})
X.head()


# In[91]:


clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X,y)


# In[92]:


graph = Source(tree.export_graphviz(clf, out_file=None, feature_names=list(X),
                                    class_names=['Died', 'Survived'], filled=True))
display(SVG(graph.pipe(format='svg')))


# In[95]:


#the previos model is overfitted, changin the tree depth + splitting original dataset into train and test parts
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[96]:


#checking how original model predicts the data
clf.score(X,y)


# In[97]:


#checking how the original model will predict train and test data
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


# In[98]:


print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


# In[99]:


#changing the model by defining tree's depth, now the test data is better predicted, 
#however the predcition of train data is lower
clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


# In[100]:


#testing what level of depth is most suitable, iterating diff numbers of depth
max_depth_values = range(1,100)
scores_data = pd.DataFrame()


# In[101]:


for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=max_depth)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    
    temp_score_data = pd.DataFrame({'max_depth': max_depth,
                                  'train_score': [train_score],
                                  'test_score': [test_score] })
    scores_data = scores_data.append(temp_score_data)
    
scores_data.head()


# In[102]:


#updating our score_data table for more convinient usage, basically unpivoting it
score_data_long = pd.melt(scores_data, id_vars=['max_depth'], 
                          value_vars=['test_score','train_score'],var_name='set_type', value_name='score')
score_data_long.head()


# In[110]:


#creating the graph to see how the both scores correlate between each others
sns.lineplot(x='max_depth', y='score', hue='set_type', data=score_data_long)


# In[104]:


from sklearn.model_selection import cross_val_score


# In[105]:


#prediction results for our train data which was splitted into 5 parts and trained a model with k-fold cross-validation(CV)
#The following procedure is followed for each of the k “folds”:
#  - A model is trained using  of the folds as training data;
#. - the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
cross_val_score(clf, X_train, y_train, cv=5)


# In[106]:


# mean accuracy using k-fold cross-validation
cross_val_score(clf, X_train, y_train, cv=5).mean()


# In[107]:


#re-creating our table to find the best number of depth wo overfitting the model by using cross-validation
scores_data_cv = pd.DataFrame()
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=max_depth)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth': max_depth,
                                  'train_score': [train_score],
                                  'test_score': [test_score],
                                  'cross_val_score': [mean_cross_val_score]})
    scores_data_cv = scores_data_cv.append(temp_score_data)
scores_data_cv.head()


# In[111]:


score_data_cv_long = pd.melt(scores_data_cv, id_vars=['max_depth'], 
                          value_vars=['test_score','train_score','cross_val_score'],
                             var_name='set_type', value_name='score')
score_data_cv_long.head()
sns.lineplot(x='max_depth', y='score', hue='set_type', data=score_data_cv_long)


# In[112]:


#defining what is the best tree depth - max value of cross_val_score = 13
score_data_cv_long.query("set_type == 'cross_val_score'").max()


# In[113]:


#recreating our model using found depth
best_clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=10)
best_clf.fit(X_train, y_train)
best_clf.score(X_test, y_test)


# In[114]:


from sklearn.model_selection import GridSearchCV


# In[115]:


clf = tree.DecisionTreeClassifier()
clf


# In[116]:


prarameters = {'max_depth': ['gini','entropy'], 'max_depth': range(1,30)}


# In[117]:


grid_search_cv_clf = GridSearchCV(clf, prarameters, cv=5)


# In[118]:


grid_search_cv_clf


# In[119]:


grid_search_cv_clf.fit(X_train, y_train)


# In[120]:


grid_search_cv_clf.best_params_


# In[121]:


best_clf = grid_search_cv_clf.best_estimator_
best_clf


# In[122]:


best_clf.score(X_test, y_test)


# In[123]:


y_pred = best_clf.predict(X_test)


# In[124]:


y_predicted_prob = best_clf.predict_proba(X_test)


# In[125]:


from sklearn.metrics import precision_score, recall_score


# In[126]:


precision_score(y_test,y_pred)


# In[127]:


recall_score(y_test,y_pred)


# In[128]:


pd.Series(y_predicted_prob[:, 1]).hist()


# In[129]:


#cut off all predicted values, and put all values > 0.8 to class 1, otherwhise class 0


# In[130]:


y_pred = np.where(y_predicted_prob[:,1] > 0.8, 1, 0)


# In[131]:


precision_score(y_test,y_pred)


# In[132]:


recall_score(y_test,y_pred)


# In[133]:


pd.Series(y_predicted_prob[:, 1]).unique()


# In[134]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.datasets import make_classification

fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[135]:


#picking up better tree parameters to avoid overtraining

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=100,min_samples_leaf=10)


# In[136]:


clf.fit(X_train, y_train)


# In[138]:


graph = Source(tree.export_graphviz(clf, out_file=None
                                   , feature_names=list(X), class_names=['Died','Survived']
                                   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[140]:


#searching for the best parameters
clf = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['gini','entropy'], 'max_depth': range(1,15),           'min_samples_split': range(2,400,20),'min_samples_leaf': range(2,50)}
grid_search_cv_clf_new = GridSearchCV(clf, parametrs, cv=5)
grid_search_cv_clf_new.fit(X_train, y_train)
grid_search_cv_clf_new.best_params_


# In[143]:


best_clf_new = grid_search_cv_clf_new.best_estimator_
best_clf_new


# In[144]:


best_clf_new.score(X_test, y_test)


# In[145]:


y_pred = best_clf_new.predict(X_test)


# In[148]:


y_predicted_prob = best_clf_new.predict_proba(X_test)
y_predicted_prob


# In[149]:


#Using RandomForest classifier
from sklearn.ensemble import RandomForestClassifier


# In[150]:


clf_rf = RandomForestClassifier()


# In[151]:


parameters = {'n_estimators':[10,20,30], 'max_depth':[2,5,7,10]}

grid_search_cv_rlf_rf = GridSearchCV(clf_rf,parameters,cv=5)


# In[154]:


grid_search_cv_rlf_rf.fit(X_train,y_train)


# In[155]:


grid_search_cv_rlf_rf.best_params_


# In[160]:


best_clf=grid_search_cv_rlf_rf.best_estimator_


# In[162]:


best_clf.score(X_test,y_test)


# In[164]:


feature_importances = best_clf.feature_importances_


# In[168]:


feature_importances_df =pd.DataFrame({'features':list(X_train),
                                     'feature_importances':feature_importances})

#same feature importnace tble can be done for decision tree as well


# In[171]:


feature_importances_df.sort_values('feature_importances', ascending=False)


# # Iris dataset

# In[43]:


train_iris = pd.read_csv('/Users/darynatrybunska/Downloads/train_iris.csv',index_col=0).sort_index()
test_iris = pd.read_csv('/Users/darynatrybunska/Downloads/test_iris.csv',index_col=0).sort_index()


# In[44]:


train_iris.head()


# In[45]:


X_train = train_iris.drop(['species'], axis=1)
y_train = train_iris.species

X_test = test_iris.drop(['species'], axis=1)
y_test = test_iris.species


# In[46]:


np.random.seed(0)
max_depth_values = range(1,100)
iris_scores = pd.DataFrame()
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=max_depth)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    
    temp_score_data = pd.DataFrame({'max_depth': max_depth,
                                  'score': [train_score],
                                  'test': [test_score]})
    iris_scores = iris_scores.append(temp_score_data)
iris_scores.head()


# In[47]:


iris_scores_long = pd.melt(iris_scores, id_vars=['max_depth'], 
                          value_vars=['score','test'],
                             var_name='set_type', value_name='score')
iris_scores_long.head()
sns.lineplot(x='max_depth', y='score', hue='set_type', data=iris_scores_long)


# # Dogs & Cats

# In[48]:


train_dc = pd.read_csv('/Users/darynatrybunska/Downloads/dogs_n_cats.csv')
X = train_dc.drop('Вид', axis=1)
y = train_dc.Вид
clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
clf.fit(X,y)


# In[49]:


X_test = pd.read_json('/Users/darynatrybunska/Downloads/dataset_209691_15.txt')
#neworder = ['Длина', 'Высота', 'Шерстист', 'Гавкает', 'Лазает по деревьям']
#X_test_new = X.reindex(columns = neworder)
np.unique(clf.predict(X_test),return_counts=True)


# # Songs

# In[50]:


songs_data = pd.read_csv('/Users/darynatrybunska/Downloads/songs.csv')


# In[51]:


songs_data.head()
X = songs_data.drop(['song','artist','lyrics'], axis=1)
X = pd.get_dummies(X)
y = songs_data.artist


# In[52]:


X.head()


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[54]:


max_depth_values = range(1,100)
songs_scores = pd.DataFrame()


# In[55]:



for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=max_depth)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth': max_depth,
                                  'train_score': [train_score],
                                  'test_score': [test_score],
                                  'cross_val_score': [mean_cross_val_score]})
    songs_scores = songs_scores.append(temp_score_data)


# In[56]:


songs_scores.head(30)


# In[57]:


best_clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
best_clf.fit(X_train,y_train)


# In[58]:


from sklearn.metrics import precision_score

predictions = best_clf.predict(X_test)


# In[61]:


train_data = pd.read_csv('/Users/darynatrybunska/Downloads/train_data_tree.csv')


# In[75]:


X_t = train_data.drop('num', axis=1)
y_t = train_data.num


# In[76]:


best_clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = best_clf.fit(X_t,y_t)
tree.plot_tree(clf,filled=True)


# In[ ]:


#IG - это разница между энтропией в корне дерева и средневзвешенной энтропией двух листьев, что из нее выходят.
IG = 0.996 - (n0*E0 + n1*E1)/N


# In[77]:


IG = 0.996 - (0.903*157 + 0.826*81)/238


# In[78]:


IG

