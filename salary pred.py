import pandas as pd
import pickle
import matplotlib.pyplot as plt

df=pd.read_csv('survey_results_public.csv')

df.head(5)

df.columns

df=df[['Country','EdLevel','YearsCodePro','Employment','ConvertedCompYearly']]
df=df.rename({'ConvertedCompYearly':'Salary'},axis=1)
df.head()

df=df[df['Salary'].notnull()]
df.head()

df=df.dropna()
df.isnull().sum()

df=df[df['Employment']=='Employed full-time']
df=df.drop('Employment',axis=1)

df.info()

df['Country'].value_counts()

def shorten_categories(categories,cutoff):
    categorical_map={}
    for i in range(len(categories)):
        if categories.values[i]>=cutoff:
            categorical_map[categories.index[i]]=categories.index[i]
        else:
            categorical_map[categories.index[i]]='other'
    return categorical_map

country_map=shorten_categories(df.Country.value_counts(),400)
df['Country']=df['Country'].map(country_map)
df.Country.value_counts()

fig,ax=plt.subplots(1,1,figsize=(12,7))
df.boxplot('Salary','Country',ax=ax)
plt.suptitle('Salary v country')
plt.title(' ')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

df=df[df['Salary']<=400000]
df=df[df['Country']!='Other']

fig,ax=plt.subplots(1,1,figsize=(12,7))
df.boxplot('Salary','Country',ax=ax)
plt.suptitle('Salary v country')
plt.title(' ')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

df['YearsCodePro'].unique()

def clean_experience(x):
    if x =='More than 50 years':
        return 50
    if x =='Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro']=df['YearsCodePro'].apply(clean_experience)

df['EdLevel'].unique()

def clean_education(x):
    if 'Bachelor' in x:
        return 'Bachelor'
    if 'Master' in x:
        return 'Master'
    if 'Professional degree' in x or 'other doctoral' in x:
        return 'Post degree'
    return 'Less than a Bachelors'
    
df['EdLevel']=df['EdLevel'].apply(clean_education)

df['EdLevel'].unique()

from sklearn.preprocessing import LabelEncoder
le_education=LabelEncoder()
df['EdLevel']=le_education.fit_transform(df['EdLevel'])
df['EdLevel'].unique()

le_country=LabelEncoder()
df['Country']=le_country.fit_transform(df['Country'])
df['Country'].unique()

x=df.drop('Salary',axis=1)
y=df['Salary']

from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)

y_pred=linear_reg.predict(x)

from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
error=np.sqrt(mean_squared_error(y,y_pred))

from sklearn.tree import DecisionTreeRegressor
dec_tree_reg=DecisionTreeRegressor(random_state=5)
dec_tree_reg.fit(x,y)

y_pred=dec_tree_reg.predict(x)
error=np.sqrt(mean_squared_error(y,y_pred))
error

from sklearn.ensemble import RandomForestRegressor
random_forest_reg=RandomForestRegressor(random_state=0)
random_forest_reg.fit(x,y)
y_pred=random_forest_reg.predict(x)
error=np.sqrt(mean_squared_error(y,y_pred))
error

from sklearn.model_selection import GridSearchCV

max_depth=[None,2,4,6,7,10,12]
parameters={'max_depth':max_depth}

regressor=DecisionTreeRegressor(random_state=0)
gs=GridSearchCV(regressor,parameters,scoring='neg_mean_squared_error')
gs.fit(x,y)

regressor=gs.best_estimator_
regressor.fit(x,y)
y_pred=regressor.predict(x)
error=np.sqrt(mean_squared_error(y,y_pred))
error

data={'model':regressor,'le_country':le_country,'le_education':le_education}
with open('saved_steps.pkl','wb') as file:
    pickle.dump(data,file)
    
    

