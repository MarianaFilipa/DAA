import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, plot_confusion_matrix,ConfusionMatrixDisplay

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import classification_report, plot_confusion_matrix



df_train = pd.read_csv("sbstpdaa2223/training_data.csv")
df = pd.read_csv("sbstpdaa2223/test_data.csv") # TEST SET


# -------------------- TRAINING SET-----------------------

df_train.drop(['affected_roads'],axis=1, inplace = True)


# O atributo cidade não acrescenta nada, uma vez que tem todos os mesmo valor, e portanto não vai afetar o modelo
df_train.drop('city_name', axis = 1, inplace = True)

# O atributo avg_precipitation também é unico por isso não acrescenta muito 
#print(df_no_city['avg_precipitation'].nunique())
df_train.drop('avg_precipitation', axis = 1, inplace = True)

labels = df_train['avg_rain'].astype('category').cat.categories.tolist()
replace_map_rain = {'avg_rain':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_rain)
df_train.replace(replace_map_rain, inplace = True)

labels = df_train['incidents'].astype('category').cat.categories.tolist()
replace_map_incidents = {'incidents':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_incidents)
df_train.replace(replace_map_incidents, inplace = True)

labels = df_train['magnitude_of_delay'].astype('category').cat.categories.tolist()
replace_map_delay = {'magnitude_of_delay':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_delay)
df_train.replace(replace_map_delay, inplace = True)

labels = df_train['luminosity'].astype('category').cat.categories.tolist()
replace_map_luminosity = {'luminosity':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_luminosity)
df_train.replace(replace_map_luminosity, inplace = True)

df_train['record_date'] = pd.to_datetime(df['record_date'], format = "%Y-%m-%d %H:%M", errors='coerce')

assert df_train['record_date'].isnull().sum() == 0, 'missing record date'

#Não tem muito valor => todos estes acidentes foram no mesmo ano
#df['record_date_year'] = df['record_date'].dt.year

df_train['record_date_month'] = df['record_date'].dt.month
df_train['record_date_day'] = df['record_date'].dt.day
df_train['record_date_hour'] = df['record_date'].dt.hour

df_train.drop('record_date', axis = 1, inplace= True)


X_train = df_train.drop(['incidents'], axis = 1)
y_train = df_train['incidents']


# TEST 

df.drop(['affected_roads'],axis=1, inplace = True)


# O atributo cidade não acrescenta nada, uma vez que tem todos os mesmo valor, e portanto não vai afetar o modelo
df.drop('city_name', axis = 1, inplace = True)

# O atributo avg_precipitation também é unico por isso não acrescenta muito 
#print(df_no_city['avg_precipitation'].nunique())
df.drop('avg_precipitation', axis = 1, inplace = True)

labels = df['avg_rain'].astype('category').cat.categories.tolist()
replace_map_rain = {'avg_rain':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_rain)
df.replace(replace_map_rain, inplace = True)

labels = df['incidents'].astype('category').cat.categories.tolist()
replace_map_incidents = {'incidents':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_incidents)
df.replace(replace_map_incidents, inplace = True)

labels = df['magnitude_of_delay'].astype('category').cat.categories.tolist()
replace_map_delay = {'magnitude_of_delay':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_delay)
df.replace(replace_map_delay, inplace = True)

labels = df['luminosity'].astype('category').cat.categories.tolist()
replace_map_luminosity = {'luminosity':{k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#print(replace_map_luminosity)
df.replace(replace_map_luminosity, inplace = True)

df['record_date'] = pd.to_datetime(df['record_date'], format = "%Y-%m-%d %H:%M", errors='coerce')

assert df['record_date'].isnull().sum() == 0, 'missing record date'

#Não tem muito valor => todos estes acidentes foram no mesmo ano
#df['record_date_year'] = df['record_date'].dt.year

df['record_date_month'] = df['record_date'].dt.month
df['record_date_day'] = df['record_date'].dt.day
df['record_date_hour'] = df['record_date'].dt.hour

df.drop('record_date', axis = 1, inplace= True)



modelTree = DecisionTreeClassifier().fit(X_train, y_train)

y_pred_tree = modelTree.predict(df)


#y_pred = model.predict(testdatasetAtt)
#df = pd.DataFrame(testdatasetID)
# RowId,Incidents
pred['RowId'] = range(1, len(y_pred_tree)+1)
pred['Incidents'] = y_pred_tree
pred.to_csv('Group16_Try1.csv', index = False)