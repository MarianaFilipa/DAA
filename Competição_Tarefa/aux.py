
# Trabalho Competição Kaggle

## Imports:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, plot_confusion_matrix,ConfusionMatrixDisplay

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import classification_report, plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier




def dataVisualization(df):
    print("** INFO **")
    print(df.info())
    print("** DESCRIBE **")
    print(df.describe())
    print("** HEAD **")
    print(df.head())
    print("** CORR MATRIX **")
    #sns.heatmap(df.corr())
    #plt.show()

    # check if there are any missing values in this dataframe
    print("** NULL VALUES?? **")
    print(df.isnull().sum())


def getMapIncidents(df):
    labels = df['incidents'].astype('category').cat.categories.tolist()
    replace_map_incidents = {'Incidents':{k: v for v,k in zip(labels,list(range(1,len(labels)+1)))}}
    #print(replace_map_incidents)
    return replace_map_incidents



def preprocessing(df, istest):
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

    if(istest == True):
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




def Split_Training_Test_Set(df):
    X = df.drop(['incidents'], axis = 1)
    y = df['incidents']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    return X_train, X_test, y_train, y_test


def getModel(modelType):
    if (modelType == "DecisionTree"):
        model = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = 10)
    elif(modelType == "RandomForest"):
        model = RandomForestClassifier(criterion = 'entropy', n_estimators = 125)
    else:
        model = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best')
    return model


def gridSearch(model, params):

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(model,params,scoring='neg_mean_squared_error',n_jobs=-1, cv=cv)
    
    return grid
    


if __name__ == "__main__":
    df = pd.read_csv("sbstpdaa2223/training_data.csv")

    dataVisualization(df)
    

    dictReplaceIncidents = getMapIncidents(df)
    preprocessing(df, True)
    
    dataVisualization(df)

    X_train, X_test, y_train, y_test = Split_Training_Test_Set(df)
    
    #X = df.drop(['incidents'], axis = 1)
    #y = df['incidents']

    #Decision Tree Model
    #model = getModel("RandomForest")


    #model = model.fit(X_train, y_train)
    #y_pred_tree = model.predict(X_test)
    # Classification_report
    #print(classification_report(y_test,y_pred_tree))

    #scores = cross_val_score(model, X, y, cv=10)
    #print(scores.mean())

    #Decision Tree:
    # log_loss deu asneira
    #params = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [5,10,15]}    
    #grid = gridSearch(model, params)
    #grid = grid.fit(X, y)
    #print(grid.best_params_)

    #Random Forest Classifier:
    # log_loss deu asneira
    #params = {'criterion': ['entropy'], 'max_depth': [13,15,18], 'n_estimators': [120, 125, 130]}    
    #grid = gridSearch(model, params)
    #grid = grid.fit(X,y)
    #print(grid.best_params_)



    
    ## TESTE
    X = df.drop(['incidents'], axis = 1)
    y = df['incidents']
    modelTree = getModel("RandomForest")
    modelTree = modelTree.fit(X, y)


    df_test = pd.read_csv("sbstpdaa2223/test_data.csv")
    preprocessing(df_test, False)

    y_pred_tree = modelTree.predict(df_test)
    # Classification_report
    #print(classification_report(y_test,y_pred_tree))


    pred = pd.DataFrame()
    print(dictReplaceIncidents)

    # RowId,Incidents
    pred['RowId'] = range(1, len(y_pred_tree)+1)
    pred['Incidents'] = y_pred_tree
    pred.replace(dictReplaceIncidents, inplace=True)
    pred.to_csv('Group16_Try2.csv', index = False)  
    