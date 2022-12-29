Try 3: \
* Preprocessing:\
        - encoding_categorical_data(df)\
        - drop_columns_zero_std(df)\
        - drop_missing\
        - df = count_all_roads_per_line(df)\
* Model:\
        -RandomForest: criterion='entropy', n_estimators=125, max_depth=18, max_features='log2'\
* Results:
```
            precision    recall  f1-score   support

           1       0.90      0.89      0.90       312
           2       0.86      0.90      0.88       203
           3       0.91      0.77      0.83       153
           4       0.98      0.99      0.99       622
           5       0.89      0.95      0.92       185

    accuracy                           0.93      1475
   macro avg       0.91      0.90      0.90      1475
weighted avg       0.93      0.93      0.93      1475
```
* Kaggle Accuracy: 0.94 
    
**Deve ter overfitting**
**Corrigir abordagem de tratamento dos nan do test_data**

Try 4:
* Preprocessing:\
        - encoding_categorical_data(df)\
        - drop_columns_zero_std(df)\
        - drop_missing\
        - df = count_all_roads_per_line(df)\
        - df = balancedData(df)\ (não aplicado no test data)
* Model:\
        -RandomForest: criterion='entropy', n_estimators=125, max_depth=18, max_features='log2'\
* Results:
```
            precision    recall  f1-score   support

           1       0.98      0.97      0.98       631
           2       0.97      0.99      0.98       603
           3       0.98      0.99      0.99       601
           4       1.00      0.97      0.98       599
           5       0.98      0.99      0.98       593

    accuracy                           0.98      3027
   macro avg       0.98      0.98      0.98      3027
weighted avg       0.98      0.98      0.98      3027
```
* Kaggle Accuracy: 0.93
    
**Corrigir abordagem de tratamento dos nan do test_data**


Grid Search
'criterion': 'gini', 'max_depth': 15, 'max_features': 'log2', 'n_estimators': 125


### Análise de Confusion Matrix:

Podemos observar que à uma confusão: há situações em que valores da classe 4 são previsto como 2, porquê...? (none confundido com o low, acho que faz sentido poder existir confusão)

há situações de confusão entre valores de 1 que são previstos como o 5...? (high confundido com o very high, acho que faz sentido)

### Análise de boxplots:
* Replace magnitude_delay: 
```
{'magnitude_of_delay': {'MAJOR': 1, 'MODERATE': 2, 'UNDEFINED': 3}}
```
* Replace luminosity:
```
{'luminosity': {'DARK': 1, 'LIGHT': 2, 'LOW_LIGHT': 3}}
```
* Replace avg rain:
```
{'avg_rain': {'Sem Chuva': 1, 'chuva forte': 2, 'chuva fraca': 3, 'chuva moderada': 4}}
```
* Replace incidents:
```
{'incidents': {'High': 1, 'Low': 2, 'Medium': 3, 'None': 4, 'Very_High': 5}}
```


Try 5:
* Preprocessing:\
        - encoding_categorical_data(df)\
        - drop_columns_zero_std(df)\
        - drop_missing\
        - df = count_all_roads_per_line(df)\
        - df = balancedData(df)\ (não aplicado no test data)
        - df = replaceOutliers\ (não aplicado no test data)
* Model:\
        -RandomForest: 'criterion': 'gini', 'max_depth': 15, 'max_features': 'log2', 'n_estimators': 125\
* Results:
```
             precision    recall  f1-score   support

           1       0.98      0.91      0.95       631
           2       0.97      0.98      0.97       603
           3       0.96      0.99      0.98       601
           4       0.99      0.97      0.98       599
           5       0.93      0.98      0.96       593

    accuracy                           0.97      3027
   macro avg       0.97      0.97      0.97      3027
weighted avg       0.97      0.97      0.97      3027
```
* Kaggle Accuracy: 0.90
    


Try 6:
* Preprocessing:\
        - encoding_categorical_data(df)\
        - drop_columns_zero_std(df)\
        - drop_missing
        - df = count_all_roads_per_line(df)\
        - df = balancedData(df)\ (não aplicado no test data)
        - df = replaceOutliers\
* Model:\
        -RandomForest: 'criterion': 'gini', 'max_depth': 15, 'max_features': 'log2', 'n_estimators': 125\
* Results:
```
             precision    recall  f1-score   support

           1       0.98      0.91      0.95       631
           2       0.97      0.98      0.97       603
           3       0.96      0.99      0.98       601
           4       0.99      0.97      0.98       599
           5       0.93      0.98      0.96       593

    accuracy                           0.97      3027
   macro avg       0.97      0.97      0.97      3027
weighted avg       0.97      0.97      0.97      3027
```
* Kaggle Accuracy: 0.89
    


### Análise do Tratamento de Outliers:

Para tratar os outliers tentamos usar replacement pelos quartis mais próximos, contudo, este tratamento não melhorou a nossa accuracy. Optamos assim por descartar temporariamente os outliers.

Podemos tentar justificar isto com: podem haver circunstâncias climatéricas anormais que fazem com que o número de acidentes aumente. Tirar estes outliers pode fazer com que se percam dados relevantes sobre o ambiente do sistema.



* Teste: 
- encoding_categorical_data(df)\
- drop_columns_zero_std(df)\
- replace_missing_affected_roads(df)
- df = count_different_roads_per_line(df)

```
 precision    recall  f1-score   support

           1       0.87      0.90      0.88       313
           2       0.86      0.86      0.86       201
           3       0.90      0.79      0.84       175
           4       0.97      0.98      0.98       632
           5       0.90      0.94      0.92       179

    accuracy                           0.92      1500
   macro avg       0.90      0.89      0.90      1500
weighted avg       0.92      0.92      0.92      1500
```


* Teste:
- encoding_categorical_data(df)\
- drop_columns_zero_std(df)\
- replace_missing_affected_roads(df)
- df = count_all_roads_per_line(df)
- drop_magnitude_of_delay(df)\
- drop_columns_zero_std(df)\

```
precision    recall  f1-score   support

           1       0.91      0.90      0.91       313
           2       0.86      0.89      0.88       201
           3       0.89      0.82      0.85       175
           4       0.98      0.99      0.99       632
           5       0.92      0.92      0.92       179

    accuracy                           0.93      1500
   macro avg       0.91      0.90      0.91      1500
weighted avg       0.93      0.93      0.93      1500
```


Try 7:
* Preprocessing:\
        - encoding_categorical_data(df)\
        - drop_columns_zero_std(df)\
        - replace_missing
        - df = count_all_roads_per_line(df)\
        - df.drop('magnitude_of_delay')
* Model:\
        -RandomForest: criterion='entropy', n_estimators=125, max_depth=18, max_features='log2', class_weight="balanced_subsample"\
* Results:
```
             precision    recall  f1-score   support

           1       0.90      0.91      0.90       313
           2       0.85      0.89      0.87       201
           3       0.89      0.79      0.84       175
           4       0.98      0.99      0.99       632
           5       0.93      0.92      0.92       179

    accuracy                           0.93      1500
   macro avg       0.91      0.90      0.90      1500
weighted avg       0.93      0.93      0.93      1500
```
* Kaggle Accuracy: 0.936
        **Não deve ter muito overfitting**