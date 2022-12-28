Try 3: \
* Preprocessing:\
        - encoding_categorical_data(df)\
        - drop_columns_zero_std(df)\
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