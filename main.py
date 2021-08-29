import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

"""
Speed Dating Experiment -
Machine learning model for predicting a match between two people.
"""

# Import dataset
speed_dating_data = pd.read_csv('Speed Dating Data.csv', encoding="ISO-8859-1")
dating_data = speed_dating_data[['gender', 'iid', 'attr3_s', 'sinc3_s', 'intel3_s', 'fun3_s', 'amb3_s', 'attr', 'sinc',
                                 'intel', 'fun', 'amb', 'shar', 'match']]

# Delete non available values
dating_data_clean = dating_data.dropna().reset_index()

# Create the training and the testing sets
x = dating_data_clean[['gender', 'attr3_s', 'sinc3_s', 'intel3_s', 'fun3_s', 'amb3_s', 'attr', 'sinc', 'intel', 'fun',
                       'amb', 'shar']]
y = dating_data_clean['match']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=3, stratify=y)

# Logistic Regression Model
model_log_reg = LogisticRegression(C=3, random_state=43)
log_reg = model_log_reg.fit(x_train, y_train)
predict_train_log_reg = log_reg.predict(x_train)
predict_test_log_reg = log_reg.predict(x_test)

# Accuracy
print('Training Accuracy Score:', accuracy_score(y_train, predict_train_log_reg))
print('Validation Accuracy Score :', accuracy_score(y_test, predict_test_log_reg))
# Precision, recall and f1-score table
print(classification_report(y_test, model_log_reg.predict(x_test)))
