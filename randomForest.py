from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load dataset
joke = pd.read_csv("comedy.csv", sep=',')

#preprocessing data
X = joke.content
y = joke.flattened_categories
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

count_vect = CountVectorizer()

X_train = count_vect.fit_transform(X_train)
y_train = count_vect.transform(y_train)
X_test = count_vect.transform(X_test)
y_test = count_vect.transform(y_test)



X_train_array = X_train.toarray()
y_train_array = y_train.toarray()
print(y_train_array)
print(X_train_array)

#using classification
rfc = RandomForestClassifier()
rfc.fit(X_train_array, y_train_array)
pred_rfc = rfc.predict(X_test.toarray())
print(classification_report(y_test.toarray(), pred_rfc))
