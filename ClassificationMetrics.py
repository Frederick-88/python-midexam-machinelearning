import numpy as np
import pandas

# read csv file, if you receive error message while reading it. can add encoding property like below
dataframe = pandas.read_csv("top-rated-movie-genres.csv", encoding='latin-1')
dataframe.head()

# convert text to number to be used
convertToNumber = {'Biography': 0, 'Adventure': 1, 'Comedy': 2, 'Drama': 3, 'Action': 4, 'Animation': 5, 'Crime': 6, 'Drama': 7, 'Mystery': 8}
dataframe['Genre1'] = dataframe['Genre1'].map(convertToNumber)

features = ['Rating', 'CVotesMale', 'CVotesFemale']
X = dataframe[features]
Y = dataframe['Genre1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
X_train = scaleX.fit_transform(X_train)
X_test = scaleX.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
decision_tree = decision_tree.fit(X_train, y_train)

# make prediction
y_pred = decision_tree.predict(X_test)

# Accuracy
from sklearn import metrics
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))

# F1 Score
f1score = metrics.f1_score(y_test, y_pred, average='micro', labels=np.unique(y_pred))
print('F1 Score:', f1score)

# Precision
print('Precision Score:', metrics.precision_score(y_test, y_pred, average='micro'))

# Recall
print('Recall Score:', metrics.recall_score(y_test, y_pred, average='micro'))






















