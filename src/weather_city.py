import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

labelled = pd.read_csv(sys.argv[1])
unlabelled = pd.read_csv(sys.argv[2])
output = sys.argv[3]

X = labelled.drop(['city', 'year'], axis=1)
y = labelled['city'].values

# same data split every time you run the code with random_state
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=14))

model.fit(X_train, y_train)
model_score = model.score(X_valid, y_valid)
print('Model Score: ', model_score)
#model_score = model.score(X_train, y_train)
#print('Training Score: ', model_score)

X_unlabelled = unlabelled.drop(['city', 'year'], axis=1)
predictions = model.predict(X_unlabelled)

df = pd.DataFrame({'truth': y_valid, 'prediction': model.predict(X_valid)})
#print(df[df['truth'] != df['prediction']])

# output file
pd.Series(predictions).to_csv(output, index=False, header=False)