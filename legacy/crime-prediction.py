import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split # class for train/test data split
from sklearn.preprocessing import LabelEncoder
# sklearn.preprocessing is a module in scikit learn that contains tools to prepare your data before training a machine learning model
# labelencoder is a class inside preprocessing that is used to convert categorical labels into numberic labels. Because ML models can't handle text labels directly.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

conn = sqlite3.connect('crime_data.db')
query = "SELECT * FROM ml_data"
df = pd.read_sql_query(query, conn)


# Encoding data + preprocessing
le = LabelEncoder()
df['CODE_DEFINED'] = le.fit_transform(df['CODE_DEFINED'])

# Data Train/Test Split
X = df[['month', 'day_of_week', 'day', 'year']]
y = df['CODE_DEFINED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,       # Number of trees
    max_depth=None,         # Max depth of each tree
    random_state=42,        # For reproducibility
    criterion='gini' 
    )
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(model, 'crime_model.pkl')
joblib.dump(le, 'label_encoder.pkl')