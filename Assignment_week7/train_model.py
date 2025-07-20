
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = sns.load_dataset("mpg")
df.dropna(inplace=True)
df['origin'] = LabelEncoder().fit_transform(df['origin'])
df['name'] = LabelEncoder().fit_transform(df['name'])


X = df.drop("mpg", axis=1)
y = df["mpg"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mpg_model.pkl")
