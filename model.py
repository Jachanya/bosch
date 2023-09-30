import numpy as np
import pandas as pd
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Collect program data
df = pd.read_csv("program_data.csv")
df.iloc[:, :8] /= 128
df.iloc[:, 9:17] /= 256

X, y = df.iloc[:-1,:], df.iloc[1:, :]
print(X.shape)

model = MultiOutputRegressor(LinearRegression())

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.10, random_state=42)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

filename = "finalized_model.sav"
pickle.dump(model, open(filename, 'wb'))
