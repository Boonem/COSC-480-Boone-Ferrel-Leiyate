import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

import os

# Sets file navigation to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))



# "All"
#filterColumns = [
#        "Track Duration (ms)", "Popularity", "Danceability", "Energy",
#        "Key", "Loudness", "Mode", "Speechiness", "Acousticness",
#        "Instrumentalness", "Liveness", "Valence", "Tempo", "Time Signature",
#]

# All that aren't categorical
filterColumns = ["Track Duration (ms)", "Popularity", "Danceability", "Energy","Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo"]

input_file = input("Enter the filename you to use: ")

# Takes in csv, filters to only columns we want to use, returns TestRecord list
def read_csv(file_path):
    return [TestRecord(**row) for _, row in (pd.read_csv(file_path, usecols=filterColumns)).iterrows()]

class TestRecord:
    def __init__(self, **kwargs):
        # Filter to just fields we care about for training
        for key, value in kwargs.items():
            setattr(self, key, value)

# Loads data from TestRecord instances
def prepare_data(records):
    # Postures everything except popularity as input, and only popularity as output
    X = np.array([[getattr(record, col) for col in record.__dict__ if col != 'Popularity'] for record in records])
    y = np.array([record.Popularity for record in records])

    # Normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return model

X_train, X_test, y_train, y_test = prepare_data(read_csv(input_file))

model = build_model((X_train.shape[1],))
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test).flatten()
test_loss, test_mae = model.evaluate(X_test, y_test)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print(f'Mean absolute error: {test_mae:.4f}')
print(f'Mean squared error: {test_mse:.4f}')
print(f'Root mean squared error: {test_rmse:.4f}')
print(f'R-squared: {test_r2:.4f}')