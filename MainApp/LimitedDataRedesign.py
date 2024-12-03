import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers, regularizers
from fuzzywuzzy import fuzz
from DataCollection import dataCollection

import os

# Sets file navigation to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Pruning attempts for improving prediction
filterColumns = ["Track Duration (ms)", "Popularity", "Danceability", "Energy","Loudness",
                "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence"]

def getDataFiles(directory='Datasets'):
    return os.listdir(directory)

def find_name_in_csvs(folder_path, name_to_search):
    # List all files in the folder
    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith('.csv')]

    # Result
    search_name_lower = name_to_search.lower()
    found = False
    search_result_name = ''
    search_result_accuracy = 0
    search_result_path = ''


    # Iterate over each CSV file
    for file in files:
        file_path = os.path.join(folder_path, file)

        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the first column exists
            if df.shape[1] > 0:
                first_column = df.iloc[:, 0].astype(str)  # Ensure it's in string format
                
                # Check for a match in the first column using fuzzywuzzy
                for entry in first_column:
                    entry_lower = entry.lower()
                    attempt_ratio = fuzz.token_sort_ratio(search_name_lower, entry_lower)
                    if attempt_ratio > search_result_accuracy:
                        search_result_name = entry_lower
                        search_result_accuracy = attempt_ratio
                        search_result_path = file
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return search_result_name, search_result_accuracy, search_result_path

# Example Usage
DSDirectory = 'Datasets'
name_to_search = input("Enter the name of the song: ")
print(name_to_search)
search_result_name, search_result_accuracy, search_result_path = find_name_in_csvs(DSDirectory, name_to_search)

print(f"Closest match: {search_result_name} (Accuracy: {search_result_accuracy}%)")
input_file = DSDirectory + "/" + search_result_path


#input_file = input("Enter the filename you to use: ")
#genre = get_genre()
mode="500"
#mode = input("Enter 'single' or 'all' based on the mode used for data collection: ")
#input_file = select_file_by_number(genre, mode)  # File path to CSV data
# Check if a file was selected
if input_file:
    print(f"Using file: {input_file}")
else:
    print("No file selected.")
    exit()  # Exit if no file was selected

# Takes in csv, filters to only columns we want to use, returns TestRecord list
def read_csv():
    return [TestRecord(**row) for _, row in (pd.read_csv(input_file, usecols=filterColumns)).iterrows()]

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

def build_model(input_shape, dropout_rate, batch_norm, layer_sizes=[64, 32, 16], l2_reg=0.0):
    model = keras.Sequential()
    
    # First layer
    model.add(layers.Dense(layer_sizes[0], activation='relu', input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(layers.BatchNormalization())
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Hidden layers
    for size in layer_sizes[1:]:
        model.add(layers.Dense(size, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        if batch_norm:
            model.add(layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1))
    
    model.compile(optimizer='sgd', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return model

X_train, X_test, y_train, y_test = prepare_data(read_csv())
# Enjoy the silence
keras.utils.disable_interactive_logging()

model = build_model(
    input_shape=(X_train.shape[1],), 
    dropout_rate=0.12, 
    batch_norm=False, 
    layer_sizes=[192, 192, 192, 192, 192, 192, 192, 192], 
    l2_reg=0.0001
)

model.fit(X_train, y_train, epochs=e, batch_size=32, validation_split=0.2, verbose=0)
y_pred = model.predict(X_test).flatten()
test_loss, test_mae = model.evaluate(X_test, y_test)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

