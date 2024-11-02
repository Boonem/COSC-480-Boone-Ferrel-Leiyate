import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from fuzzywuzzy import process


import os

# Sets file navigation to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# List of predefined genres(test for fuzzywuzzy, add for more)
available_genres = ["rock", "pop", "jazz", "classical", "hip-hop", "country"]

def get_genre():
    input_genre = input("Enter the genre you want to collect data for: ")
    best_match, score = process.extractOne(input_genre, available_genres)

    if score >= 80:  # Accepting matches with a score of 80 or higher
        print(f"Using genre: {best_match} (matched with {input_genre})")
        return best_match
    else:
        print("No close match found for genre. Please try again.")
        return get_genre()  # Recursively ask until a valid match is found


# Function to list and select a file by number
def select_file_by_number(genre, mode):
    # Directory where data files are saved
    directory = "../DataCollection"
    # Find all matching files
    files = [f for f in os.listdir(directory) if f.startswith(f"{genre}_{mode}")]
    
    if not files:
        print(f"No files found for genre '{genre}' and mode '{mode}'.")
        return None
    
    # Display files with numbers
    print("Available files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    
    # Prompt user to select a file by number
    choice = int(input("Enter the number of the file you want to use: ")) - 1
    
    # Validate choice
    if 0 <= choice < len(files):
        return os.path.join(directory, files[choice])
    else:
        print("Invalid selection.")
        return None
    

# "All"
#filterColumns = [
#        "Track Duration (ms)", "Popularity", "Danceability", "Energy",
#        "Key", "Loudness", "Mode", "Speechiness", "Acousticness",
#        "Instrumentalness", "Liveness", "Valence", "Tempo", "Time Signature",
#]

# All that aren't categorical
filterColumns = ["Track Duration (ms)", "Popularity", "Danceability", "Energy","Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo"]


#input_file = input("Enter the filename you to use: ")
genre = get_genre()
mode = input("Enter 'single' or 'all' based on the mode used for data collection: ")
input_file = select_file_by_number(genre, mode)  # File path to CSV data
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

def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    return model

X_train, X_test, y_train, y_test = prepare_data(read_csv())

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