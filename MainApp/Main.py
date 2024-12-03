import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers, regularizers
from fuzzywuzzy import process
from DataCollection import dataCollection

import os

# Sets file navigation to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Commented out following two lines until needed
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
def select_file_by_number(genre, mode="500"):
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

def getDataFiles(directory='Datasets'):
    return os.listdir(directory)

operation_mode = input("Choose an option (UPDATED): \n1. Run model on existing file\n2. Run Parameter Tests\n3. Test single Song(WIP)\n")
genre=""
input_file=""
dataCollect=""
DSDirectory = 'Datasets'

#Old option 1 before spotify API changes
#if (operation_mode == "1"):
    #genre = input("Enter genre name: ")
    #dataCollect = dataCollection(genre, DSDirectory)
    #dataCollect.collect()
    #input_file = DSDirectory + "/" + genre +"_"+"500_tracks.csv"

if (operation_mode == "1"):
    print("Available Datasets:")
    datasets = getDataFiles(DSDirectory)
    count = 1
    for filename in datasets:
       print(f"{count}. {filename}")
       count= count+1
    file_choice = datasets[int(input("Enter file number: ")) - 1]
    input_file = DSDirectory + "/" + file_choice

elif (operation_mode == "2"):
    print("Available Datasets:")
    datasets = getDataFiles(DSDirectory)
    count = 1
    for filename in datasets:
       print(f"{count}. {filename}")
       count= count+1
    file_choice = datasets[int(input("Enter file number: ")) - 1]
    input_file = DSDirectory + "/" + file_choice

elif (operation_mode == "3"):
    print("Available Datasets:")
    datasets = getDataFiles(DSDirectory)
    count = 1
    for filename in datasets:
       print(f"{count}. {filename}")
       count= count+1
    file_choice = datasets[int(input("Enter file number: ")) - 1]
    input_file = DSDirectory + "/" + file_choice

#Old option 4
#elif (operation_mode == "4"):
    #id = input("enter track id:")
    #dataCollect = dataCollection()
    #print(dataCollection.getSongGenres(id))


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

#
#
#Update these values whenever better parameters are found
optimal_dr=0.1
optimal_bn=False
optimal_ls=[128, 128, 128, 128, 128, 128]
optimal_l2=0.0001
optimal_e=57

#Standard model running, update with 
if (operation_mode == "1"):
    model = build_model(
        input_shape=(X_train.shape[1],), 
        dropout_rate=optimal_dr, 
        batch_norm=optimal_bn, 
        layer_sizes=optimal_ls, 
        l2_reg=optimal_l2
    )

    model.fit(X_train, y_train, epochs=optimal_e, batch_size=32, validation_split=0.2)
    y_pred = model.predict(X_test).flatten()
    test_loss, test_mae = model.evaluate(X_test, y_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\tDR\tBN\tLS\t\t\t\tL2\tE\t\tMAE\tMSE\tRMSE\tR2")
    print(f'\n\t{optimal_dr}\t{optimal_bn}\t{optimal_ls}\t{optimal_l2}\t{optimal_e}\t\t{test_mae:.4f}\t{test_mse:.4f}\t{test_rmse:.4f}\t{test_r2:.4f}')

if (operation_mode == "2"):
    output_file = open("sample_run.txt", "w")
    output_file.write(f"#\tDR\tBN\tLS\tL2\tE\t\tMAE\tMSE\tRMSE\tR2")
    model_count = 0
    for dr in [0, 0.1, 0.2]:
        for bn in [True, False]:
            # for ls in [[100, 100, 100, 100, 100, 100]]:
            for ls in [[x] * i for x in range(96, 105, 1) for i in range(6, 8)]:
                # 0.001 never seems to get a higher score
                for l2 in [0, 0.0001]:
                    for e in [53, 54, 55, 56, 57]:
                        model = build_model(
                            input_shape=(X_train.shape[1],), 
                            dropout_rate=dr, 
                            batch_norm=False, 
                            layer_sizes=ls, 
                            l2_reg=l2
                        )

                        model.fit(X_train, y_train, epochs=e, batch_size=32, validation_split=0.2)
                        y_pred = model.predict(X_test).flatten()
                        test_loss, test_mae = model.evaluate(X_test, y_test)
                        test_mse = mean_squared_error(y_test, y_pred)
                        test_rmse = np.sqrt(test_mse)
                        test_r2 = r2_score(y_test, y_pred)

                        if (test_r2 > 0.13):
                            output_file.write(f'\n{model_count}\t{dr}\t{bn}\t{ls}\t{l2}\t{e}\t\t{test_mae:.4f}\t{test_mse:.4f}\t{test_rmse:.4f}\t{test_r2:.4f}')
                            model_count += 1

    output_file.close()
