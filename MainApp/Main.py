import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers, regularizers
from fuzzywuzzy import fuzz
from DataCollection import dataCollection
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os


# Sets file navigation to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
CORS(app)

DATASETS_DIR = os.path.join(os.getcwd(), "Datasets")


# List datasets
@app.route('/list-datasets', methods=['GET'])
def list_datasets():
    """
    This endpoint lists all available datasets (CSV files) in the 'Datasets' directory.
    """
    try:
        # Ensure the dataset directory exists
        if not os.path.exists(DATASETS_DIR):
            print(f"Error: Dataset directory {DATASETS_DIR} does not exist.")
            return jsonify({"error": "Dataset directory does not exist"}), 500

        # List all .csv files
        datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
        print(f"Datasets found: {datasets}") 

        if not datasets:
            return jsonify({"error": "No datasets found in the directory"}), 404

        return jsonify({"datasets": datasets})

    except Exception as e:
        print(f"Error in list_datasets(): {e}")
        return jsonify({"error": str(e)}), 500

    
#Main.py 
@app.route('/run-main', methods=['POST'])
def run_main():
    """
    This endpoint triggers the Main.py logic based on the provided action and dataset.
    """
    try:
        # Parse JSON data from the request
        data = request.json
        print(f"Request data: {data}")

        # Validate required parameters
        action = data.get("action")
        dataset = data.get("dataset")

        if not action or not dataset:
            return jsonify({"error": "Action and dataset are required"}), 400

        # Validate dataset file
        dataset_path = os.path.join(DATASETS_DIR, dataset)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file {dataset_path} does not exist.") 
            return jsonify({"error": f"Dataset file '{dataset}' does not exist"}), 404

        command = ["python", "Main.py"]
        input_data = f"{action}\n{dataset_path}\n"

        print(f"Executing command: {command} with input: {input_data}")  # Debugging output

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_data)

        if process.returncode == 0:
            print(f"Command output: {stdout}") 
            return jsonify({"output": stdout})
        else:
            print(f"Command error: {stderr}")  
            return jsonify({"error": stderr}), 500

    except Exception as e:
        print(f"Error in run_main(): {e}") 
        return jsonify({"error": str(e)}), 500


#run LimitedDataRedesign.py()
@app.route('/run-limited', methods=['POST'])
def run_limited():
    data = request.json
    song_name = data.get("song_name")

    if not song_name:
        return jsonify({"error": "Song name is required"}), 400

    try:
        # Call LimitedDataRedesign.py 
        command = ["python", "LimitedDataRedesign.py"]
        input_data = f"{song_name}\n"

        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate(input=input_data)

        if process.returncode == 0:
            return jsonify({"output": stdout})
        else:
            return jsonify({"error": stderr}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


#Commented out following two lines until needed
# List of predefined genres(test for fuzzywuzzy, add for more)
available_genres = ["rock", "pop", "jazz", "classical", "hip-hop", "country"]

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

# All that aren't categorical
filterColumns = ["Track_Name", "Track Duration (ms)", "Popularity", "Danceability", "Energy","Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo"]

def getDataFiles(directory='Datasets'):
    return os.listdir(directory)

operation_mode = input("Choose an option (UPDATED): \n1. Run model on existing file\n2. Run Parameter Tests\n3. Test single Song(not working)\n")
genre=""
input_file=""
dataCollect=""
DSDirectory = 'Datasets'

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
    
    name_to_search = input("Enter the name of the song: ")
    print(name_to_search)
    search_result_name, search_result_accuracy, search_result_path = find_name_in_csvs(DSDirectory, name_to_search)

    print(f"Closest match: {search_result_name} (Accuracy: {search_result_accuracy}%)")
    input_file = DSDirectory + "/" + search_result_path

#No longer needed?
mode="500"

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
    X = np.array([[getattr(record, col) for col in record.__dict__ if col != 'Popularity' and col != 'Track_Name'] for record in records])
    y = np.array([record.Popularity for record in records])

    # Normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


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

X_train, X_test, y_train, y_test, scaler = prepare_data(read_csv())

#
#
#Update these values whenever better parameters are found
optimal_dr=0.12
optimal_bn=False
optimal_ls=[192, 192, 192, 192, 192, 192, 192, 192]
optimal_l2=0.0001
optimal_e=60

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

#
#
#Functions needed and implemented for option 3

#Function to find and return record for a track searched by name in a record set
def get_single_song_record(song_name,records):
    #set up variables
    name_lowered= song_name.lower()
    top_match= None
    top_ratio=0

    #iterate through records
    for record in records:
        #sort by best match
        match_ratio= fuzz.token_sort_ratio(name_lowered,record.Track_Name.lower())
        #set ratio to top ratio if best so far
        if top_ratio<match_ratio:
                top_match=record
                top_ratio=match_ratio

    return top_match,top_ratio

#Function that takes a trained model and a track name and prints
# an analysis of how over/underrated the song is based on the
# difference between the actual and predicted popularity
def compare_single_prediction(records, scaler, model, song_name):
    #get a song that matches
    target_song, accuracy = get_single_song_record(song_name, records)
    

    if not(target_song):
        print("No match for song found.")

    else:
        print(f"Match found: {target_song.Track_Name} accuracy: {accuracy}%")
        #prepare features
        audio_features = scaler.transform([[getattr(target_song, col) for col in filterColumns if col not in ['Track_Name', 'Popularity']]])
        
        actual_pop = target_song.Popularity

        predicted_pop = model.predict(audio_features).flatten()[0]

        #calculate difference and print results
        difference = predicted_pop - actual_pop
        over_under = "underrated" if difference > 0 else "overrated"
        print(f"Recorded Popularity: {actual_pop:.2f}")
        print(f"Predicted Popularity: {predicted_pop:.2f}")
        print(f"This song is {over_under} by {abs(difference):.2f}")



if (operation_mode == "3"):
    model = build_model(
                input_shape=(X_train.shape[1],), 
                dropout_rate=0.12, 
                batch_norm=False, 
                layer_sizes=[192, 192, 192, 192, 192, 192, 192, 192], 
                l2_reg=0.0001
            )

    model.fit(X_train, y_train, epochs=optimal_e, batch_size=32, validation_split=0.2)
    y_pred = model.predict(X_test).flatten()
    test_loss, test_mae = model.evaluate(X_test, y_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\tDR\tBN\tLS\t\t\t\tL2\tE\t\tMAE\tMSE\tRMSE\tR2")
    print(f'\n\t{optimal_dr}\t{optimal_bn}\t{optimal_ls}\t{optimal_l2}\t{optimal_e}\t\t{test_mae:.4f}\t{test_mse:.4f}\t{test_rmse:.4f}\t{test_r2:.4f}')

    #model.evaluate(X_test, y_test)
    compare_single_prediction(read_csv(), scaler, model, name_to_search)
