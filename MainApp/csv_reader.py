import pandas as pd

def read_csv(file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        print(f"Row {index}:")
        for column in df.columns:
            print(f"Column '{column}': {row[column]}")

file_path = 'input.csv'
read_csv(input)