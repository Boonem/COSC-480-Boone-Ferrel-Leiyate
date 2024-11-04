import pandas as pd
import os

class DataStorage:
    def __init__(self, filename):
        self.filename = filename

    def save_to_csv(self, data, headers):
        # If the data contains lists or dictionaries, format them as strings
        formatted_data = []
        for row in data:
            formatted_row = {key: (','.join(value) if isinstance(value, list) else value) for key, value in row.items()}
            formatted_data.append(formatted_row)

        df = pd.DataFrame(formatted_data, columns=headers)
        df.to_csv(self.filename, index=False)
        print(f"Data has been saved to {self.filename}")

# Example
# storage = DataStorage('spotify_tracks.csv')
# storage.save_to_csv(data, headers=['track_name', 'artist', 'genre', 'popularity', ...])
