import pandas as pd
import os

class DataStorage:
    def __init__(self, filename):
        self.filename = filename

    def save_to_csv(self, data, headers):
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(self.filename, index=False)
        print(f"Data has been saved to {self.filename}")

# storage = DataStorage('')
# album_data = [['Album 1', '2024-09-30'] 
# storage.save_to_csv(album_data, headers=['Album Name', 'Release Date'])
