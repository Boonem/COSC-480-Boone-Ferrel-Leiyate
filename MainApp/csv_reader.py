import pandas as pd

def read_csv(file_path):
    output = []
    df = pd.read_csv(file_path)
    objects_list = [TestRecord(**row) for _, row in df.iterrows()]
    print(vars(objects_list[0]))

class TestRecord:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

file_path = 'input.csv'
read_csv(input)