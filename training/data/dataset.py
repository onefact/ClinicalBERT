import torch
from pathlib import Path
import pandas as pd
from collections import Counter
import random
import numpy as np
import itertools

class NotesforPT(torch.utils.data.Dataset):
    def __init__(self, data_file):
        super().__init__()
        data_file = Path(data_file)
        if data_file.suffix == ".json":
            df = pd.read_json(data_file)
        elif data_file.suffix == ".csv":
            df = pd.read_csv(data_file)
            df["text"] = df["text"].apply(lambda x: eval(x))
        elif data_file.suffix == ".pkl":
            df = pd.read_pickle(data_file)

        self.df = df
        print("Loaded {} rows".format(len(df)))
        self.data = df.to_dict(orient='records')

    def __getitem__(self, idx):
        return self.data[idx]
