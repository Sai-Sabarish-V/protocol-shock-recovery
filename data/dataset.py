from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = [
    "taxonomic_class",
    "habitat",
    "diet",
    "body_size",
    "activity_pattern",
]


class ObjectDataset(Dataset):

    def __init__(self, csv_path=None):

        if csv_path is None:
            csv_path = Path(__file__).parent / "objects_1024.csv"

        self.df = pd.read_csv(csv_path)

        self.feature_maps = {}

        encoded_columns = []

        for column in FEATURE_COLUMNS:

            categories = sorted(self.df[column].unique())

            mapping = {
                category: index
                for index, category in enumerate(categories)
            }

            self.feature_maps[column] = mapping

            encoded_columns.append(
                self.df[column].map(mapping)
            )

        encoded_df = pd.concat(encoded_columns, axis=1)

        self.features = torch.tensor(
            encoded_df.values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return {
            "object_id": int(self.df.iloc[index]["object_id"]),
            "features": self.features[index],
        }