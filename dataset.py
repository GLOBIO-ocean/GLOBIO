import os
import json
import pandas as pd
import torch
import math
from torch.utils.data import Dataset

REGIONS = {
    "mediterranean": {"lat": (30, 46), "lon": (-6, 36)},
    "indian_ocean": {"lat": (-40, 30), "lon": (20, 120)},
    "bay_of_bengal": {"lat": (8, 22), "lon": (82, 98)},
    "north_atlantic": {"lat": (45, 70), "lon": (-70, 10)},
    "southern_ocean": {"lat": (-90, -60), "lon": (-180, 180)},
    "north_pacific": {"lat": (45, 65), "lon": (-180, -120)},
    "r1" : {"lat": (60, 90), "lon": (-180, 180)},
    "r2" : {"lat": (30, 60), "lon": (-180, 180)},
    "r3" : {"lat": (0, 30), "lon": (-180, 180)},
    "r4" : {"lat": (-30, 0), "lon": (-180, 180)},
    "r5" : {"lat": (-60, -30), "lon": (-180, 180)},
    "r6" : {"lat": (-90, -60), "lon": (-180, 180)}
}


def from_string_to_tensor(string):
    threshold = 99999
    label = 1  
    string = string[8:-2].split(",")
    out = torch.zeros(200)
    for ind in range(len(string)):
        if float(string[ind]) >= threshold:
            label = 0
        out[ind] = torch.tensor(float(string[ind]))
    return out, label


class FloatDataset(Dataset):

    def __init__(
            self, 
            path_df=None, 
            stats_path=None, 
            region=None, 
            lat_range=None, 
            lon_range=None, 
            max_samples=None
        ):

        super().__init__()

        if path_df is None:
            raise Exception("Paths should be given as input to initialize the Float class.")

        self.path_df = path_df
        self.df = pd.read_csv(self.path_df)

        if region is not None:
            if region not in REGIONS:
                raise ValueError(f"Unknown region: {region}")
            lat_range = REGIONS[region]["lat"]
            lon_range = REGIONS[region]["lon"]

        if lat_range is not None and lon_range is not None:
            self.df = self.filter_by_region(self.df, lat_range, lon_range)

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = None

        if max_samples is not None:
            # keep metadata column + first max_samples profiles
            self.df = self.df.iloc[:, :max_samples + 1]


    def normalize(self, x, name):
        if self.norm_stats is not None and name in self.norm_stats:
            mean, std = self.norm_stats[name]
            return (x - mean) / std
        return x
    
    def denormalize(self, x, name):
        if self.norm_stats is not None and name in self.norm_stats:
            mean, std = self.norm_stats[name]
            return x * std + mean
        return x

    def encode_time(self, year, day_rad):
        year_offset = (year - 2010) #2010 --> 2025

        day_sin = math.sin(day_rad)
        day_cos = math.cos(day_rad)

        return torch.tensor(year_offset), torch.tensor(day_sin), torch.tensor(day_cos)

    def encode_space(self, lat, lon):
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)

        return torch.tensor(x), torch.tensor(y), torch.tensor(z)

    def filter_by_region(self, df, lat_range, lon_range):
        latitudes = df.iloc[2, 1:].astype(float)
        longitudes = df.iloc[3, 1:].astype(float)

        mask = (
            (latitudes >= lat_range[0]) & (latitudes <= lat_range[1]) &
            (longitudes >= lon_range[0]) & (longitudes <= lon_range[1])
        )
        filtered_columns = [df.columns[0]] + list(df.columns[1:][mask.values])

        return df.loc[:, filtered_columns]
    

    def __len__(self):
        """Denotes the total number of samples"""
        return self.df.shape[1] - 1

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Device name, useful for plots
        column_name = self.df.columns[index + 1]
        device_name = os.path.basename(str(column_name))

        # Get the column corresponding to this sample
        samples = self.df.iloc[:, index + 1].tolist()

        # ---- TIME ----
        year = float(samples[0])
        day_rad = float(samples[1])
        year, day_sin, day_cos = self.encode_time(year, day_rad)

        # ---- SPACE ----
        lat = float(samples[2])
        lon = float(samples[3])
        x, y, z = self.encode_space(lat, lon)

        # ---- PROFILES ----
        temp, label_temp = from_string_to_tensor(samples[4])
        psal, label_psal = from_string_to_tensor(samples[5])
        doxy, label_doxy = from_string_to_tensor(samples[6])
        target_variable, _ = from_string_to_tensor(samples[7])

        temp = self.normalize(temp, "temp")
        psal = self.normalize(psal, "psal")
        doxy = self.normalize(doxy, "doxy")

        label = label_doxy * label_psal * label_temp  

        return {"year": year,
            "day": torch.stack([day_sin, day_cos]),
            "space" : torch.stack([x, y, z]),
            "temp": temp,
            "psal": psal,
            "doxy": doxy,
            "target_variable": target_variable,
            "label": label,
            "device_name": device_name
        }
