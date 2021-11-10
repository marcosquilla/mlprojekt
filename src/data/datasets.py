from pathlib import Path
from datetime import timedelta
import pandas as pd
import torch
from torch.utils.data import Dataset

class CarDataset_s1(Dataset):
    def __init__(self, indices, time_window:timedelta):

        if not isinstance(time_window, timedelta):
            raise TypeError("time_window is not timedelta")
        
        self.indices = indices
        self.time_window = time_window
        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
        self.demand = pd.read_csv((Path('.') / 'data' / 'processed' / 'demand.csv'), index_col=0)
        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time'])
        self.actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])

        self.demand.index = pd.to_datetime(self.demand.index, format='%Y-%m-%d %H:%M')
        self.locations.index = pd.MultiIndex.from_frame(self.locations.loc[:,['Time', 'Vehicle_Number_Plate']])
        self.actions = pd.MultiIndex.from_frame(self.actions.loc[:,['Time', 'Vehicle_Number_Plate']])
        self.locations.drop(labels=['Time', 'Vehicle_Number_Plate'], axis=1, inplace=True)

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        dem = self.demand.loc[self.indices[idx][0]]
        loc = self.locations.loc[self.indices[idx]]
        return torch.hstack(
            (torch.tensor(self.indices[idx][0].month), 
            torch.tensor(self.indices[idx][0].day), 
            torch.tensor(self.indices[idx][0].hour), 
            torch.tensor(dem.values), 
            torch.tensor(loc.values)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = torch.tensor(self.indices[idx] in self.actions).long()
        return s, a


class CarDataset_s2(Dataset):
    def __init__(self, indices, time_window:timedelta):

        if not isinstance(time_window, timedelta):
            raise TypeError("time_window is not timedelta")
        
        self.indices = indices
        self.time_window = time_window
        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
        self.demand = pd.read_csv((Path('.') / 'data' / 'processed' / 'demand.csv'), index_col=0)
        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time'])
        self.actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])

        self.demand.index = pd.to_datetime(self.demand.index, format='%Y-%m-%d %H:%M')
        self.locations.index = pd.MultiIndex.from_frame(self.locations.loc[:,['Time', 'Vehicle_Number_Plate']])
        self.actions.index = pd.MultiIndex.from_frame(self.actions.loc[:,['Time', 'Vehicle_Number_Plate']])
        self.locations.drop(labels=['Time', 'Vehicle_Number_Plate'], axis=1, inplace=True)
        self.actions.drop(labels=['Time', 'Vehicle_Number_Plate'], axis=1, inplace=True)
        self.actions.sort_index(inplace=True)

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        dem = self.demand.loc[self.indices[idx][0]]
        loc = self.locations.loc[self.indices[idx]]
        return torch.hstack(
            (torch.tensor(self.indices[idx][0].month), 
            torch.tensor(self.indices[idx][0].day), 
            torch.tensor(self.indices[idx][0].hour), 
            torch.tensor(dem.values), 
            torch.tensor(loc.values)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = torch.tensor(self.actions.loc[self.indices[idx]].values).squeeze()
        return s, a
