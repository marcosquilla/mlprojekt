from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#TODO: Add information about where specific car models are.

class CarDataset_s1(Dataset): 
    def __init__(self, indices):
        
        self.Mindices = pd.MultiIndex.from_frame(indices)
        self.Tindices = indices.values.tolist()
        self.indices = indices.index
        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)

        self.demand = pd.read_csv((Path('.') / 'data' / 'processed' / 'demand.csv'), index_col=0) 
        self.demand.index = pd.to_datetime(self.demand.index, format='%Y-%m-%d %H:%M')
        self.demand = self.demand.to_dict('index') # Convert for faster indexing

        self.actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])    
        self.actions = pd.MultiIndex.from_frame(self.actions.loc[:,['Time', 'Vehicle_Number_Plate']]) # MultiIndex faster

        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time']).drop(labels='Vehicle_Number_Plate', axis=1)
        self.locations[self.locations.columns[self.locations.dtypes=='int64']] = self.locations[self.locations.columns[self.locations.dtypes=='int64']].astype(np.uint8) # Cast types for lower RAM usage
        self.vehicle_counts = self.locations.groupby('Time')[self.locations.columns[self.locations.columns.str.contains('Zone')]].sum()
        self.vehicle_counts = self.vehicle_counts.to_dict('index') # Convert for faster indexing
        self.locations.drop(labels=['Time'], axis=1, inplace=True)
        self.locations = self.locations.values # Convert for faster indexing

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        dem = tuple(self.demand[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        count = tuple(self.vehicle_counts[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        loc = self.locations[self.indices[idx]] # Use list for faster indexing
        return torch.hstack(
            (torch.tensor(self.Tindices[idx][0].month), 
            torch.tensor(self.Tindices[idx][0].day), 
            torch.tensor(self.Tindices[idx][0].hour), 
            torch.tensor(dem),
            torch.tensor(count), 
            torch.tensor(loc)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = torch.tensor(self.Mindices[idx] in self.actions).long() # Faster to check with MultiIndex
        return s, a

class CarDataset_s2(Dataset): 
    def __init__(self, indices):

        self.Tindices = indices.values.tolist()
        self.indices = indices.index
        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)

        self.demand = pd.read_csv((Path('.') / 'data' / 'processed' / 'demand.csv'), index_col=0)
        self.demand.index = pd.to_datetime(self.demand.index, format='%Y-%m-%d %H:%M')
        self.demand = self.demand.to_dict('index') # Convert for faster indexing

        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time'])
        self.locations[self.locations.columns[self.locations.dtypes=='int64']] = self.locations[self.locations.columns[self.locations.dtypes=='int64']].astype(np.uint8) # Cast types for lower RAM usage
        self.actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])
        self.actions[self.actions.columns[self.actions.dtypes=='int64']] = self.actions[self.actions.columns[self.actions.dtypes=='int64']].astype(np.uint8) # Cast types for lower RAM usage

        self.vehicle_counts = self.locations.groupby('Time')[self.locations.columns[self.locations.columns.str.contains('Zone')]].sum()
        self.vehicle_counts = self.vehicle_counts.to_dict('index') # Convert for faster indexing
        self.locations = pd.merge(self.locations, self.actions.loc[:,['Time', 'Vehicle_Number_Plate']], how='inner', on=['Time', 'Vehicle_Number_Plate']) # Remove useless records
        self.locations.index = pd.MultiIndex.from_frame(self.locations.loc[:,['Time', 'Vehicle_Number_Plate']])
        self.locations.drop(labels=['Time', 'Vehicle_Number_Plate'], axis=1, inplace=True)
        self.actions.drop(labels=['Time', 'Vehicle_Number_Plate'], axis=1, inplace=True)
        self.actions = self.actions.values

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        dem = tuple(self.demand[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        count = tuple(self.vehicle_counts[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        loc = self.locations.loc[self.Tindices[idx][0], self.Tindices[idx][1]].values
        return torch.hstack(
            (torch.tensor(self.Tindices[idx][0].month), 
            torch.tensor(self.Tindices[idx][0].day), 
            torch.tensor(self.Tindices[idx][0].hour), 
            torch.tensor(dem), 
            torch.tensor(count),
            torch.tensor(loc)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = torch.tensor(self.actions[self.indices[idx]])
        return s, a

class DayDataset_s1(Dataset): 
    def __init__(self, indices):
        
        self.Mindices = pd.MultiIndex.from_frame(indices)
        self.Tindices = indices.values.tolist()
        self.indices = indices.index
        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)

        self.demand = pd.read_csv((Path('.') / 'data' / 'processed' / 'demand.csv'), index_col=0) 
        self.demand.index = pd.to_datetime(self.demand.index, format='%Y-%m-%d %H:%M')
        self.demand = self.demand.to_dict('index') # Convert for faster indexing

        self.actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])    
        self.actions = pd.MultiIndex.from_frame(self.actions.loc[:,['Time', 'Vehicle_Number_Plate']]) # MultiIndex faster

        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time']).drop(labels='Vehicle_Number_Plate', axis=1)
        self.locations[self.locations.columns[self.locations.dtypes=='int64']] = self.locations[self.locations.columns[self.locations.dtypes=='int64']].astype(np.uint8) # Cast types for lower RAM usage
        self.vehicle_counts = self.locations.groupby('Time')[self.locations.columns[self.locations.columns.str.contains('Zone')]].sum()
        self.vehicle_counts = self.vehicle_counts.to_dict('index') # Convert for faster indexing
        self.locations.drop(labels=['Time'], axis=1, inplace=True)
        self.locations = self.locations.values # Convert for faster indexing

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        dem = tuple(self.demand[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        count = tuple(self.vehicle_counts[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        loc = self.locations[self.indices[idx]] # Use list for faster indexing
        return torch.hstack(
            (torch.tensor(self.Tindices[idx][0].month), 
            torch.tensor(self.Tindices[idx][0].day), 
            torch.tensor(self.Tindices[idx][0].hour), 
            torch.tensor(dem),
            torch.tensor(count), 
            torch.tensor(loc)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = torch.tensor(self.Mindices[idx] in self.actions).long() # Faster to check with MultiIndex
        return s, a
