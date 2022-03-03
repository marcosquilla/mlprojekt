from pathlib import Path
from collections import deque
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

class AreaDataset_s1(Dataset): 
    def __init__(self, indices, time_step):
        
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
        self.actions = pd.MultiIndex.from_frame(self.actions.loc[:,['Time', 'Virtual_Zone_Name']]) # MultiIndex faster

        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time'])
        self.locations[self.locations.columns[self.locations.dtypes=='int64']] = self.locations[self.locations.columns[self.locations.dtypes=='int64']].astype(np.uint8) # Cast types for lower RAM usage
        self.vehicle_counts = self.locations.drop(labels=self.locations.columns[self.locations.columns.str.contains('Model')], axis=1)
        self.vehicle_counts['C'] = self.locations.loc[:,self.locations.columns.str.contains('Model')].sum(axis=1)
        self.vehicle_counts = self.vehicle_counts.pivot(index='Time', columns='Virtual_Zone_Name', values='C').fillna(0).astype(np.uint8).to_dict('index')
        self.locations = pd.get_dummies(self.locations, columns=['Virtual_Zone_Name'])
        self.locations.drop(labels=['Time'], axis=1, inplace=True)
        self.locations = self.locations.values # Convert for faster indexing

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        day = torch.zeros(7)
        day[self.Tindices[idx][0].weekday()] = 1
        hour = torch.zeros(24)
        hour[self.Tindices[idx][0].hour] = 1
        dem = tuple(self.demand[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        count = tuple(self.vehicle_counts[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        loc = self.locations[self.indices[idx]] # Use list for faster indexing
        return torch.hstack(
            (day, 
            hour, 
            torch.tensor(dem),
            torch.tensor(count), 
            torch.tensor(loc)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = torch.tensor(self.Mindices[idx] in self.actions).long() # Faster to check with MultiIndex
        return s, a

class AreaDataset_s2(Dataset): 
    def __init__(self, indices, time_step):

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

        self.vehicle_counts = self.locations.drop(labels=self.locations.columns[self.locations.columns.str.contains('Model')], axis=1)
        self.vehicle_counts['C'] = self.locations.loc[:,self.locations.columns.str.contains('Model')].sum(axis=1)
        self.vehicle_counts = self.vehicle_counts.pivot(index='Time', columns='Virtual_Zone_Name', values='C').fillna(0).astype(np.uint8).to_dict('index')
        self.locations = pd.merge(self.locations, self.actions, how='inner', left_on=['Time', 'Virtual_Zone_Name'], right_on=['Time', 'Virtual_Start_Zone_Name']) # Remove useless records
        self.locations.index = pd.MultiIndex.from_frame(self.locations.loc[:,['Time', 'Virtual_Zone_Name_x']])
        self.locations.drop(labels=['Time', 'Virtual_Start_Zone_Name', *self.locations.columns[self.locations.columns.str.contains('_y')]], axis=1, inplace=True)
        self.locations = pd.get_dummies(self.locations, columns=['Virtual_Zone_Name_x'])
        self.actions.drop(labels=['Time', 'Virtual_Start_Zone_Name', *self.actions.columns[self.actions.columns.str.contains('Model')]], axis=1, inplace=True)
        self.actions = pd.get_dummies(self.actions, columns=['Virtual_Zone_Name'])
        self.actions = self.actions.values

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        day = torch.zeros(7)
        day[self.Tindices[idx][0].weekday()] = 1
        hour = torch.zeros(24)
        hour[self.Tindices[idx][0].hour] = 1
        dem = tuple(self.demand[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        count = tuple(self.vehicle_counts[self.Tindices[idx][0]].values()) # Use the list for faster indexing
        loc = self.locations.loc[self.Tindices[idx][0], self.Tindices[idx][1]].values
        return torch.hstack(
            (day,
            hour,
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
    def __init__(self, indices, time_step):
        
        self.time_step = time_step
        self.Mindices = pd.MultiIndex.from_frame(indices)
        self.Tindices = indices.values.tolist()
        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)

        self.demand = pd.read_csv((Path('.') / 'data' / 'processed' / 'demand.csv'), index_col=0) 
        self.demand.index = pd.to_datetime(self.demand.index, format='%Y-%m-%d %H:%M')
        self.demand.index = self.demand.index.date

        self.actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])
        self.actions = pd.MultiIndex.from_frame(self.actions.loc[:,['Time', 'Virtual_Zone_Name']]) # MultiIndex faster

        self.locations = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), parse_dates=['Time'])
        self.locations[self.locations.columns[self.locations.dtypes=='int64']] = self.locations[self.locations.columns[self.locations.dtypes=='int64']].astype(np.uint8) # Cast types for lower RAM usage
        self.vehicle_counts = self.locations.drop(labels=self.locations.columns[self.locations.columns.str.contains('Model')], axis=1)
        self.vehicle_counts['C'] = self.locations.loc[:,self.locations.columns.str.contains('Model')].sum(axis=1)
        self.vehicle_counts = self.vehicle_counts.pivot(index='Time', columns='Virtual_Zone_Name', values='C').fillna(0).astype(np.uint8)
        self.vehicle_counts.index = self.vehicle_counts.index.date
        self.locations['Time'] = self.locations['Time'].dt.date
        self.locations.index = pd.MultiIndex.from_frame(self.locations.loc[:,['Time', 'Virtual_Zone_Name']])
        self.locations.drop('Time', axis=1, inplace=True)
        self.locations = pd.get_dummies(self.locations, columns=['Virtual_Zone_Name']).sort_index()

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        day = torch.zeros((len(dem),7))
        day[:,self.Tindices[idx][0].weekday()] = 1
        dem = self.demand.loc[self.Tindices[idx][0]].values.tolist()
        count = self.vehicle_counts.loc[self.Tindices[idx][0]].values.tolist()
        loc = self.locations.loc[self.Tindices[idx][0],self.Tindices[idx][1]].values.tolist()
        return torch.hstack(
            (day,
            torch.tensor(dem),
            torch.tensor(count),
            torch.tensor(loc)))
    
    def action(self, idx):
        h = np.arange(self.Mindices[idx][0], self.Mindices[idx][0]+timedelta(days=1), self.time_step).astype(datetime)
        z = np.full_like(h, self.Mindices[idx][1])
        hz = tuple(map(tuple, np.vstack((h,z)).T))
        return torch.tensor([True if i in self.actions else False for i in hz]).long()

    def __len__(self):
        return len(self.Tindices)

    def __getitem__(self, idx):
        s = self.state(idx)
        a = self.action(idx)
        return s, a

class QDataset(IterableDataset):
    def __init__(self, buffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = min(sample_size, len(buffer))

    def __iter__(self):
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

class ReplayBuffer():
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, sample_size: int):
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )