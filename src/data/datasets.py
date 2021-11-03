from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pyproj import Geod

class FleetDataset(Dataset):
    def __init__(self, timepoints, time_window:timedelta, n_actions:int=5):

        if not isinstance(timepoints, (list, tuple, set, np.ndarray, pd.Series)):
            raise TypeError("timepoints is not list-like")
        if not all(isinstance(n, datetime) for n in timepoints):
            raise TypeError("timepoints array has to contain datetime objects")
        if not isinstance(time_window, timedelta):
            raise TypeError("time_window is not timedelta")
        
        self.timepoint = timepoints
        self.n_actions = n_actions

        self.time_window = time_window
        self.wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method

        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)

        self.openings = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'))
        self.openings.loc[:,'Created_Datetime_Local'] = pd.to_datetime(self.openings['Created_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.openings = pd.get_dummies(self.openings, columns=['Platform'], drop_first=True)
    
        self.rental = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), low_memory=False)
        self.rental.loc[:,'Start_Datetime_Local'] = pd.to_datetime(self.rental['Start_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.rental.loc[:,'End_Datetime_Local'] = pd.to_datetime(self.rental['End_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.rental = pd.get_dummies(self.rental, columns=['Vehicle_Engine_Type'], drop_first=True)
        self.rental = pd.get_dummies(self.rental, columns=['Vehicle_Model'])
        one_hot_zones = pd.get_dummies(self.rental.loc[:,['Virtual_Start_Zone_Name', 'Virtual_End_Zone_Name']], columns=['Virtual_Start_Zone_Name', 'Virtual_End_Zone_Name'])
        self.rental = pd.concat([self.rental, one_hot_zones], axis=1)
        
        self.vehicles = self.rental.columns[self.rental.columns.str.contains('Vehicle_Model')] # Get names of vehicles

    def coords_to_areas(self, target):
        # Auxiliary method for demand. Calculate to which area an opening's coordinates (target) "belong to".
        _,_,dists = self.wgs84_geod.inv(
            self.area_centers['GPS_Latitude'], self.area_centers['GPS_Longitude'],
            np.full(len(self.area_centers),target['GPS_Latitude']), np.full(len(self.area_centers),target['GPS_Longitude']))
        return pd.Series(1 - dists / sum(dists)) / (len(dists) - 1) # Percentage of how much an opening belongs to each area
        
    def demand(self, idx):
        # Auxiliary method for __getitem__. Uses array timepoint as a index. Returns the demand of all areas at some point in time.
        dem = self.openings[(self.openings['Created_Datetime_Local'] > self.timepoint[idx]-self.time_window) &
        (self.openings['Created_Datetime_Local'] <= self.timepoint[idx])].copy()
        if len(dem) == 0:
            return torch.zeros(len(self.area_centers))
        else:
            dem.loc[:,self.area_centers.index.values] = 0 # Create columns with area names
            dem.loc[:,self.area_centers.index.values] = dem.apply(lambda x: self.coords_to_areas(x), axis=1) # Apply function to all openings
            return torch.tensor(dem.loc[:,self.area_centers.index].sum(axis=0).values) # Aggregate demand in the time window over areas (.loc to remove gps coords and platform). Sum of demand equals to amount of app openings
    
    def vehicle_locations(self, idx):
        # Auxiliary method for __getitem__. Uses array timepoint as a index.
        loc = self.rental[self.rental['End_Datetime_Local'] <= self.timepoint[idx]]
        loc = loc.drop_duplicates(subset='Vehicle_Number_Plate', keep='last') # Keep the last location
        current_trips = self.rental[(self.rental['Start_Datetime_Local'] <= self.timepoint[idx]) & (self.rental['End_Datetime_Local'] > self.timepoint[idx])] # Cars in use
        loc = loc[~loc['Vehicle_Number_Plate'].isin(current_trips['Vehicle_Number_Plate'])] # Filter out cars in use
        loc = loc.loc[:, ~loc.columns.str.contains('Start')].drop(columns=['End_Datetime_Local'], axis=1) # Drop unused columns
        loc = loc.groupby('Virtual_End_Zone_Name')[self.vehicles].sum() # Aggregate amount of cars
        missing_areas = pd.DataFrame(index=self.area_centers.index[~self.area_centers.index.isin(loc.index)], columns=loc.columns, data=0)
        loc = pd.melt(pd.concat([loc, missing_areas]), ignore_index=False) # Add missing areas and unpivot
        loc.index = loc.index.astype('str') + loc.variable # Join zone and vehicle model, necessary to sort
        return torch.tensor(loc.drop(labels='variable', axis=1).sort_index().values).squeeze() # Drop vehicle model (already in index) and sort

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        dem = self.demand(idx)
        loc = self.vehicle_locations(idx)
        return torch.hstack((torch.tensor(self.timepoint[idx].month), torch.tensor(self.timepoint[idx].day), torch.tensor(self.timepoint[idx].hour), dem, loc))

    def actions(self, idx):
        # Auxiliary method for __getitem__. Calculates actions
        ad = self.rental[(self.rental['Servicedrive_YN']==1) &
                        (self.rental['Start_Datetime_Local'] >= self.timepoint[idx]-self.time_window) &
                        (self.rental['End_Datetime_Local'] < self.timepoint[idx])]
        ad = ad[ad['Virtual_Start_Zone_Name'] != ad['Virtual_End_Zone_Name']].iloc[:,19:]
        ad = np.reshape(ad.to_numpy(), (-1, ad.shape[1]))[:self.n_actions]
        a = np.zeros((self.n_actions, ad.shape[1]), dtype=np.int8)
        a[:ad.shape[0]] = ad # Set actual movements
        a[ad.shape[0]:, [0, -2*len(self.area_centers), -len(self.area_centers)]] = 1 # Set redundant movements in the rest
        return torch.from_numpy(a)

    def revenue(self, idx):
        # Auxiliary method for __getitem__. Uses array timepoint as a index.
        trips_in_window = self.rental[(self.rental['Start_Datetime_Local'] >= self.timepoint[idx]-self.time_window) & (self.rental['End_Datetime_Local'] < self.timepoint[idx])]
        return torch.tensor(trips_in_window['Revenue_Net'].sum())

    def __len__(self):
        return len(self.timepoint)

    def __getitem__(self, idx):
        s = self.state(idx) # Returns position of cars in timepoint idx and demand between idx-timedelta and idx
        a = self.actions(idx) # Returns end position of cars due to service trips within idx-timedelta (only moved cars)
        s1 = self.state(idx+1) # Returns position of cars in timepoint idx+1 and demand between idx+1-timedelta and idx+1
        r = self.revenue(idx) # Returns total revenue between idx-timedelta and idx
        return s, a, s1, r

class CarDataset(Dataset):
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
        try:
            a = torch.tensor(self.actions.loc[self.indices[idx]].values).squeeze()
        except KeyError: # Car not relocated
            a = torch.zeros(len(self.area_centers), dtype=torch.int8)
            a[torch.argmax(s[-len(self.area_centers):])] = 1 # Move to current location
        return s, a
