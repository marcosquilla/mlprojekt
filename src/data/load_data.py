# pylint: disable=E1101

from pathlib import Path
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import combinations, product
from pyproj import Geod
from tqdm import tqdm

#TODO: Time_window vs time_delta tuning

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int=16, time_step:timedelta=timedelta(minutes=30), time_window:timedelta=timedelta(minutes=30),
    time_start=datetime(2020, 2, 1, 0, 56, 26), time_end=datetime(2021, 5, 3, 23, 59, 55), 
    test_size=0.2, val_size=0, shuffle_time=False, num_workers=0, n_actions=5):
        super().__init__()

        if not isinstance(batch_size, int):
            raise TypeError("batch_size has to be integer")
        if not isinstance(time_step, timedelta):
            raise TypeError("time_step is not timedelta")
        if not isinstance(time_start, datetime):
            raise TypeError("time_start is not datetime")
        if not isinstance(time_end, datetime):
            raise TypeError("time_end is not datetime")

        self.num_workers = num_workers
        self.n_actions=n_actions
        self.batch_size = batch_size
        self.time_window = time_window
        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime) # Link between indexes and datetimes
        self.train_idx, self.test_idx = train_test_split(self.timepoints, test_size=test_size, shuffle=shuffle_time)
        if val_size>0:
            self.train_idx , self.val_idx = train_test_split(self.train_idx, test_size=val_size/(1-test_size), shuffle=shuffle_time)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_data = sarDataset(self.train_idx, time_window=self.time_window, n_actions=self.n_actions)
        if stage in (None, "validate"):
            self.val_data = sarDataset(self.val_idx, time_window=self.time_window, n_actions=self.n_actions)
        if stage in (None, "test"):
            self.test_data = sarDataset(self.test_idx, time_window=self.time_window, n_actions=self.n_actions)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def prepare_data(self, rental_folder:str='SN rentals', open_folder:str='SN App requests', optimise=False, opt_zones=False, n_zones=50):
        #TODO: Include data download.
        # Limits of modelling
        swlat=55.4355410101663
        swlon=12.140848911388979
        nelat=56.06417055142977
        nelon=12.688363746232875
        # Takes raw files, concatenates them, selects useful columns and saved into a single file.
        if not ((Path('.') / 'data' / 'processed' / 'rental.csv').is_file() and 
                (Path('.') / 'data' / 'processed' / 'areas.csv').is_file()):
            if not (Path('.') / 'data' / 'interim' / 'rental.csv').is_file():
                rent_files = glob.glob(str((Path('.') / 'data' / 'raw' / rental_folder / '*.xlsx')))
                rent_dfs = [pd.read_excel(f, skiprows=[0,1]) for f in rent_files]
                rental = pd.concat(rent_dfs,ignore_index=True)
                rental = rental[rental['[Partner_Rental_ID]']!='[Partner_Rental_ID]']
                rental.columns = rental.columns.str.replace("[","", regex=False)
                rental.columns = rental.columns.str.replace("]","", regex=False)
                rental = rental.loc[:, ['Vehicle_Number_Plate', 'Vehicle_Engine_Type',
                'Vehicle_Model', 'Revenue_Net',
                'Start_Datetime_Local', 'End_Datetime_Local',
                'Start_GPS_Latitude', 'Start_GPS_Longitude',
                'End_GPS_Latitude', 'End_GPS_Longitude', 'Package_Description',
                'Operation_State_Name_Before', 'Operation_State_Name_After', 'Reservation_YN',
                'Prebooking_YN', 'Servicedrive_YN', 'Start_Zone_Name', 'End_Zone_Name']]
                # Filter rentals outside of analysis zone
                rental = rental[
                    (rental['Start_GPS_Latitude'] > swlat) & (rental['Start_GPS_Latitude'] < nelat) & 
                    (rental['Start_GPS_Longitude'] > swlon) & (rental['Start_GPS_Longitude'] < nelon) &
                    (rental['End_GPS_Latitude'] > swlat) & (rental['End_GPS_Latitude'] < nelat) & 
                    (rental['End_GPS_Longitude'] > swlon) & (rental['End_GPS_Longitude'] < nelon)]
                rental.to_csv(Path('.') / 'data' / 'interim' / 'rental.csv', index=False)
            else:
                rental = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), low_memory=False)

            # Virtual area creation with KMeans and assignment to all rentals
            if optimise:
                print('Finding optimal amount of virtual zones')
                X = rental.loc[:,['Start_GPS_Latitude','Start_GPS_Longitude']].sample(frac=0.05) # Using only a fraction of rental to train quicker
                scores = []
                for n in tqdm(range(20, 200, 20)):
                    km = KMeans(n_clusters=n).fit(X)
                    scores.append([n, silhouette_score(X, km.labels_)])
                scores = pd.DataFrame(scores)
                n_zones = int(scores.iloc[scores.iloc[:,1].argmax(),0]) # Pick n_zones with highes silhouette_score
                scores.to_csv(Path('.') / 'reports' /'virtual_area_opt.csv', index=False)
            else:
                if opt_zones:
                    scores = pd.read_csv(Path('.') / 'reports' / 'virtual_area_opt.csv', index_col='0')
                    n_zones = int(scores.iloc[np.argmax(scores)].name)
                else:
                    n_zones = n_zones

            # Determine the correct zones using the whole dataset
            km = KMeans(n_clusters=n_zones, verbose=1).fit(rental.loc[:,['Start_GPS_Latitude','Start_GPS_Longitude']])
            areas = pd.DataFrame(km.cluster_centers_, columns=['GPS_Latitude','GPS_Longitude'])
            rental.loc[:,'Virtual_Start_Zone_Name'] = km.labels_
            rental.loc[:,'Virtual_End_Zone_Name'] = [label for label in km.predict(rental.loc[:,['End_GPS_Latitude','End_GPS_Longitude']])]

            rental.to_csv(Path('.') / 'data' / 'processed' / 'rental.csv', index=False)
            areas.to_csv(Path('.') / 'data' / 'processed' / 'areas.csv')


        if not (Path('.') / 'data' / 'processed' / 'openings.csv').is_file():
            open_files = glob.glob(str((Path('.') / 'data' / 'raw' / open_folder / '*.csv')))
            open_dfs = [pd.read_csv(f) for f in open_files]
            openings = pd.concat(open_dfs,ignore_index=True)
            openings = openings[openings['Source_Location_ID']!='Source_Location_ID']
            openings = openings.loc[:, ['Created_Datetime_Local', 'Platform', 'GPS_Longitude', 'GPS_Latitude']]
            openings = openings[
                (openings['GPS_Latitude'] > swlat) & (openings['GPS_Latitude'] < nelat) & 
                (openings['GPS_Longitude'] > swlon) & (openings['GPS_Longitude'] < nelon)]
            openings.to_csv(Path('.') / 'data' / 'processed' / 'openings.csv', index=False)

class sarDataset(Dataset):
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

        self.openings = pd.read_csv((Path('.') / 'data' / 'processed' / 'openings.csv'))
        self.openings.loc[:,'Created_Datetime_Local'] = pd.to_datetime(self.openings['Created_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.openings = pd.get_dummies(self.openings, columns=['Platform'], drop_first=True)
    
        self.rental = pd.read_csv((Path('.') / 'data' / 'processed' / 'rental.csv'), low_memory=False)
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
        a[:ad.shape[0]] = ad
        return torch.from_numpy(a.reshape(-1))

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

if __name__ == "__main__":
    dm = DataModule(batch_size=1)
    dm.setup(stage='fit')
    s, a, *_= next(iter(dm.train_dataloader()))
    print(s.shape[1])