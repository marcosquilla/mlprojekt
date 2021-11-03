from pathlib import Path
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyproj import Geod
from tqdm import tqdm
from src.data.datasets import FleetDataset, CarDataset

class FleetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int=16, time_step:timedelta=timedelta(minutes=30), time_window:timedelta=timedelta(minutes=30),
    time_start=datetime(2020, 2, 1, 0, 56, 26), time_end=datetime(2021, 5, 3, 23, 59, 55), 
    test_size=0.2, shuffle_time=False, num_workers=0, n_actions=5):
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

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_data = FleetDataset(self.train_idx, time_window=self.time_window, n_actions=self.n_actions)
        if stage in (None, "test"):
            self.test_data = FleetDataset(self.test_idx, time_window=self.time_window, n_actions=self.n_actions)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def prepare_data(self, rental_folder:str='SN rentals', open_folder:str='SN App requests', optimise=False, opt_zones=False, n_zones=50):
        # Limits of modelling
        swlat=55.4355410101663
        swlon=12.140848911388979
        nelat=56.06417055142977
        nelon=12.688363746232875
        # Takes raw files, concatenates them, selects useful columns and saved into a single file.
        if not ((Path('.') / 'data' / 'interim' / 'rental.csv').is_file() and 
                (Path('.') / 'data' / 'processed' / 'areas.csv').is_file()):
            if not (Path('.') / 'data' / 'interim' / 'rental_join.csv').is_file():
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
                rental.to_csv(Path('.') / 'data' / 'interim' / 'rental_join.csv', index=False)
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

class CarDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int=16, time_step:timedelta=timedelta(minutes=30), time_window:timedelta=timedelta(minutes=30),
    time_start=datetime(2020, 2, 1, 0, 56, 26), time_end=datetime(2021, 5, 3, 23, 59, 55), 
    test_size=0.2, shuffle=False, num_workers=0):
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
        self.batch_size = batch_size
        self.time_window = time_window
        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime)
        self.prepare_data()
        self.indices = pd.MultiIndex.from_frame(pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), usecols=['Time', 'Vehicle_Number_Plate'], parse_dates=['Time'])).reorder_levels([1,0])

        self.train_idx, self.test_idx = train_test_split(self.indices, test_size=test_size, shuffle=shuffle)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_data = CarDataset(self.train_idx, time_window=self.time_window)
        if stage in (None, "test"):
            self.test_data = CarDataset(self.test_idx, time_window=self.time_window)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def prepare_data(self, rental_folder:str='SN rentals', open_folder:str='SN App requests', optimise=False, opt_zones=False, n_zones=50):
        # Limits of modelling
        swlat=55.4355410101663
        swlon=12.140848911388979
        nelat=56.06417055142977
        nelon=12.688363746232875
        # Takes raw files, concatenates them, selects useful columns and saved into a single file.
        if not ((Path('.') / 'data' / 'interim' / 'rental.csv').is_file() and 
                (Path('.') / 'data' / 'processed' / 'areas.csv').is_file()):
            if not (Path('.') / 'data' / 'interim' / 'rental_join.csv').is_file():
                print('Joining rental data')
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
                rental.to_csv(Path('.') / 'data' / 'interim' / 'rental_join.csv', index=False)
            else:
                rental = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental_join.csv'), low_memory=False)

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

            rental.to_csv(Path('.') / 'data' / 'interim' / 'rental.csv', index=False)
            areas.to_csv(Path('.') / 'data' / 'processed' / 'areas.csv')

        if not (Path('.') / 'data' / 'interim' / 'openings.csv').is_file():
            print('Joining openings data')
            open_files = glob.glob(str((Path('.') / 'data' / 'raw' / open_folder / '*.csv')))
            open_dfs = [pd.read_csv(f) for f in open_files]
            openings = pd.concat(open_dfs,ignore_index=True)
            openings = openings[openings['Source_Location_ID']!='Source_Location_ID']
            openings = openings.loc[:, ['Created_Datetime_Local', 'Platform', 'GPS_Longitude', 'GPS_Latitude']]
            openings = openings[
                (openings['GPS_Latitude'] > swlat) & (openings['GPS_Latitude'] < nelat) & 
                (openings['GPS_Longitude'] > swlon) & (openings['GPS_Longitude'] < nelon)]
            openings.to_csv(Path('.') / 'data' / 'interim' / 'openings.csv', index=False)

        if not ((Path('.') / 'data' / 'processed' / 'locations.csv').is_file() and
                (Path('.') / 'data' / 'processed' / 'actions.csv').is_file()):
            print('Creating locations and actions datasets')
            self.rental = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), parse_dates=['Start_Datetime_Local', 'End_Datetime_Local'],low_memory=False)
            self.rental = pd.get_dummies(self.rental, columns=['Vehicle_Model'])
            self.rental.rename(columns={'Virtual_End_Zone_Name': 'Virtual_Zone_Name'}, inplace=True) # Rename
            self.rental['VZE_ori'] = self.rental['Virtual_Zone_Name'] # Keep original column
            self.rental = pd.get_dummies(self.rental, columns=['Virtual_Zone_Name'])
            cols_loc = np.append(self.rental.columns[(self.rental.columns.str.contains('Plate') | self.rental.columns.str.contains('Virtual_Zone_Name_') | self.rental.columns.str.contains('Vehicle_Model'))].values, 'Time')
            cols_act = np.append(self.rental.columns[(self.rental.columns.str.contains('Plate') | self.rental.columns.str.contains('Virtual_Zone_Name_'))].values, 'Time')

            locations = pd.concat([self.vehicle_locations(i).loc[:,cols_loc] for i, _ in enumerate(tqdm(self.timepoints))])
            locations.to_csv(Path('.') / 'data' / 'processed' / 'locations.csv', index=False)

            actions = pd.concat([self.actions(i).loc[:,cols_act] for i, _ in enumerate(tqdm(self.timepoints))])
            actions.to_csv(Path('.') / 'data' / 'processed' / 'actions.csv', index=False)

            del self.rental, actions, locations
        
        if not (Path('.') / 'data' / 'processed' / 'demand.csv').is_file():
            print('Creating demand dataset')
            self.wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method
            self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
            self.openings = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'), parse_dates=['Created_Datetime_Local'])

            pd.DataFrame(pd.Series('Time').append(pd.Series([i for i in range(50)]))).T.to_csv(Path('.') / 'data' / 'processed' / 'demand.csv', header=False, index=False) # Create csv with header
            l = len(self.timepoints)
            n_splits = 10 # Number of splits for data saving. Increase for lower RAM usage
            for p in tqdm(range(n_splits)):
                demand_dist = pd.concat([pd.DataFrame(self.demand(i)).T for i in tqdm(range(int(l*p/n_splits), int(l*(p+1)/n_splits)), leave=False)])
                demand_dist.set_index('Time', inplace=True)
                demand_dist.to_csv(Path('.') / 'data' / 'processed' / 'demand.csv', index=True, header=False, mode='a')
                del demand_dist

            del self.openings, self.area_centers, self.wgs84_geod

    def coords_to_areas(self, target):
        # Auxiliary method for demand. Calculate to which area an opening's coordinates (target) "belong to".
        _,_,dists = self.wgs84_geod.inv(
            self.area_centers['GPS_Latitude'], self.area_centers['GPS_Longitude'],
            np.full(len(self.area_centers),target['GPS_Latitude']), np.full(len(self.area_centers),target['GPS_Longitude']))
        return pd.Series(1 - dists / sum(dists)) / (len(dists) - 1) # Percentage of how much an opening belongs to each area

    def demand(self, idx):
        # Auxiliary method for __getitem__. Returns the demand of all areas at some point in time.
        dem = self.openings[(self.openings['Created_Datetime_Local'] > self.timepoints[idx]-self.time_window) &
        (self.openings['Created_Datetime_Local'] <= self.timepoints[idx])].copy()
        if len(dem) == 0:
            dem = pd.Series(data=0, index=np.arange(len(self.area_centers)))
            dem['Time'] = self.timepoints[idx]
            return dem
        else:
            dem.loc[:,self.area_centers.index.values] = 0 # Create columns with area names
            dem.loc[:,self.area_centers.index.values] = dem.apply(lambda x: self.coords_to_areas(x), axis=1) # Apply function to all openings
            dem = dem.loc[:,self.area_centers.index].sum(axis=0) # Aggregate demand in the time window over areas (.loc to remove gps coords and platform). Sum of demand equals to amount of app openings
            dem['Time'] = self.timepoints[idx]
            return dem

    def vehicle_locations(self, idx):
        # Auxiliary method for __getitem__.
        loc = self.rental[self.rental['End_Datetime_Local'] <= self.timepoints[idx]]
        loc = loc.drop_duplicates(subset='Vehicle_Number_Plate', keep='last') # Keep the last location
        current_trips = self.rental[(self.rental['Start_Datetime_Local'] <= self.timepoints[idx]) & (self.rental['End_Datetime_Local'] > self.timepoints[idx])] # Cars in use
        loc = loc[~loc['Vehicle_Number_Plate'].isin(current_trips['Vehicle_Number_Plate'])] # Filter out cars in use
        loc['Time'] = self.timepoints[idx]
        return loc

    def actions(self, idx):
        # Auxiliary method for __getitem__. Calculates actions
        a = self.rental[(self.rental['Servicedrive_YN']==1) &
                        (self.rental['Start_Datetime_Local'] >= self.timepoints[idx]-self.time_window) &
                        (self.rental['End_Datetime_Local'] < self.timepoints[idx])]
        a = a[a['Virtual_Start_Zone_Name'] != a['VZE_ori']].iloc[:1]
        a['Time'] = self.timepoints[idx]
        return a
