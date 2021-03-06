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
from src.data.datasets import AreaDataset_s1, AreaDataset_s2, DayDataset_s1

class AreaDataModule(pl.LightningDataModule):
    def __init__(self, s, lstm, batch_size:int=16, time_step:timedelta=timedelta(minutes=30), time_window:timedelta=timedelta(minutes=30),
    time_start=datetime(2020, 2, 2, 0, 0, 0), time_end=datetime(2021, 5, 3, 23, 59, 59), 
    test_size=0.2, val_size=0.1, shuffle=False, num_workers=0, n_zones=20):
        super().__init__()

        assert s == 'stage_1' or s == 'stage_2', 'Unknown stage'
        if not isinstance(time_step, timedelta):
            raise TypeError("time_step is not timedelta")
        if not isinstance(time_start, datetime):
            raise TypeError("time_start is not datetime")
        if not isinstance(time_end, datetime):
            raise TypeError("time_end is not datetime")

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.time_window = time_step
        self.time_step = time_step
        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime)
        self.n_zones = n_zones
        self.prepare_data()

        if s == 'stage_1':
            if lstm:
                self.indices = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), usecols=['Time', 'Virtual_Zone_Name'], parse_dates=['Time'])[['Time', 'Virtual_Zone_Name']]
                self.indices['Time'] = self.indices['Time'].dt.date
                self.indices.drop_duplicates(keep='first', inplace=True)
                self.dataset = DayDataset_s1
            else:
                self.indices = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), usecols=['Time', 'Virtual_Zone_Name'], parse_dates=['Time'])[['Time', 'Virtual_Zone_Name']]
                self.dataset = AreaDataset_s1
        elif s == 'stage_2':
            self.indices = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), usecols=['Time', 'Virtual_Start_Zone_Name'], parse_dates=['Time'])[['Time', 'Virtual_Start_Zone_Name']]
            self.dataset = AreaDataset_s2
        
        self.tv_idx, self.test_idx = train_test_split(self.indices, test_size=test_size, shuffle=shuffle)
        self.train_idx, self.val_idx = train_test_split(self.tv_idx, test_size=val_size/(1-test_size), shuffle=shuffle)


    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_data = self.dataset(self.train_idx, self.time_step)
            self.val_data = self.dataset(self.val_idx, self.time_step)
        if stage in (None, "test"):
            self.test_data = self.dataset(self.test_idx, self.time_step)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def prepare_data(self, rental_folder:str='SN rentals', open_folder:str='SN App requests', optimise=False, opt_zones=False):
        # Limits of modelling
        swlat=55.4355410101663
        swlon=12.140848911388979
        nelat=56.06417055142977
        nelon=12.688363746232875
        self.wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method
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
                    n_zones = self.n_zones

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
            tqdm.pandas()
            self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'))
            openings = openings.progress_apply(self.dr2a, axis=1)
            openings.to_csv(Path('.') / 'data' / 'interim' / 'openings.csv', index=False)

        if not ((Path('.') / 'data' / 'processed' / 'locations.csv').is_file() and
                (Path('.') / 'data' / 'processed' / 'actions.csv').is_file()):
            print('Creating locations and actions datasets')
            self.rental = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), parse_dates=['Start_Datetime_Local', 'End_Datetime_Local'], low_memory=False)
            self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'))
            self.rental = pd.get_dummies(self.rental, columns=['Vehicle_Model'])
            self.rental.rename(columns={'Virtual_End_Zone_Name': 'Virtual_Zone_Name'}, inplace=True) # Rename

            locations = pd.concat([self.vehicle_locations(i) for i, _ in enumerate(tqdm(self.timepoints))])
            locations.iloc[:,:-1] = locations.iloc[:,:-1].astype(int)
            locations.to_csv(Path('.') / 'data' / 'processed' / 'locations.csv', index=False)

            actions = pd.concat([self.actions(i) for i,_ in enumerate(tqdm(self.timepoints))])
            actions = pd.merge(actions, locations, how='inner', right_on=['Time', 'Virtual_Zone_Name'], left_on=['Time', 'Virtual_Start_Zone_Name'], suffixes=(None,'_slet'))
            actions = actions.loc[:,actions.columns[~actions.columns.str.contains('_slet')]]
            actions.to_csv(Path('.') / 'data' / 'processed' / 'actions.csv', index=False)

            del self.rental, locations, actions
        
        if not (Path('.') / 'data' / 'processed' / 'demand.csv').is_file():
            print('Creating demand dataset')
            self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
            self.openings = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'), parse_dates=['Created_Datetime_Local'])

            pd.DataFrame(pd.Series('Time').append(pd.Series([i for i in range(len(self.area_centers))]))).T.to_csv(Path('.') / 'data' / 'processed' / 'demand.csv', header=False, index=False) # Create csv with header
            l = len(self.timepoints)
            n_splits = 10 # Number of splits for data saving. Increase for lower RAM usage
            for p in tqdm(range(n_splits)):
                demand_dist = pd.concat([pd.DataFrame(self.demand(i)).T for i in tqdm(range(int(l*p/n_splits), int(l*(p+1)/n_splits)), leave=False)])
                demand_dist.set_index('Time', inplace=True)
                demand_dist.to_csv(Path('.') / 'data' / 'processed' / 'demand.csv', index=True, header=False, mode='a')
                del demand_dist

            del self.openings, self.area_centers

    def coords_to_areas(self, target):
        # Auxiliary method for demand. Calculate to which area an opening's coordinates (target) "belong to".
        _,_,dists = self.wgs84_geod.inv(
            self.area_centers['GPS_Longitude'], self.area_centers['GPS_Latitude'],
            np.full(len(self.area_centers),target['GPS_Longitude']), np.full(len(self.area_centers),target['GPS_Latitude']))
        return pd.Series(1 - dists / sum(dists)) / (len(dists) - 1) # Percentage of how much an opening belongs to each area

    def demand(self, idx):
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
        loc = self.rental[self.rental['End_Datetime_Local'] <= self.timepoints[idx]]
        loc = loc.drop_duplicates(subset='Vehicle_Number_Plate', keep='last') # Keep the last location
        current_trips = self.rental[(self.rental['Start_Datetime_Local'] < self.timepoints[idx]) & (self.rental['End_Datetime_Local'] >= self.timepoints[idx])] # Cars in use
        loc = loc[~loc['Vehicle_Number_Plate'].isin(current_trips['Vehicle_Number_Plate'])] # Filter out cars in use
        loc = loc.groupby('Virtual_Zone_Name')[loc.columns[loc.columns.str.contains('Vehicle_Model')].values].sum().reset_index()
        for mz in np.arange(len(self.area_centers))[~(np.isin(np.arange(len(self.area_centers)),loc['Virtual_Zone_Name']))]: # Add zones with 0 cars
            loc.loc[mz-0.5] = np.zeros(len(loc.columns))
            loc.loc[mz-0.5, 'Virtual_Zone_Name'] = mz
            loc = loc.sort_index().reset_index(drop=True)
        loc['Time'] = self.timepoints[idx]
        return loc

    def actions(self, idx):
        a = self.rental[(self.rental['Servicedrive_YN']==1) &
                        (self.rental['Start_Datetime_Local'] > self.timepoints[idx]) &
                        (self.rental['Start_Datetime_Local'] <= self.timepoints[idx]+self.time_window)]
        a = a[a['Virtual_Start_Zone_Name'] != a['Virtual_Zone_Name']].iloc[:1]
        a = a.groupby('Virtual_Start_Zone_Name')[['Virtual_Zone_Name', *a.columns[a.columns.str.contains('Vehicle_Model')].values.tolist()]].sum().reset_index()
        a['Time'] = self.timepoints[idx]
        return a

    def dr2a(self, row): # Distance from request to area
        _,_,dists = self.wgs84_geod.inv(
                self.area_centers['GPS_Longitude'], self.area_centers['GPS_Latitude'],
                np.full(len(self.area_centers),row['GPS_Longitude']), np.full(len(self.area_centers),row['GPS_Latitude']))
        ca = np.argmin(dists)
        row['Area'] = ca
        row['Distance'] = dists[ca]
        return row
        