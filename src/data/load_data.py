# pylint: disable=E1101

from pathlib import Path
from copy import deepcopy
import glob
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyproj import Geod
from tqdm import tqdm

#TODO: Check action times and regulate time_window and timepoint_delta

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int=16, time_step:timedelta=timedelta(hours=1), 
    time_start=datetime(2020, 2, 1, 0, 56, 26), time_end=datetime(2021, 5, 3, 23, 59, 55)):
        super().__init__()

        if not isinstance(batch_size, int):
            raise TypeError("batch_size has to be integer")
        if not isinstance(time_step, timedelta):
            raise TypeError("time_step is not timedelta")
        if not isinstance(time_start, datetime):
            raise TypeError("time_start is not datetime")
        if not isinstance(time_end, datetime):
            raise TypeError("time_end is not datetime")

        self.batch_size=batch_size
        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime) # Link between indexes and datetimes

    def setup(self, stage=None):
        #TODO: Include percentages as parameters.
        if stage in (None, "fit"):
            train_idx = self.timepoints[:int(len(self.timepoints)*0.7)]
            self.train_data = sarDataset(train_idx)
        if stage in (None, "validate"):
            val_idx = self.timepoints[int(len(self.timepoints)*0.7):int(len(self.timepoints)*0.8)]
            self.val_data = sarDataset(val_idx)
        if stage in (None, "test"):
            test_idx = self.timepoints[int(len(self.timepoints)*0.8):]
            self.test_data = sarDataset(test_idx)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def prepare_data(self, n_zones:int=300, rental_folder:str='SN rentals', open_folder:str='SN App requests', optimise=False):
        #TODO: Include data download.
        # Limits of modelling
        swlat=55.4355410101663
        swlon=12.140848911388979
        nelat=56.06417055142977
        nelon=12.688363746232875
        # Takes raw files, concatenates them, selects useful columns and saved into a single file.
        if not ((Path.cwd() / 'data' / 'processed' / 'rental.csv').is_file() and 
                (Path.cwd() / 'data' / 'processed' / 'areas.csv').is_file()):
            rent_files = glob.glob(str(Path.cwd() / 'data' / 'raw' / rental_folder / '*.xlsx'))
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

            # Virtual area creation with KMeans and assignment to all rentals
            if optimise:
                print('Finding optimal amount of virtual zones')
                X = rental.loc[:,['Start_GPS_Latitude','Start_GPS_Longitude']].sample(frac=0.05) # Using only a fraction of rental to train quicker
                scores = []
                n_original_areas = len(pd.unique(rental['Start_Zone_Name']))
                for n in tqdm(range(n_original_areas, 4*n_original_areas, int(n_original_areas/50))):
                    km = KMeans(n_clusters=n).fit(X)
                    scores.append([n, silhouette_score(X, km.labels_)])
                scores = pd.DataFrame(scores)
                n_zones = int(scores.iloc[scores.iloc[:,1].argmax(),0]) # Pick n_zones with highes silhouette_score
                scores.to_csv(Path.cwd() / 'reports' /'virtual_area_opt.csv', index=False)
            else:
                scores = pd.read_csv(Path.cwd() / 'reports' / 'virtual_area_opt.csv', index_col='0')
                n_zones = int(scores.iloc[np.argmax(scores)].name)

            # Determine the correct zones using the whole dataset
            km = KMeans(n_clusters=n_zones, verbose=1).fit(rental.loc[:,['Start_GPS_Latitude','Start_GPS_Longitude']])
            areas = pd.DataFrame(km.cluster_centers_, columns=['GPS_Latitude','GPS_Longitude'])
            areas.index = ['virtual_zone_'+str(label) for label in areas.index]
            rental['Virtual_Start_Zone_Name'] = ['virtual_zone_'+str(label) for label in km.labels_]
            rental['Virtual_End_Zone_Name'] = ['virtual_zone_'+str(label) for label in km.predict(rental.loc[:,['End_GPS_Latitude','End_GPS_Longitude']])]

            rental.to_csv(Path.cwd() / 'data' / 'processed' / 'rental.csv', index=False)
            areas.to_csv(Path.cwd() / 'data' / 'processed' / 'areas.csv')


        if not (Path.cwd() / 'data' / 'processed' / 'openings.csv').is_file():
            open_files = glob.glob(str(Path.cwd() / 'data' / 'raw' / open_folder / '*.csv'))
            open_dfs = [pd.read_csv(f) for f in open_files]
            openings = pd.concat(open_dfs,ignore_index=True)
            openings = openings[openings['Source_Location_ID']!='Source_Location_ID']
            openings = openings.loc[:, ['Created_Datetime_Local', 'Platform', 'GPS_Longitude', 'GPS_Latitude']]
            openings = openings[
                (openings['GPS_Latitude'] > swlat) & (openings['GPS_Latitude'] < nelat) & 
                (openings['GPS_Longitude'] > swlon) & (openings['GPS_Longitude'] < nelon)]
            openings.to_csv(Path.cwd() / 'data' / 'processed' / 'openings.csv', index=False)

class sarDataset(Dataset):
    def __init__(self, timepoints, time_window:timedelta=timedelta(hours=1)):

        if not isinstance(timepoints, (list, tuple, set, np.ndarray, pd.Series)):
            raise TypeError("timepoints is not list-like")
        if not all(isinstance(n, datetime) for n in timepoints):
            raise TypeError("timepoints array has to contain datetime objects")
        if not isinstance(time_window, timedelta):
            raise TypeError("time_window is not timedelta")
        
        self.timepoint = timepoints

        self.time_window = time_window
        self.wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method
        
        pd.options.mode.chained_assignment = None

        self.load_data()        

    def load_data(self):
        self.area_centers = pd.read_csv(Path.cwd().parent / 'data' / 'processed' / 'areas.csv', index_col=0)
        self.area_centers.set_index('Area', inplace=True)

        self.openings = pd.read_csv(Path.cwd() / 'data' / 'processed' / 'openings.csv')
        self.openings['Created_Datetime_Local'] = pd.to_datetime(self.openings['Created_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.openings = pd.get_dummies(self.openings, columns=['Platform'], drop_first=True)
    
        self.rental = pd.read_csv((Path.cwd() / 'data' / 'processed' / 'rental.csv'), low_memory=False)
        self.rental['Start_Datetime_Local'] = pd.to_datetime(self.rental['Start_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.rental['End_Datetime_Local'] = pd.to_datetime(self.rental['End_Datetime_Local'], format='%Y-%m-%d %H:%M')
        self.rental = pd.get_dummies(self.rental, columns=['Vehicle_Engine_Type'], drop_first=True)
        self.rental = pd.get_dummies(self.rental, columns=['Vehicle_Model'])
        
        self.vehicles = self.rental.columns[self.rental.columns.str.contains('Vehicle_Model')] # Get names of vehicles
        print('Data load finished')

    def distance(self,lat1,lon1,lat2,lon2):
        # Auxiliary method for coords_to_areas. Calculates a distance between 2 coordinates
        _,_,dist = self.wgs84_geod.inv(lon1,lat1,lon2,lat2)
        return dist

    def coords_to_areas(self, target: pd.Series):
        # Auxiliary method for demand. Calculate to which area an opening's coordinates (target) "belong to".
        # Following 3 lines needed to have a Series of the same length as area_centers with target as value
        dist = deepcopy(self.area_centers) # Without deepcopy area_centers is modified in the next 2 lines
        dist['TGPS_Latitude'] = target['GPS_Latitude']
        dist['TGPS_Longitude'] = target['GPS_Longitude']
        
        dists = self.distance(dist['GPS_Latitude'], dist['GPS_Longitude'], dist['TGPS_Latitude'], dist['TGPS_Longitude'])
        return pd.Series(1 - dists / sum(dists)) # Percentage of how much an opening belongs to each area
        
    def demand(self, idx):
        # Auxiliary method for __getitem__. Uses array timepoint as a index. Returns the demand of all areas at some point in time.
        dem = self.openings[(self.openings['Created_Datetime_Local'] > self.timepoint[idx]-self.time_window) &
        (self.openings['Created_Datetime_Local'] <= self.timepoint[idx])]
        dem[self.area_centers.index.values] = 0 # Create columns with area names
        dem[self.area_centers.index.values] = dem.apply(lambda x: self.coords_to_areas(x), axis=1) # Apply function to all openings
        dem = dem.sum(axis=0).loc[self.area_centers.index] # Aggregate demand in the time window over areas (.loc to remove gps coords and platform)
        return pd.DataFrame(dem, columns=['demand']) # Sum of demand equals to amount of app openings
    
    def vehicle_locations(self, idx):
        # Auxiliary method for __getitem__. Uses array timepoint as a index.
        loc = self.rental[self.rental['End_Datetime_Local'] <= self.timepoint[idx]].drop('Revenue_Net', axis=1)
        loc = loc.sort_values(by='End_Datetime_Local').drop_duplicates(subset='Vehicle_Number_Plate', keep='last') # Keep the last location
        current_trips = self.rental[(self.rental['Start_Datetime_Local'] <= self.timepoint[idx]) & (self.rental['End_Datetime_Local'] > self.timepoint[idx])] # Cars in use
        loc = loc[~loc['Vehicle_Number_Plate'].isin(current_trips['Vehicle_Number_Plate'])] # Filter out cars in use
        loc = loc.loc[:, ~loc.columns.str.contains('Start')].drop(columns=['End_Datetime_Local'], axis=1) # Drop unused columns
        loc.rename(columns={'End_Zone_Name': 'Zone'}, inplace=True)
        loc = loc.groupby('Zone')[self.vehicles].sum() # Aggregate amount of cars
        missing_areas = pd.DataFrame(index=self.area_centers.index[~self.area_centers.index.isin(loc.index)], columns=loc.columns, data=0)
        return pd.concat([loc, missing_areas]).sort_index() # Add missing areas, sort and return

    def state(self, idx):
        # Auxiliary method for __getitem__. Joins vehicle locations and demand
        s = deepcopy(self.area_centers) # Create rows with all locations
        s = pd.concat([s, self.vehicle_locations(idx)], axis=1) # Locations now
        dem = self.demand(idx)
        dem.columns = dem.columns + '_' + str(int(self.time_window.days*24 + self.time_window.seconds/3600)) + 'h' # Add time window to column name
        s = pd.concat([s, dem], axis=1).iloc[:,2:] # Add demands and locations to final dataframe and discard coordinates
        return s

    def actions(self, idx):
        # Auxiliary method for __getitem__. Calculates actions
        a = self.rental[(self.rental['Servicedrive_YN']==1) &
                        (self.rental['Start_Datetime_Local'] >= self.timepoint[idx]-self.time_window) &
                        (self.rental['End_Datetime_Local'] < self.timepoint[idx])]
        a = a[a['Start_Zone_Name'] != a['End_Zone_Name']]
        a = a.loc[:, [*self.vehicles, 'Start_Zone_Name', 'End_Zone_Name', 'Servicedrive_YN']]
        a = pd.melt(a, id_vars=['Start_Zone_Name', 'End_Zone_Name'], value_vars=[*self.vehicles])
        a.rename(columns={'variable': 'Vehicle_Model'}, inplace=True)
        return a.groupby(['Vehicle_Model', 'Start_Zone_Name', 'End_Zone_Name']).sum().unstack()

    def revenue(self, idx):
        # Auxiliary method for __getitem__. Uses array timepoint as a index.
        trips_in_window = self.rental[(self.rental['Start_Datetime_Local'] >= self.timepoint[idx]-self.time_window) & (self.rental['End_Datetime_Local'] < self.timepoint[idx])]
        return trips_in_window['Revenue_Net'].sum()

    def __len__(self):
        return len(self.timepoint)

    def __getitem__(self, idx):
        s = self.state(idx) # Returns position of cars in timepoint idx and demand between idx-timedelta and idx
        a = self.actions(idx) # Returns end position of cars due to service trips within idx-timedelta (only moved cars)
        s1 = self.state(idx+1) # Returns position of cars in timepoint idx+1 and demand between idx+1-timedelta and idx+1
        r = self.revenue(idx) # Returns total revenue between idx-timedelta and idx
        return s, a, s1, r

if __name__ == "__main__":
    dm = DataModule()
    dm.prepare_data(optimise=True)
    # data = sarDataset(np.arange(datetime(2020, 2, 1, 0, 56, 26), datetime(2020, 2, 1, 0, 56, 26), timedelta(hours=1)).astype(datetime))
    # s, a, s1, r = data[1000]
    # print('s:', s, '\n\n')
    # print('a:', a, '\n\n')
    # print('s1:', s1, '\n\n')
    # print('r:', r, '\n\n')