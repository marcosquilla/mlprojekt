from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method

# How much time left
# Increase areas
# Run extreme cases in sim
# Plot n cars per area and requests
# Speed improvement: merge all datasets into 1

class Sim():
    def __init__(self, time_step:timedelta=timedelta(minutes=30), time_start=datetime(2020, 2, 2, 0, 0, 0), time_end=datetime(2021, 5, 3, 23, 59, 59)):

        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime)
        self.reference_datetime = datetime(2000,1,1)
        self.timepoints_secs = ((self.timepoints-self.reference_datetime).astype('timedelta64[ms]')/1000).astype(int)
        self.dist2time = 0.846283 # Factor to get travel time from distance in s/m (Avg. trip duration/Avg. dist)
        self.dist2reve = 1.468  # Factor to get revenue from distance
        self.max_dist = 500 # Maximum distance from request to area (walking distance)
        self.max_cars = 1000 # Max cars per area

        self.revenue = 0

        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
        rentals = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), parse_dates=['Start_Datetime_Local'])
        self.car_locs = rentals.groupby('Vehicle_Number_Plate')[['Virtual_Start_Zone_Name', 'Vehicle_Model']].first().reset_index()
        self.car_locs['Arrival'] = int((time_start-self.reference_datetime).total_seconds())
        rentals['Month'] = rentals['Start_Datetime_Local'].dt.month
        rentals['Weekday'] = rentals['Start_Datetime_Local'].dt.weekday
        rentals['Hour'] = rentals['Start_Datetime_Local'].dt.hour
        self.dest = rentals.groupby(['Month', 'Weekday', 'Hour', 'Virtual_Start_Zone_Name'])['Virtual_End_Zone_Name']

        if not (Path('.') / 'data' / 'processed' / 'requests.csv').is_file():
            tqdm.pandas()
            self.openings = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'), parse_dates=['Created_Datetime_Local'])
            self.requests = self.openings.progress_apply(self.dr2a, axis=1)
            self.requests.to_csv(Path('.') / 'data' / 'processed' / 'requests.csv', index=False)
        self.requests = pd.read_csv((Path('.') / 'data' / 'processed' / 'requests.csv'), parse_dates=['Created_Datetime_Local'])
        self.requests['Created_Datetime_Local'] = ((self.requests['Created_Datetime_Local']-self.reference_datetime).astype('timedelta64[ms]')/1000).astype(int)

        self.dists = np.zeros([len(self.area_centers), len(self.area_centers)])
        for o, d in list(product(self.area_centers.index, repeat=2)):
            _,_,self.dists[o,d] = wgs84_geod.inv(
                self.area_centers.iloc[o]['GPS_Longitude'], self.area_centers.iloc[o]['GPS_Latitude'],
                self.area_centers.iloc[d]['GPS_Longitude'], self.area_centers.iloc[d]['GPS_Latitude'])

        self.areas = [Area(coords,
                    self.car_locs.groupby('Virtual_Start_Zone_Name').get_group(i).loc[:,['Arrival', 'Vehicle_Number_Plate', 'Vehicle_Model']].values, 
                    self.max_cars, 
                    i) for i, coords in self.area_centers.iterrows()]

    def step(self, idx): # idx only to get statistical data
        rs = self.requests[(self.requests['Created_Datetime_Local']>=self.timepoints_secs[idx]) & (self.requests['Created_Datetime_Local']<self.timepoints_secs[idx+1])]
        for request in rs.itertuples():
            if request[6]<=self.max_dist:
                try:
                    d = self.dest.get_group((
                        self.timepoints[idx].month, 
                        self.timepoints[idx].weekday(), 
                        self.timepoints[idx].hour, 
                        request[5])).sample(1).values.tolist()[0]  # Sample a destination from historical data based on origin and datetime
                except KeyError: # No previous trip from origin area at this time
                    d = np.random.randint(0, len(self.areas), 1).tolist()[0]
                r = self.areas[request[5]].depart(
                    self,
                    self.areas[d],
                    request[1]) 
                self.revenue += r
                return r

    def get_revenue(self):
        return self.revenue

    def get_state(self, idx): # idx only to see available cars
        time = self.timepoints_secs[idx]
        locs = np.array([
            [[a.name]*len(a.available_cars(time)), 
            a.cars[a.available_cars(time),1].tolist(),
            a.cars[a.available_cars(time),2].tolist()] for a in self.areas]) # [[Location][Plate][Model]]
        return locs

    def move_car(self, origin, dest, time, model):
        self.areas[origin].depart(self, self.areas[dest], time=time, model=model)

    def dr2a(self, row): # Distance from request to area
        _,_,dists = wgs84_geod.inv(
                self.area_centers['GPS_Longitude'], self.area_centers['GPS_Latitude'],
                np.full(len(self.area_centers),row['GPS_Longitude']), np.full(len(self.area_centers),row['GPS_Latitude']))
        ca = np.argmin(dists)
        row['Area'] = ca
        row['Distance'] = dists[ca]
        return row

class Area(): # Contains area properties and states. Used in simulator
    def __init__(self, location, cars:np.array, max_cars:int, name):
        self.name = name
        self.location = location
        self.max_cars = max_cars
        self.cars = cars # [[Arrival][Plate][Model]]

    def arrival(self, time, car, model): # Only to be called by depart method from another object
        if self.total_cars() < self.max_cars: # If not enough space deny trip
            np.vstack((self.cars, (time, car, model)))
            return True
        else:
            return False

    def depart(self, params, destination, time, car_i=0, model=None) -> float: # car_i = 0 FIFO
        revenue = 0
        if destination.name != self.name and len(self.available_cars(time))>0:
            distance = params.dists[self.name, destination.name] # Arrival time
            time += distance*params.dist2time
            if model is not None:
                try:
                    car_i = np.where(self.cars[:, 2]==model)[0][car_i] # Filters all cars of specific model
                except IndexError: # If no cars of this model, choose from anything else
                    pass
            if destination.arrival(time, self.cars[car_i, 1], self.cars[car_i, 2]):
                revenue = distance*params.dist2reve
                np.delete(self.cars, car_i, axis=0)             
        return revenue
    
    def total_cars(self):
        return len(self.cars)

    def available_cars(self, time):
        return np.where(self.cars[:,0]<=time)

def run_hist(runs=1):
    actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])
    for _ in tqdm(range(runs)):
        sim = Sim()
        for i, t in enumerate(tqdm(sim.timepoints[:-1], leave=False)):
            sim.step(i)
            for _, action in actions[actions['Time']==t].iterrows(): # Perform historical moves
                sim.move_car(
                    action['Virtual_Start_Zone_Name'],
                    action['Virtual_Zone_Name'],
                    (t-sim.reference_datetime).total_seconds(),
                    action[action==1].index[action[action==1].index.str.contains('Vehicle')][0][14:])
        pd.DataFrame([sim.get_revenue()]).to_csv('reports/sim_hist_rev.csv', index=False, header=False, mode='a')

def run_no_moves(runs=1):
    for _ in tqdm(range(runs)):
        sim = Sim()
        for i, t in enumerate(tqdm(sim.timepoints[:-1], leave=False)):
            sim.step(i)
        pd.DataFrame([sim.get_revenue()]).to_csv('reports/sim_no_moves.csv', index=False, header=False, mode='a')

if __name__ == "__main__":
    run_hist(runs=20)
    run_no_moves(runs=20)
