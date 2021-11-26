from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from itertools import product
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method

class sim():
    def __init__(self, time_step:timedelta=timedelta(minutes=30), time_start=datetime(2020, 2, 2, 0, 0, 0), time_end=datetime(2021, 5, 3, 23, 59, 59)):
        
        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime)

        self.dist2time = 5 # Factor to get travel time from distance
        self.dist2reve = 2*self.dist2time  # Factor to get revenue from distance
        self.max_dist = 500 # Maximum distance from request to area (walking distance)
        self.max_cars = 1000 # Max cars per area

        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
        self.openings = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'), parse_dates=['Created_Datetime_Local'])
        rentals = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), parse_dates=['Start_Datetime_Local'])
        self.car_locs = rentals.groupby('Vehicle_Number_Plate')[['Virtual_Start_Zone_Name', 'Vehicle_Model']].first().reset_index()
        self.car_locs['Arrival'] = time_start
        rentals['Month'] = rentals['Start_Datetime_Local'].dt.month
        rentals['Weekday'] = rentals['Start_Datetime_Local'].dt.weekday
        rentals['Hour'] = rentals['Start_Datetime_Local'].dt.hour
        self.dest = rentals.groupby(['Month', 'Weekday', 'Hour', 'Virtual_Start_Zone_Name'])['Virtual_End_Zone_Name']
        del rentals

        self.dists = np.zeros([len(self.area_centers), len(self.area_centers)])
        for o, d in list(product(self.area_centers.index, repeat=2)):
            _,_,self.dists[o,d] = wgs84_geod.inv(
                self.area_centers.iloc[o]['GPS_Longitude'], self.area_centers.iloc[o]['GPS_Latitude'],
                self.area_centers.iloc[d]['GPS_Longitude'], self.area_centers.iloc[d]['GPS_Latitude'])

        self.areas = [area(coords,
                    self.car_locs.groupby('Virtual_Start_Zone_Name').get_group(i).loc[:,['Arrival', 'Vehicle_Number_Plate', 'Vehicle_Model']].values, 
                    self.max_cars, 
                    i) for i, coords in self.area_centers.iterrows()]

    def get_revenue(self, idx):
        revenue = 0
        requests = self.openings[(self.openings['Created_Datetime_Local']>=self.timepoints[idx]) & (self.openings['Created_Datetime_Local']<self.timepoints[idx+1])]
        for _, request in requests.iterrows():
            _,_,dists = wgs84_geod.inv(
                    self.area_centers['GPS_Longitude'], self.area_centers['GPS_Latitude'],
                    np.full(len(self.area_centers),request['GPS_Longitude']), np.full(len(self.area_centers),request['GPS_Latitude']))
            ca = np.argmin(dists) # Closest area
            if dists[ca]<=self.max_dist:
                revenue += self.areas[ca].depart(self.dest.get_group((
                    self,
                    self.timepoints[idx].month, 
                    self.timepoints[idx].weekday, 
                    self.timepoints[idx].hour, 
                    ca)), request['Created_Datetime_Local'])
        return revenue

class area(): # Contains area properties and states. Used in simulator
    def __init__(self, location, cars, max_cars, name):
        self.name = name
        self.location = location
        self.max_cars = max_cars
        self.cars = cars # [[Arrival][Plate][Model]]

    def arrival(self, time, car, model): # Only to be called by depart method from another object
        if self.total_cars() == self.max_cars: # If not enough space deny trip
            return False
        else:
            self.cars.append([time, car, model])
            return True

    def depart(self, params, destination, time, car_i=0) -> float: # car_i = 0 FIFO
        revenue = 0
        if destination.name != self.name and self.available_cars()>0:
            distance = params.dists[self.name, destination.name] # Arrival time
            time += distance*params.dist2time
            if destination.arrival(time, self.cars[car_i, 1], self.cars[car_i, 2]):
                revenue = distance*params.dist2reve
                np.delete(self.cars, car_i, axis=0)
        return revenue
    
    def total_cars(self):
        return len(self.cars)

    def available_cars(self, time):
        return len(np.where(self.cars[:,0]<=time))