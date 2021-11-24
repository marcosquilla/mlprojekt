from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method

area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
openings = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'), parse_dates=['Created_Datetime_Local'])

dists = np.zeros([len(area_centers), len(area_centers)])
for o, d in list(product(area_centers.index, repeat=2)):
    _,_,dists[o,d] = wgs84_geod.inv(
        area_centers.iloc[o]['GPS_Longitude'], area_centers.iloc[o]['GPS_Latitude'],
        area_centers.iloc[d]['GPS_Longitude'], area_centers.iloc[d]['GPS_Latitude'])

dist2time = 5 # Factor to get travel time from distance
dist2reve = 2*dist2time  # Factor to get revenue from distance
max_dist = 500 # Maximum distance from request to area (walking distance)

class area():
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

    def depart(self, destination, time, car_i) -> float:
        revenue = 0
        if destination.name != self.name:
            distance = dists[self.name, destination.name] # Arrival time
            time += distance*dist2time
            if destination.arrival(time, self.cars[car_i, 1], self.cars[car_i, 2]):
                revenue = distance*dist2reve
                np.delete(self.cars, car_i, axis=0)
        return revenue
    
    def total_cars(self):
        return len(self.cars)

    def available_cars(self, time):
        return len(np.where(self.cars[:,0]<=time))

requests = openings[(openings['Created_Datetime_Local']>step_start) & (openings['Created_Datetime_Local']>step_start+delta)]
for _, request in requests.iterrows():
    _,_,dists = wgs84_geod.inv(
            area_centers['GPS_Longitude'], area_centers['GPS_Latitude'],
            np.full(len(area_centers),request['GPS_Longitude']), np.full(len(area_centers),request['GPS_Latitude']))
    ca = np.argmin(dists) # Closest area
    if dists[ca]<=max_dist:
        

# props = <some list here>
# objects = [MyClass(property=foo, property2=prop) for prop in props]
# for obj in objects:
#     obj.do_stuff(variable=foobar)