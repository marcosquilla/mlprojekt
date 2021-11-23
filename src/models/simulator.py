from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method

area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
dists = np.zeros([len(area_centers), len(area_centers)])
for o, d in list(product(area_centers.index, repeat=2)):
    _,_,dists[o,d] = wgs84_geod.inv(
        area_centers.iloc[o]['GPS_Longitude'], area_centers.iloc[o]['GPS_Latitude'],
        area_centers.iloc[d]['GPS_Longitude'], area_centers.iloc[d]['GPS_Latitude'])

dist2time = 10
dist2reve = 5

class car():
    def __init__(self, model, location, plate=None):
        self.model = model
        self.location = location
        self.plate = plate
        
    def travels_to(self, destination, time):
        revenue = 0
        if destination!=self.location:
            distance = dists[self.location.name, destination.name] # Arrival time
            time += distance*dist2time
            if destination.arrival(self, time): # If enough space in destination execute trip
                revenue = distance*dist2reve
                self.location.departure(self)
        return revenue


class area():
    def __init__(self, location, cars:np.array, max_cars):
        self.location = location
        self.max_cars = max_cars
        self.cars = cars # [[Arrival time][Car]]

    def arrival(self, car, time):
        if self.total_cars() == self.max_cars: # If not enough space deny trip
            return False
        else:
            self.cars.append([time, car])
            return True

    def departure(self, car):
        np.delete(self.cars, (np.where(self.cars[:,1]==car)), axis=0)
    
    def total_cars(self):
        return len(self.cars)

    def available_cars(self, time):
        return len(np.where(self.cars[:,0]<=time))




# props = <some list here>
# objects = [MyClass(property=foo, property2=prop) for prop in props]
# for obj in objects:
#     obj.do_stuff(variable=foobar)