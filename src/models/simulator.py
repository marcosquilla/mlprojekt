from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from torch import nn
from itertools import product
from pyproj import Geod
from src.data.datasets import ReplayBuffer
wgs84_geod = Geod(ellps='WGS84') # Distance will be measured in meters on this ellipsoid - more accurate than a spherical method

# Rename cqn versions. Test new version on all costs and sets

class Sim():
    def __init__(self, time_step:timedelta=timedelta(minutes=30), time_start=datetime(2020, 2, 2, 0, 0, 0), time_end=datetime(2021, 5, 3, 23, 59, 59), cost=1):

        self.timepoints = np.arange(time_start, time_end, time_step).astype(datetime)
        self.reference_datetime = datetime(2000,1,1)
        self.timepoints_secs = ((self.timepoints-self.reference_datetime).astype('timedelta64[ms]')/1000).astype(int)
        self.dist2time = 1 # Factor to get travel time from distance in s/m (Avg. trip duration/Avg. dist)
        self.dist2reve = 0.00001  # Factor to get revenue from distance
        self.rcr = cost # Cost of making a move
        self.r2t = 0.125 # Fraction of requests that turn to trips
        self.max_dist = 5000 # Maximum distance from request to area (walking distance)
        self.max_cars = 1000 # Max cars per area
        self.i = int(0)

        self.revenue = 0

        self.area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
        rentals = pd.read_csv((Path('.') / 'data' / 'interim' / 'rental.csv'), parse_dates=['Start_Datetime_Local'])
        self.car_locs = rentals.groupby('Vehicle_Number_Plate')[['Virtual_Start_Zone_Name', 'Vehicle_Model']].first().reset_index()
        self.car_locs['Arrival'] = int((time_start-self.reference_datetime).total_seconds())
        rentals['Weekday'] = rentals['Start_Datetime_Local'].dt.weekday
        rentals['Hour'] = rentals['Start_Datetime_Local'].dt.hour
        self.dest = rentals.groupby(['Weekday', 'Hour', 'Virtual_Start_Zone_Name'])['Virtual_End_Zone_Name']

        vmodels = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), nrows=1)
        self.vmodels = vmodels.columns[vmodels.columns.str.contains('Vehicle')].str.replace('Vehicle_Model_', '')

        self.requests = pd.read_csv((Path('.') / 'data' / 'interim' / 'openings.csv'), parse_dates=['Created_Datetime_Local'])
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
        
        self.demand = self.requests[(self.requests['Created_Datetime_Local']>=self.timepoints_secs[0]) & (self.requests['Created_Datetime_Local']<self.timepoints_secs[1])]

    def dr2a(self, row): # Distance from request to area
        _,_,dists = wgs84_geod.inv(
                self.area_centers['GPS_Longitude'], self.area_centers['GPS_Latitude'],
                np.full(len(self.area_centers),row['GPS_Longitude']), np.full(len(self.area_centers),row['GPS_Latitude']))
        ca = np.argmin(dists)
        row['Area'] = ca
        row['Distance'] = dists[ca]
        return row

    def open2dem(self, dem, area_centers):
        if len(dem) == 0:
            dem = pd.Series(data=0, index=np.arange(len(area_centers)))
            return dem
        else:
            dem.loc[:,area_centers.index.values] = 0 # Create columns with area names
            dem.loc[:,area_centers.index.values] = dem.apply(self.coords_to_areas, axis=1) # Apply function to all openings
            dem = dem.loc[:,area_centers.index].sum(axis=0) # Aggregate demand in the time window over areas (.loc to remove gps coords and platform). Sum of demand equals to amount of app openings
            return dem

    def coords_to_areas(self, target):
        # Auxiliary method for demand. Calculate to which area an opening's coordinates (target) "belong to".
        _,_,dists = wgs84_geod.inv(
            self.area_centers['GPS_Longitude'], self.area_centers['GPS_Latitude'],
            np.full(len(self.area_centers),target['GPS_Longitude']), np.full(len(self.area_centers),target['GPS_Latitude']))
        return pd.Series(1 - dists / sum(dists)) / (len(dists) - 1) # Percentage of how much an opening belongs to each area

    def step(self): # idx only to get statistical data
        try:
            self.demand = self.requests.loc[(self.requests['Created_Datetime_Local']>=self.timepoints_secs[self.i]) & (self.requests['Created_Datetime_Local']<self.timepoints_secs[self.i+1])].copy()
            rs = self.demand.sample(frac=self.r2t)
            r = 0
            for request in rs.itertuples():
                if request[6]<=self.max_dist:
                    try:
                        d = self.dest.get_group((
                            self.timepoints[self.i].weekday(), 
                            self.timepoints[self.i].hour, 
                            request[5])).sample(1).values.tolist()[0]  # Sample a destination from historical data based on origin and datetime
                    except KeyError: # No previous trip from origin area at this date and time. Pick random one.
                        d = np.random.randint(0, len(self.areas), 1).tolist()[0]
                    r += self.areas[request[5]].depart(
                        self,
                        self.areas[d],
                        request[1]) 
            self.revenue += r
            self.i += 1
            return r, False
        except IndexError:
            return 0, True

    def get_revenue(self):
        return self.revenue

    def get_state(self): # Returns batch with state for each area
        time = self.timepoints_secs[self.i]
        locs = np.hstack(
            [np.array([([a.name]*len(a.available_cars(time))),
             (a.cars[a.available_cars(time),2].tolist())]) for a in self.areas]).T # [[Location][Plate][Model]]

        day = torch.zeros(7)
        day[self.timepoints[self.i].weekday()] = 1
        hour = torch.zeros(24)
        hour[self.timepoints[self.i].hour] = 1

        cs = [day, hour, 
            torch.tensor(self.open2dem(self.demand, self.area_centers).tolist()),
            torch.tensor([len(a.available_cars(self.timepoints_secs[self.i])) for a in self.areas])] # Constant dimensions in step

        s = torch.stack([torch.hstack([*cs,
            torch.tensor([len(locs[(locs[:,0]==str(area.name)) & (locs[:,1]==vm)]) for vm in self.vmodels]), # Number of each car model in each area
            torch.tensor([a.name==area.name for a in self.areas], dtype=int)]) for area in self.areas]) # One-hot encode of area identity
        
        d = len(self.timepoints_secs)-2<=self.i
        return s, d

    def move_car(self, origin, dest, model=None):
        if dest<len(self.areas):
            c = self.areas[origin].depart(self, self.areas[dest], self.timepoints_secs[self.i], model=model)
        else:
            c = 0
        self.revenue += -c*self.rcr
        return -c*self.rcr

class Area(): # Contains area properties and states. Used in simulator
    def __init__(self, location, cars:np.array, max_cars:int, name):
        self.name = name
        self.location = location
        self.max_cars = max_cars
        self.cars = cars # [[Arrival][Plate][Model]]

    def arrival(self, time, car, model): # Only to be called by depart method from another area
        if self.total_cars() < self.max_cars: # If not enough space deny trip
            self.cars = np.vstack((self.cars, np.asarray((time, car, model), dtype=object)))
            return True
        else:
            return False

    def depart(self, params, destination, time, car_i=0, model=None) -> float: # car_i = 0 FIFO
        revenue = 0
        if destination.name != self.name and self.available_cars(time).shape[0]>0:
            distance = params.dists[self.name, destination.name] # Arrival time
            time += distance*params.dist2time
            if model is not None:
                try:
                    car_i = np.where(self.cars[:, 2]==model)[0][car_i] # Filters all cars of specific model
                except IndexError: # If no cars of this model, choose from anything else
                    pass
            if destination.arrival(time, self.cars[car_i, 1], self.cars[car_i, 2]):
                #print(self.name, "to", destination.name)
                revenue = distance*params.dist2reve
                self.cars = np.delete(self.cars, car_i, axis=0)
        return revenue
    
    def total_cars(self):
        return len(self.cars)

    def available_cars(self, time):
        return np.where(self.cars[:,0]<=time)[0]

class Agent():
    def __init__(self, replay_buffer:ReplayBuffer, time_step=timedelta(minutes=30), time_start=datetime(2020, 2, 2, 0, 0, 0), time_end=datetime(2021, 5, 3, 23, 59, 59), cost=1, save_step=True):
        self.save = save_step
        self.time_step = time_step
        self.time_start = time_start
        self.time_end = time_end
        self.replay_buffer = replay_buffer
        self.cost = cost
        self.reset()

    def reset(self):
        self.sim = Sim(time_step=self.time_step, time_start=self.time_start, time_end=self.time_end, cost=self.cost)
        self.state, _ = self.sim.get_state()

    def get_action(self, net:nn.Module, epsilon:float, device):
        if net == None or np.random.random() < epsilon:
            action = np.random.randint(len(self.sim.areas),size=len(self.sim.areas))
        else:
            self.state = self.state.to(device)

            q_values = net(self.state)
            _, action = torch.max(q_values, dim=1)
            action = action.cpu().detach().numpy()
        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu"):
        reward = 0       
        actions = self.get_action(net, epsilon, device)

        for origin, destination in enumerate(actions):
            reward += self.sim.move_car(origin, destination)

        r, _ = self.sim.step()
        reward += r

        new_state, done = self.sim.get_state()

        if self.save:
            for i, action in enumerate(actions):
                self.replay_buffer.append(
                    (self.state[i].cpu().detach().numpy(), 
                    action, 
                    reward, 
                    done, 
                    new_state[i].cpu().detach().numpy()))

        self.state = new_state
        if done:
            self.reset()
        return reward, done