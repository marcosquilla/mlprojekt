import os
from argparse import ArgumentParser
from pathlib import Path
import pickle
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from src.models.simulator import Sim
from src.models.models import BC_Area_s1, BC_Area_s2, DQN, CQN
from src.data.datasets import ReplayBuffer
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84')
pd.options.mode.chained_assignment = None 
from datetime import timedelta, datetime

def get_ckpt_path(args):
    if args.dqn:
        ckpt_path = (Path('.') / 'models' / "dqn" / 'lightning_logs' / str('version_' + str(args.dqn_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path)), key=os.path.getmtime)
    elif args.cqn:
        ckpt_path = (Path('.') / 'models' / "cqn" / 'lightning_logs' / str('version_' + str(args.dqn_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path)), key=os.path.getmtime)
    else:
        ckpt_path_s1 = (Path('.') / 'models' / "stage_1" / 'lightning_logs' / str('version_' + str(args.stage_1_version)) / 'checkpoints' / '*.ckpt')
        ckpt_path_s2 = (Path('.') / 'models' / "stage_2" / 'lightning_logs' / str('version_' + str(args.stage_2_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path_s1)), key=os.path.getmtime), Path('.') / max(glob.glob(str(ckpt_path_s2)), key=os.path.getmtime)

def run_hist(args):
    actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])
    pd.DataFrame([]).to_csv(f'reports/scores/sim_hist_rev.csv', index=False, header=False)
    for _ in tqdm(range(args.runs)):
        sim = Sim(time_start=datetime(2020,6,1,0,0,0))
        sim.step()
        if args.save_buffer:
            buffer = ReplayBuffer(10000000)
        for _, t in enumerate(tqdm(sim.timepoints[:-2], leave=False)):
            state, _ = sim.get_state()
            for _, action in actions[actions['Time']==t].iterrows(): # Perform historical moves
                sim.move_car(
                    action['Virtual_Start_Zone_Name'],
                    action['Virtual_Zone_Name'],
                    action[action==1].index[action[action==1].index.str.contains('Vehicle')][0][14:])
            reward = sim.step()
            
            if args.save_buffer:
                new_state, done = sim.get_state()
                for i, s in enumerate(state): # Perform historical moves
                    action = actions[(actions['Time']==t) & (actions['Virtual_Start_Zone_Name']==i)]
                    if len(action)==0: # No action for time and origin area found
                        action = i
                    else:
                        action = action['Virtual_Zone_Name'].values.tolist()[0]
                    buffer.append(
                        (s.cpu().detach().numpy(), 
                        action, 
                        reward, 
                        done, 
                        new_state[i].cpu().detach().numpy()))
        if args.save_buffer:
            with open(str(Path('.') / 'data' / 'processed' / f'bufferhist.pkl'), 'wb') as f:
                pickle.dump(buffer, f)

        pd.DataFrame([sim.get_revenue()]).to_csv('reports/scores/sim_hist_rev.csv', index=False, header=False, mode='a')

def run_no_moves(args):
    pd.DataFrame([]).to_csv(f'reports/scores/sim_no_moves.csv', index=False, header=False)
    for _ in tqdm(range(args.runs)):
        sim = Sim()
        for _ in tqdm(sim.timepoints[:-1], leave=False):
            sim.step()
        pd.DataFrame([sim.get_revenue()]).to_csv('reports/scores/sim_no_moves.csv', index=False, header=False, mode='a')

def run_single_moves(args):
    pd.DataFrame([]).to_csv(f'reports/scores/sim_{args.steps}_me.csv', index=False, header=False)
    for _ in tqdm(range(args.runs)):
        sim = Sim(time_start=datetime(2020,6,1,0,0,0))
        for i, _ in enumerate(tqdm(sim.timepoints[:-1], leave=False)):
            sim.step()
            if i % args.steps == 0:
                sim.move_car(
                    np.random.randint(0, len(sim.areas), 1).tolist()[0], 
                    3) # Moves 1 random car to area 3
        pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/scores/sim_{args.steps}_me.csv', index=False, header=False, mode='a')

def run_bc(args):
    path_s1, path_s2 = get_ckpt_path(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage_1 = BC_Area_s1.load_from_checkpoint(path_s1).to(device)
    stage_2 = BC_Area_s2.load_from_checkpoint(path_s2).to(device)
    stage_1.eval()
    stage_2.eval()

    pd.DataFrame([]).to_csv(f'reports/scores/sim_{args.stage_1_version}{args.stage_2_version}_bc.csv', index=False, header=False)    
    for _ in tqdm(range(args.runs)):
        sim = Sim(time_start=datetime(2020,6,1,0,0,0))
        for _ in tqdm(sim.timepoints[:-1], leave=False):
            sim.step()
            s, _ = sim.get_state()
            s = s.to(device)
            to_move = torch.where(stage_1(s).squeeze()>0)[0]
            if len(to_move)>0:
                dest = torch.argmax(stage_2(s[to_move]), dim=1)
                for j, a in enumerate(to_move):
                    sim.move_car(a.item(), dest[j].item())
        pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/scores/sim_{args.stage_1_version}{args.stage_2_version}_bc.csv', index=False, header=False, mode='a')

def run_rl(args):
    path = get_ckpt_path(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dqn:
        dqn = DQN.load_from_checkpoint(path).to(device)
        suf = 'rl'
    elif args.cqn:
        dqn = CQN.load_from_checkpoint(path).to(device)
        suf = 'orl'
    dqn.eval()

    pd.DataFrame([]).to_csv(f'reports/scores/sim_{args.dqn_version}_{suf}.csv', index=False, header=False)
    
    for _ in tqdm(range(args.runs)):
        #sim = Sim(time_step=timedelta(minutes=60), time_end=datetime(2020, 2, 5, 16, 00, 00))
        sim = Sim(time_start=datetime(2020,6,1,0,0,0))
        td = []
        #tc = []
        pd.DataFrame([]).to_csv(f'reports/scores/td_{args.dqn_version}_{suf}.csv', index=False, header=False)
        for _ in tqdm(sim.timepoints[:-1], leave=False):
            #tc.append([c.total_cars() for c in sim.areas])
            #print(f"Step {sim.i}")
            #print([c.total_cars() for c in sim.areas])
            sim.step()
            s, _ = sim.get_state()
            s = s.to(device)
            _, dests = torch.max(dqn(s), dim=1)
            td.append(dests.cpu().detach().numpy().tolist())
            #print("Moves")
            for a, dest in enumerate(dests):
                sim.move_car(a, dest.item())
        pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/scores/sim_{args.dqn_version}_{suf}.csv', index=False, header=False, mode='a')
        pd.DataFrame(td).to_csv(f'reports/td_{args.dqn_version}_{suf}.csv', index=False, header=False, mode='a')
        #pd.DataFrame(tc).to_csv(f'reports/scores/tc_{args.dqn_version}_{suf}.csv', index=False, header=False, mode='a')

def main(args):
    if args.hist:
        run_hist(args)
    elif args.no_moves:
        run_no_moves(args)
    elif args.single:
        run_single_moves(args)
    elif args.bc:
        run_bc(args)
    elif args.dqn or args.cqn:
        run_rl(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hist', action='store_true', help='Simulate moving cars according to history')
    parser.add_argument('--no_moves', action='store_true', help='Simulate without moving cars')
    parser.add_argument('--single', action='store_true', help='Simulate moving a single random car to area 3 every --steps steps')
    parser.add_argument('--bc', action='store_true', help='Simulate with behavioural cloning')
    parser.add_argument('--dqn', action='store_true', help='Simulate with DQN')
    parser.add_argument('--cqn', action='store_true', help='Simulate with Offline DQN')
    parser.add_argument('--save_buffer', action='store_true', help='Save historical buffer')
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--steps', default=32, type=int)
    parser.add_argument('--stage_1_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    parser.add_argument('--stage_2_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    parser.add_argument('--dqn_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    args = parser.parse_args()

    main(args)