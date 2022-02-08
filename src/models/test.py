import os
from argparse import ArgumentParser
from pathlib import Path
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from src.models.simulator import Sim
from src.models.models import BC_Area_s1, BC_Area_s2, BCLSTM_Area_s1
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84')
pd.options.mode.chained_assignment = None 

def get_ckpt_path(args):
    ckpt_path_s1 = (Path('.') / 'models' / "stage_1" / 'lightning_logs' / str('version_' + str(args.stage_1_version)) / 'checkpoints' / '*.ckpt')
    ckpt_path_s2 = (Path('.') / 'models' / "stage_2" / 'lightning_logs' / str('version_' + str(args.stage_2_version)) / 'checkpoints' / '*.ckpt')
    return Path('.') / max(glob.glob(str(ckpt_path_s1)), key=os.path.getmtime), Path('.') / max(glob.glob(str(ckpt_path_s2)), key=os.path.getmtime)

def run_hist(args):
    actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])
    for _ in tqdm(range(args.runs)):
        sim = Sim()
        for _, t in enumerate(tqdm(sim.timepoints[:-1], leave=False)):
            sim.step()
            for _, action in actions[actions['Time']==t].iterrows(): # Perform historical moves
                sim.move_car(
                    action['Virtual_Start_Zone_Name'],
                    action['Virtual_Zone_Name'],
                    action[action==1].index[action[action==1].index.str.contains('Vehicle')][0][14:])
        pd.DataFrame([sim.get_revenue()]).to_csv('reports/sim_hist_rev.csv', index=False, header=False, mode='a')

def run_no_moves(args):
    for _ in tqdm(range(args.runs)):
        sim = Sim()
        for _ in tqdm(sim.timepoints[:-1], leave=False):
            sim.step()
        pd.DataFrame([sim.get_revenue()]).to_csv('reports/sim_no_moves.csv', index=False, header=False, mode='a')

def run_single_moves(args):
    pd.DataFrame([]).to_csv(f'reports/sim_{args.steps}_me.csv', index=False, header=False)
    pd.DataFrame([]).to_csv(f'reports/sim_{args.steps}_tc.csv', index=False, header=False)
    for _ in tqdm(range(args.runs)):
        sim = Sim()
        for i, t in enumerate(tqdm(sim.timepoints[:-1], leave=False)):
            tc = [a.total_cars() for a in sim.areas]
            sim.step()
            if i % args.steps == 0:
                sim.move_car(
                    np.random.randint(0, len(sim.areas), 1).tolist()[0], 
                    3) # Moves 1 random car to area 3
            pd.DataFrame(tc).to_csv(f'reports/sim_{args.steps}_tc.csv', index=False, header=False, mode='a')
        pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/sim_{args.steps}_me.csv', index=False, header=False, mode='a')

def run_bc(args):
    path_s1, path_s2 = get_ckpt_path(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage_1 = BC_Area_s1.load_from_checkpoint(path_s1).to(device)
    stage_2 = BC_Area_s2.load_from_checkpoint(path_s2).to(device)
    stage_1.eval()
    stage_2.eval()

    if not (Path('.') / 'reports' / f'sim_{args.stage_1_version}{args.stage_2_version}_bc.csv').is_file():
        pd.DataFrame([]).to_csv(f'reports/sim_{args.stage_1_version}{args.stage_2_version}_bc.csv', index=False, header=False)
    
    for _ in tqdm(range(args.runs)):
        sim = Sim()
        for _ in tqdm(sim.timepoints[:-1], leave=False):
            sim.step()
            s, _ = sim.get_state()
            s = s.to(device)
            to_move = torch.where(stage_1(s).squeeze()>0)[0]
            if len(to_move)>0:
                dest = torch.argmax(stage_2(s[to_move]), dim=1)
                for j, a in enumerate(to_move):
                    sim.move_car(a.item(), dest[j].item())
        pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/sim_{args.stage_1_version}{args.stage_2_version}_bc.csv', index=False, header=False, mode='a')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hist', action='store_true', help='Simulate moving cars according to history')
    parser.add_argument('--no_moves', action='store_true', help='Simulate without moving cars')
    parser.add_argument('--single', action='store_true', help='Simulate moving a single random car to area 3 every --steps steps')
    parser.add_argument('--bc', action='store_true', help='Simulate with behavioural cloning')
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--stage_1_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    parser.add_argument('--stage_2_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    args = parser.parse_args()

    if args.hist:
        run_hist(args)
    if args.no_moves:
        run_no_moves(args)
    if args.single:
        run_single_moves(args)
    if args.bc:
        run_bc(args)