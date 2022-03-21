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
        ckpt_path = (Path('.') / 'models' / "dqn" / 'lightning_logs' / str('version_' + str(args.rl_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path)), key=os.path.getmtime)
    elif args.cqn:
        ckpt_path = (Path('.') / 'models' / "cqn" / 'lightning_logs' / str('version_' + str(args.rl_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path)), key=os.path.getmtime)
    else:
        ckpt_path_s1 = (Path('.') / 'models' / "stage_1" / 'lightning_logs' / str('version_' + str(args.stage_1_version)) / 'checkpoints' / '*.ckpt')
        ckpt_path_s2 = (Path('.') / 'models' / "stage_2" / 'lightning_logs' / str('version_' + str(args.stage_2_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path_s1)), key=os.path.getmtime), Path('.') / max(glob.glob(str(ckpt_path_s2)), key=os.path.getmtime)

def run_hist(args, sim):
    actions = pd.read_csv((Path('.') / 'data' / 'processed' / 'actions.csv'), parse_dates=['Time'])
    
    _, d = sim.step()
    if args.save_buffer:
        buffer = ReplayBuffer(10000000)
    while not d:
        state, _ = sim.get_state()
        for _, action in actions[actions['Time']==sim.timepoints[sim.i]].iterrows(): # Perform historical moves
            sim.move_car(
                action['Virtual_Start_Zone_Name'],
                action['Virtual_Zone_Name'],
                action[action==1].index[action[action==1].index.str.contains('Vehicle')][0][14:])
        reward, d = sim.step()
        
        if args.save_buffer:
            new_state, done = sim.get_state()
            for i, s in enumerate(state): # Perform historical moves
                action = actions[(actions['Time']==sim.timepoints[sim.i]) & (actions['Virtual_Start_Zone_Name']==i)]
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

    pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/{args.set}/scores_{args.cost}/sim_hist_rev.csv', index=False, header=False, mode='a')

def run_single_moves(args, sim):
    d = False
    while not d:
        _, d = sim.step()
        if sim.i % args.steps == 0:
            sim.move_car(
                np.random.randint(0, len(sim.areas), 1).tolist()[0], 
                3) # Moves 1 random car to area 3
    pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/{args.set}/scores_{args.cost}/sim_{args.steps}_me.csv', index=False, header=False, mode='a')

def run_bc(args, sim):
    path_s1, path_s2 = get_ckpt_path(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage_1 = BC_Area_s1.load_from_checkpoint(path_s1).to(device)
    stage_2 = BC_Area_s2.load_from_checkpoint(path_s2).to(device)
    stage_1.eval()
    stage_2.eval()
    d = False
    while not d:
        _, d = sim.step()
        s, _ = sim.get_state()
        s = s.to(device)
        to_move = torch.where(stage_1(s).squeeze()>0)[0]
        if len(to_move)>0:
            dest = torch.argmax(stage_2(s[to_move]), dim=1)
            for j, a in enumerate(to_move):
                sim.move_car(a.item(), dest[j].item())
    pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/{args.set}/scores_{args.cost}/sim_{args.stage_1_version}{args.stage_2_version}_bc.csv', index=False, header=False, mode='a')

def run_rl(args, sim):
    path = get_ckpt_path(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dqn:
        dqn = DQN.load_from_checkpoint(path).to(device)
        suf = 'dqn'
    elif args.cqn:
        dqn = CQN.load_from_checkpoint(path).to(device)
        suf = 'cqn'
    dqn.eval()
    d = False

    #sim = Sim(time_step=timedelta(minutes=60), time_end=datetime(2020, 2, 5, 16, 00, 00))
    td = []
    while not d:
        #print(f"Step {sim.i}")
        #print([c.total_cars() for c in sim.areas])
        _, d = sim.step()
        s, _ = sim.get_state()
        s = s.to(device)
        _, dests = torch.max(dqn(s), dim=1)
        if args.save_moves:
            td.append(dests.cpu().detach().numpy().tolist())
        #print("Moves")
        for a, dest in enumerate(dests):
            sim.move_car(a, dest.item())
    pd.DataFrame([sim.get_revenue()]).to_csv(f'reports/{args.set}/scores_{args.cost}/sim_{args.rl_version}_{suf}.csv', index=False, header=False, mode='a')
    if args.save_moves:
        pd.DataFrame(td).to_csv(f'reports/{args.set}/moves/td_{args.rl_version}_{suf}.csv', index=False, header=False, mode='a')

def cost_experiment(args):
    if args.cexp:
        args.hist, args.single, args.cexp = True, False, False
        main(args)
        args.hist, args.single = False, True
        for sp in range(8):
            args.steps = 2**sp
            main(args)
    else:
        args.bc, args.dqn, args.cqn, args.cexpg = True, False, False, False
        main(args)
        args.bc = False
        for rl_version in range(4,7):
            args.rl_version = rl_version
            args.dqn, args.cqn = True, False 
            main(args)
            args.dqn, args.cqn = False, True        
            main(args)

        args.rl_version += 1
        main(args) # Run last cqn version (hist)

def main(args):
    assert args.set=='train' or args.set=='test', '--set must be train or test'
    if args.set=='train':
        sim = Sim(time_end=datetime(2020,6,1,0,0,0), cost=args.cost)
    else:
        sim = Sim(time_start=datetime(2020,6,1,0,0,0), cost=args.cost)

    if args.hist:
        run_hist(args, sim)
    if args.single:
        run_single_moves(args, sim)
    if args.bc:
        run_bc(args, sim)
    if args.dqn or args.cqn:
        run_rl(args, sim)
    if args.cexp or args.cexpg:
        cost_experiment(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--hist', action='store_true', help='Simulate moving cars according to history')
    parser.add_argument('--single', action='store_true', help='Simulate moving a single random car to area 3 every --steps steps')
    parser.add_argument('--bc', action='store_true', help='Simulate with behavioural cloning')
    parser.add_argument('--dqn', action='store_true', help='Simulate with DQN')
    parser.add_argument('--cqn', action='store_true', help='Simulate with Offline DQN')
    parser.add_argument('--save_buffer', action='store_true', help='Save historical buffer')
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--steps', default=32, type=int)
    parser.add_argument('--stage_1_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    parser.add_argument('--stage_2_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    parser.add_argument('--rl_version', default=0, type=int, help='Version to load from lightning_logs/version_#/checkpoint/.')
    parser.add_argument('--save_moves', action='store_true', help='Save moves from RL algorithms')
    parser.add_argument('--cost', default=1, type=float, help='Making a move costs revenue*--cost')
    parser.add_argument('--cexp', action='store_true', help='Run experiment with fixed cost without DL models')
    parser.add_argument('--cexpg', action='store_true', help='Run experiment with fixed cost of DL models')
    parser.add_argument('--set', default='test', type=str, help='Period to use: train or test')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(f'reports/{args.set}/scores_{args.cost}/'), exist_ok=True)
    for _ in tqdm(range(args.runs)):
        main(args)