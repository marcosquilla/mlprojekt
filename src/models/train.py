import os
from copy import deepcopy
import glob
from argparse import ArgumentParser
from pathlib import Path
import pickle
from tqdm import tqdm
from datetime import timedelta
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.models import BC_Area_s1, BC_Area_s2, BCLSTM_Area_s1, DQN
from src.data.datasets import ReplayBuffer
from src.data.datamodules import AreaDataModule, QDataModule
from src.models.simulator import Agent
import warnings
#warnings.filterwarnings('error')

# TODO: Time_window vs time_delta tuning. Data download.
# https://towardsdatascience.com/building-a-neural-network-on-amazon-sagemaker-with-pytorch-lightning-63730ec740ea
# https://pytorch-lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html

def get_ckpt_path(args, stage):
    if args.load_version is not None:
        ckpt_path = (Path('.') / 'models' / stage / 'lightning_logs' / str('version_' + str(args.load_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path)), key=os.path.getmtime)
    else:
        return None

def create_buffer_agent(args):
    if not (Path('.') / 'data' / 'processed' / 'buffer.pkl').is_file():
        buffer = ReplayBuffer(args.buffer_capacity)
        agent = Agent(buffer)
        print('Populating buffer')
        for _ in tqdm(range(args.warm_up)):
            agent.play_step(net=None, epsilon=1.0)
        print("Saving buffer")
        with open(str(Path('.') / 'data' / 'processed' / 'buffer.pkl'), 'wb') as f:
                pickle.dump(buffer, f)
        print("Done")
    else:
        print("Loading buffer")
        with open(str(Path('.') / 'data' / 'processed' / 'buffer.pkl'), 'rb') as f:
            buffer = pickle.load(f)
        print("Done")
        agent = Agent(buffer)
    return buffer, agent

def setup_model_dm(args, s, ckpt=None):
    area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
    cars = pd.unique(pd.read_csv(Path('.') / 'data' / 'interim' / 'rental.csv', usecols=[2]).iloc[:,0])
    if s == 'stage_1' or s == 'stage_2':
        dm = AreaDataModule(s=s, lstm=args.lstm, shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.num_workers, n_zones=args.n_zones)
        if s == 'stage_1':
            in_size = int(3+len(cars)+3*len(area_centers)) # Date (and time), car models and location (current), amount of cars in all zones, and demand
            o = len(pd.read_csv(Path('.') / 'data' / 'processed' / 'actions.csv', usecols=[0]))
            if args.lstm:
                if ckpt is None:
                    t = pd.read_csv((Path('.') / 'data' / 'processed' / 'locations.csv'), usecols=['Time', 'Virtual_Zone_Name'], parse_dates=['Time'])[['Time', 'Virtual_Zone_Name']]
                    t['Time'] = t['Time'].dt.date
                    t.drop_duplicates(keep='first', inplace=True)
                    pos_weight = (len(t)*timedelta(days=1)/dm.time_step-o)/o
                    model = BCLSTM_Area_s1(in_size=in_size-1, hidden_size=100, num_layers=3, lr=args.lr, l2=args.l2, pos_weight=pos_weight)
                else:
                    model = BCLSTM_Area_s1.load_from_checkpoint(ckpt)
            else:
                if ckpt is None:
                    t = len(pd.read_csv(Path('.') / 'data' / 'processed' / 'locations.csv', usecols=[0]))
                    pos_weight = (t-o)/o
                    model = BC_Area_s1(hidden_layers=(60*in_size, 30*in_size, 15*in_size),
                    in_size=in_size, lr=args.lr, l2=args.l2, pos_weight=pos_weight)
                else:
                    model = BC_Area_s1.load_from_checkpoint(ckpt)
        elif s == 'stage_2':
            in_size = int(3+len(cars)+3*len(area_centers)) # Date, car models and location (current), amount of cars in all zones, and demand
            out_size = len(area_centers)
            if ckpt is None:
                model = BC_Area_s2(hidden_layers=(30*in_size, int(15*in_size+5*out_size), 10*out_size),
                in_out=(in_size, out_size), lr=args.lr, l2=args.l2)
            else:
                model = BC_Area_s2.load_from_checkpoint(ckpt)
    elif s == 'dqn':
        in_size = int(3+len(cars)+3*len(area_centers)) # Date, car models and location (current), amount of cars in all zones, and demand
        out_size = len(area_centers)
        buffer, agent = create_buffer_agent(args)
        dm = QDataModule(buffer=buffer, sample_size=args.sample_size, batch_size=args.batch_size, num_workers=args.num_workers)
        model = DQN(
            in_out=(in_size, out_size), buffer=buffer, agent=agent, 
            hidden_layers=(30*in_size, int(15*in_size+5*out_size), 10*out_size),
            lr=args.lr, l2=args.l2, gamma=args.gamma, sync_rate=args.sync_rate,
            eps_stop=args.eps_stop, eps_start=args.eps_start, eps_end=args.eps_end, double_dqn=args.ddqn)
    return model, dm

def run_stage(args, stage):
    ckpt = get_ckpt_path(args, stage)
    args_trainer = deepcopy(args)
    args_trainer.default_root_dir = args.default_root_dir + '/' + stage
    if stage == 's1' or stage == 's2':
        trainer = pl.Trainer.from_argparse_args(args_trainer, callbacks=[EarlyStopping(monitor='measure', mode='max', patience=10)])
    elif stage == 'dqn':
        trainer = pl.Trainer.from_argparse_args(args_trainer)
    model, dm = setup_model_dm(args=args, s=stage, ckpt=ckpt)
    if args.fit:
        if args.auto_lr_find or (args.auto_scale_batch_size is not None):
            trainer.tune(model, dm)
        if ckpt is None:
            trainer.fit(model, dm)
        else:
            trainer.fit(model, dm, ckpt_path=ckpt)
    if args.test and not args.dqn:
        if ckpt is None or args.fit:
            trainer.test(model, dm)
        else:
            trainer.test(model, dm, ckpt_path=ckpt)

def main(args):
    pl.seed_everything(seed=args.seed, workers=True)
    if args.s1:
        run_stage(args, "stage_1")
    if args.s2:
        run_stage(args, "stage_2")
    if args.dqn:
        run_stage(args, "dqn")

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    save_dir = Path('.') / "models"
    parser.set_defaults(default_root_dir=str(save_dir))
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size used in the datamodule. Will be ignored if --auto_scale_batch_size')
    parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate used in training. Will be ignored if --auto_lr_find')
    parser.add_argument('--l2', default=1e-6, type=float, help='L2 regularisation used in training')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for DataLoader')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle data before splitting in train-test sets')
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--load_version', type=int, help='Load specific version from lightning_logs/version_#/checkpoint/. Loads latest checkpoint to test or continue training')
    parser.add_argument('--fit', action='store_true', help='Fit model from start or checkpoint')
    parser.add_argument('--test', action='store_true', help='Test model. If --fit, will test after fitting. If --load_version, will test after loading the checkpoint')
    parser.add_argument('--s1', action='store_true', help='Use stage 1')
    parser.add_argument('--s2', action='store_true', help='Use stage 2')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM model')
    parser.add_argument('--dqn', action='store_true', help='Use DQN model')
    parser.add_argument('--ddqn', action='store_true', help='Use Double DQN loss')
    parser.add_argument('--buffer_capacity', default=1000000, type=int, help='Buffer capacity for DQN')
    parser.add_argument('--sample_size', default=21936, type=int, help='Sample size from buffer')
    parser.add_argument('--gamma', default=0.99, type=float, help='Gamma for DQN')
    parser.add_argument('--warm_up', default=21936, type=int, help='Number of warm up random steps for DQN')
    parser.add_argument('--eps_start', default=1.0, type=float, help='Epsilon at start of training for DQN')
    parser.add_argument('--eps_end', default=0.01, type=float, help='Epsilon at end of training for DQN')
    parser.add_argument('--eps_stop', default=21936, type=int, help='Stop reducing epsilon after --eps_stop steps')
    parser.add_argument('--sync_rate', default=500, type=int, help='Sync Q and target network every --sync_rate steps')
    parser.add_argument('--n_zones', default=20, type=int, help='Number of zones when rebuilding dataset')
    args = parser.parse_args()
    warnings.filterwarnings(action="ignore", category=pl.utilities.warnings.LightningDeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    main(args)