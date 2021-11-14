import os
from copy import deepcopy
import glob
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from .models import BC_Car_s1, BC_Car_s2
from src.data.datamodules import CarDataModule_s1, CarDataModule_s2
import warnings

#TODO: Time_window vs time_delta tuning. Data download.

def Car_train_s1(batch_size, lr, l2, num_workers, shuffle, ckpt=None):
    dm = CarDataModule_s1(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
    cars = pd.unique(pd.read_csv(Path('.') / 'data' / 'interim' / 'rental.csv', usecols=[2]).iloc[:,0])
    in_size = int(3+len(cars)+2*len(area_centers)) # Date, car models and locations, and demand
    if ckpt is None:
        model = BC_Car_s1(hidden_layers=(30*in_size, 20*in_size, 10*in_size), in_size=in_size, lr=lr, l2=l2)
    else:
        model = BC_Car_s1.load_from_checkpoint(ckpt)
    return model, dm

def Car_train_s2(batch_size, lr, l2, num_workers, shuffle, ckpt=None):
    dm = CarDataModule_s2(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
    cars = pd.unique(pd.read_csv(Path('.') / 'data' / 'interim' / 'rental.csv', usecols=[2]).iloc[:,0])
    in_size = int(3+len(cars)+2*len(area_centers)) # Date, car models and locations, and demand
    out_size = len(area_centers)
    if ckpt is None:
        model = BC_Car_s2(hidden_layers=(30*in_size, int(15*in_size+5*out_size), 10*out_size), in_out=(in_size, out_size), lr=lr, l2=l2)
    else:
        model = BC_Car_s2.load_from_checkpoint(ckpt)
    return model, dm

def run_stage(args, stage, model, dm):
    ckpt = get_ckpt_path(stage)
    args_trainer = deepcopy(args)
    args_trainer.default_root_dir = args.default_root_dir + '/' + stage
    trainer = pl.Trainer.from_argparse_args(args_trainer)
    if args.fit:
        if args.auto_lr_find or (args.auto_scale_batch_size is not None):
            trainer.tune(model, dm)
        if ckpt is None:
            trainer.fit(model, dm)
        else:
            trainer.fit(model, dm, ckpt_path=ckpt)
    if args.test:
        if ckpt is None or args.fit:
            trainer.test(model, dm)
        else:
            trainer.test(model, dm, ckpt_path=ckpt)

def get_ckpt_path(stage):
    if args.load_version is not None:
        ckpt_path = (Path('.') / 'models' / stage / 'lightning_logs' / str('version_' + str(args.load_version)) / 'checkpoints' / '*.ckpt')
        return Path('.') / max(glob.glob(str(ckpt_path)), key=os.path.getmtime)
    else:
        return None

def main(args):
    pl.seed_everything(seed=args.seed, workers=True)
    if args.s1:
        model_s1, dm_s1 = Car_train_s1(batch_size=args.batch_size, lr=args.lr, l2=args.l2,
         num_workers=args.num_workers, shuffle=args.shuffle, ckpt=get_ckpt_path("stage_1"))
        run_stage(args, "stage_1", model_s1, dm_s1)
    if args.s2:
        model_s2, dm_s2 = Car_train_s2(batch_size=args.batch_size, lr=args.lr, l2=args.l2,
         num_workers=args.num_workers, shuffle=args.shuffle, ckpt=get_ckpt_path("stage_2"))
        run_stage(args, "stage_2", model_s2, dm_s2)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    save_dir = Path('.') / "models"
    parser.set_defaults(default_root_dir=str(save_dir))
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size used in the datamodule. Will be ignored if --auto_scale_batch_size')
    parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate used in training. Will be ignored if --auto_lr_find')
    parser.add_argument('--l2', default=1e-5, type=float, help='L2 regularisation used in training')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for DataLoader')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle data before splitting in train-test sets')
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--load_version', type=int, help='Load specific version from lightning_logs/version_#/checkpoint/. Loads latest checkpoint to test or continue training')
    parser.add_argument('--fit', action='store_true', help='Fit model from start or checkpoint')
    parser.add_argument('--test', action='store_true', help='Test model. If --fit, will test after fitting. If --load_version, will test after loading the checkpoint')
    parser.add_argument('--s1', action='store_true', help='Use stage 1')
    parser.add_argument('--s2', action='store_true', help='Use stage 2')
    args = parser.parse_args()
    warnings.filterwarnings(action="ignore", category=pl.utilities.warnings.LightningDeprecationWarning)

    main(args)