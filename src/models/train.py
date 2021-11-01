from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from .models import BC_Fleet, BC_Car
from ..data.datamodules import FleetDataModule, CarDataModule
import warnings

#TODO: Time_window vs time_delta tuning. Test step. Data download.

def Fleet_train(batch_size=8, num_workers=0, n_actions=5, shuffle=True):
    dm = FleetDataModule(shuffle_time=shuffle, batch_size=batch_size, num_workers=num_workers, n_actions=n_actions)
    dm.setup(stage='fit')
    s, a, *_ = next(iter(dm.train_dataloader()))
    in_size = s.shape[1]
    out_size = a.shape[1]*a.shape[2]
    model = BC_Fleet(hidden_layers=(20*in_size, int(10*in_size+5*out_size), 10*out_size), in_out=(in_size, out_size), n_actions=n_actions)
    return model, dm

def Car_train(batch_size=128, lr=1e-5, l2=0, num_workers=0, shuffle=True):
    dm = CarDataModule(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
    cars = pd.unique(pd.read_csv(Path('.') / 'data' / 'interim' / 'rental.csv', usecols=[2]).iloc[:,0])
    in_size = int(3+len(cars)+2*len(area_centers)) # Date, car models and locations, and demand
    out_size = len(area_centers)
    model = BC_Car(hidden_layers=(30*in_size, int(15*in_size+5*out_size), 10*out_size), in_out=(in_size, out_size), lr=lr, l2=l2)
    return model, dm

def main(args):
    pl.seed_everything(seed=args.seed, workers=True)
    trainer = pl.Trainer.from_argparse_args(args)
    model, dm = Car_train(batch_size=args.batch_size, lr=args.lr, l2=args.l2, num_workers=args.num_workers, shuffle=args.shuffle)
    if args.auto_lr_find or (args.auto_scale_batch_size is not None):
        trainer.tune(model, dm)
    trainer.fit(model, dm)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--l2', default=1e-5, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    warnings.filterwarnings(action="ignore", category=pl.utilities.warnings.LightningDeprecationWarning)

    main(args)