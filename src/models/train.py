from pathlib import Path
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from .models import BC_Fleet, BC_Car
from ..data.load_data import FleetDataModule, CarDataModule

#TODO: Time_window vs time_delta tuning

def Fleet_train(n_actions=5):
    dm = FleetDataModule(shuffle_time=True, batch_size=16, num_workers=int(os.cpu_count()/2), n_actions=n_actions)
    dm.setup(stage='fit')
    s, a, *_ = next(iter(dm.train_dataloader()))
    in_size = s.shape[1]
    out_size = a.shape[1]*a.shape[2]
    model = BC_Fleet(hidden_layers=(20*in_size, int(10*in_size+5*out_size), 10*out_size), in_out=(in_size, out_size), n_actions=n_actions)
    return model, dm

def Car_train():
    area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
    cars = pd.unique(pd.read_csv(Path('.') / 'data' / 'interim' / 'rental.csv', usecols=[2]).iloc[:,0])
    in_size = int(3+len(cars)+2*len(area_centers))
    out_size = len(area_centers)
    dm = CarDataModule(shuffle=True, batch_size=16, num_workers=0)
    model = BC_Car(hidden_layers=(3*in_size, int(1.5*in_size+0.5*out_size), 1*out_size), in_out=(in_size, out_size))
    return model, dm

if __name__ == "__main__":
    pl.seed_everything(seed=42, workers=True)
    fdr = 10
    if torch.cuda.device_count()>0:
        trainer = pl.Trainer(gpus=-1, precision=16, fast_dev_run=fdr, weights_summary='full')
    else:
        trainer = pl.Trainer(log_every_n_steps=5, fast_dev_run=fdr, weights_summary='full')

    model, dm = Car_train()
    trainer.fit(model, dm)