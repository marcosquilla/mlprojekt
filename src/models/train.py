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
    dm = CarDataModule(shuffle=True, batch_size=32, num_workers=2)
    area_centers = pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0)
    in_size = int(3+len(dm.cars)+len(area_centers))
    out_size = len(area_centers)
    model = BC_Car(hidden_layers=(30*in_size, int(15*in_size+5*out_size), 10*out_size), in_out=(in_size, out_size))
    return model, dm

if __name__ == "__main__":
    pl.seed_everything(seed=42, workers=True)
    if torch.cuda.device_count()>0:
        trainer = pl.Trainer(gpus=-1, precision=16)
    else:
        trainer = pl.Trainer(log_every_n_steps=5)

    trainer = pl.Trainer(fast_dev_run=10)

    model, dm = Car_train()
    trainer.fit(model, dm)