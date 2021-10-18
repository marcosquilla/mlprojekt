from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..data.load_data import FleetDataModule

class BC_FFNN(pl.LightningModule):
    def __init__(self, hidden_layers=(100, 50), in_out=(603, 555), n_actions:int=5):
        super().__init__()

        self.in_features = in_out[0]

        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.layers_hidden.append(nn.Dropout(0.25))
            self.layers_hidden.append(nn.ReLU())
            self.in_features = neurons

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], in_out[1]))
        self.layers_hidden = nn.Sequential(*self.layers_hidden)

        self.n_areas = len(pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0))
        self.n_actions = n_actions

    def forward(self, s):
        a = self.layers_hidden(s.float())
        a = a.reshape(a.shape[0], self.n_actions, -1)
        a_pred = torch.zeros_like(a)
        a_pred[:, :, :-2*self.n_areas] = F.softmax(a[:, :, :-2*self.n_areas], dim=2) #Car
        a_pred[:, :, -2*self.n_areas:-self.n_areas] = F.softmax(a[:, :, -2*self.n_areas:-self.n_areas], dim=2) #Origin
        a_pred[:, :, -self.n_areas:] = F.softmax(a[:, :, -self.n_areas:], dim=2) #Destination
        return a_pred

    def training_step(self, batch, batch_idx):
        s, a, *_ = batch
        a_logits = self(s)
        loss_car = F.binary_cross_entropy_with_logits(a_logits[:, :, :-2*self.n_areas], a[:, :, :-2*self.n_areas].float())
        loss_origin = F.binary_cross_entropy_with_logits(a_logits[:, :, -2*self.n_areas:-self.n_areas], a[:, :, -2*self.n_areas:-self.n_areas].float())
        loss_destination = F.binary_cross_entropy_with_logits(a_logits[:, :, -self.n_areas:], a[:, :, -self.n_areas:].float())
        loss = loss_car + loss_origin + loss_destination
        self.log('Loss', loss, on_epoch=True, logger=True)
        a_pred = torch.zeros_like(a_logits, dtype=torch.int8).scatter(2, torch.argmax(a_logits[:, :, :-2*self.n_areas], dim=2).unsqueeze(1), 1)
        a_pred = a_pred.scatter(2, a_pred.shape[2]-2*self.n_areas+torch.argmax(a_logits[:, :, -2*self.n_areas:-self.n_areas], dim=2).unsqueeze(1), 1)
        a_pred = a_pred.scatter(2, a_pred.shape[2]-self.n_areas+torch.argmax(a_logits[:, :, -self.n_areas:], dim=2).unsqueeze(1), 1)
        f1_cars = f1_score(a[:, :, :-2*self.n_areas].cpu().detach().numpy().reshape(-1), a_pred[:, :, :-2*self.n_areas].cpu().detach().numpy().reshape(-1))
        f1_origin = f1_score(a[:, :, -2*self.n_areas:-self.n_areas].cpu().detach().numpy().reshape(-1), a_pred[:, :, -2*self.n_areas:-self.n_areas].cpu().detach().numpy().reshape(-1))
        f1_destination = f1_score(a[:, :, -self.n_areas:].cpu().detach().numpy().reshape(-1), a_pred[:, :, -self.n_areas:].cpu().detach().numpy().reshape(-1))
        self.log('F1 cars', f1_cars)
        self.log('F1 origin', f1_origin)
        self.log('F1 destination', f1_destination)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


if __name__ == "__main__":
    pl.seed_everything(seed=42, workers=True)
    n_actions = 5
    dm = FleetDataModule(shuffle_time=True, batch_size=16, num_workers=int(os.cpu_count()/2), n_actions=n_actions)
    dm.setup(stage='fit')
    s, a, *_ = next(iter(dm.train_dataloader()))
    in_size = s.shape[1]
    out_size = a.shape[1]*a.shape[2]
    model = BC_FFNN(hidden_layers=(20*in_size, int(10*in_size+5*out_size), 10*out_size), in_out=(in_size, out_size), n_actions=n_actions)

    if torch.cuda.device_count()>0:
        trainer = pl.Trainer(gpus=-1, precision=16)
    else:
        trainer = pl.Trainer(log_every_n_steps=5)

    #trainer = pl.Trainer(fast_dev_run=10)

    trainer.fit(model, dm)