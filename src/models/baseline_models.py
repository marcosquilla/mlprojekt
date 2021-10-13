from ..data.load_data import DataModule
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

class BC_FFNN(pl.LightningModule):
    def __init__(self, hidden_layers=[100, 50], in_out=[603, 555], n_actions:int=5):
        super().__init__()

        self.in_features = in_out[0]

        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.layers_hidden.append(nn.ReLU())
            self.in_features = neurons

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], in_out[1]))
        self.layers_hidden.append(nn.Sigmoid())
        self.layers_hidden = nn.Sequential(*self.layers_hidden)

        self.n_areas = len(pd.read_csv((Path('.') / 'data' / 'processed' / 'areas.csv'), index_col=0))

    def forward(self, s):
        return self.layers_hidden(s.float()).round().int()

    def training_step(self, batch, batch_idx):
        s, a, *_ = batch
        loss = F.binary_cross_entropy(self(s), a.int())
        self.log('Loss', loss, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


if __name__ == "__main__":
    pl.seed_everything(seed=42)
    dm = DataModule(shuffle_time=True, batch_size=10, num_workers=2, n_actions=5)
    dm.setup(stage='fit')
    s, a, *_ = next(iter(dm.train_dataloader()))
    model = BC_FFNN(in_out=[s.shape[1], a.shape[1]], n_actions=5)

    if torch.cuda.device_count()>0:
        trainer = pl.Trainer(gpus=torch.cuda.device_count(), precision=16)
    else:
        trainer = pl.Trainer()
    
    trainer.fit(model, dm)
    # clf = LogisticRegression(n_jobs=-1, verbose=1)
    # s, a, *_ = next(iter(dm.train_dataloader()))
    # clf.fit(s, a)

    # for batch in tqdm(dm.train_dataloader()):
    #     s, a, *_ = batch
    #     clf.partial_fit(s.detach().numpy().squeeze().reshape(1, -1), a.detach().numpy().squeeze())
    
    #dm.setup(stage='test')
    # s, a, *_ = next(iter(dm.test_dataloader()))
    # print(clf.score(s, a))

    # batch_acc = []
    # for batch in tqdm(dm.test_dataloader()):
    #     s, a, *_ = batch
    #     batch_acc.append(clf.score(s.detach().numpy().squeeze().reshape(1, -1), a.detach().numpy().squeeze()))
    # batch_acc = np.array(batch_acc)
    # np.savetxt('reports/baseline_acc.csv', batch_acc, delimiter=',')