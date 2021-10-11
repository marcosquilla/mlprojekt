from ..data.load_data import DataModule
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression

class BC_FFNN(pl.LightningModule):
    def __init__(self, hidden_layers=[1000, 500]):
        super().__init__()

        self.in_features = 603

        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.layers_hidden.append(nn.ReLU())
            self.in_features = neurons

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], 550))
        self.layers_hidden.append(nn.Sigmoid())
        self.layers_hidden = nn.Sequential(*self.layers_hidden)

    def forward(self, x):
        return self.layers_hidden(x).round()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.binary_cross_entropy(self(x), y)
        self.log('Loss', loss, on_epoch=True, logger=True)
        return loss



if __name__ == "__main__":
    print(torch.cuda.device_count())
    pl.seed_everything(seed=42)
    dm = DataModule(shuffle_time=True, batch_size=10)
    model = BC_FFNN()
    trainer = pl.Trainer(gpus=1, precision=16)
    dm.setup(stage='fit')

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