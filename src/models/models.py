from pathlib import Path
from datetime import datetime
import pickle
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import F1
from src.data.datasets import QDataset, ReplayBuffer
from src.models.simulator import Agent

class BC_Area_s1(pl.LightningModule): # Step 1: Decide to move or not
    def __init__(self, in_size, hidden_layers=(100, 50), lr=1e-5, l2=1e-5, pos_weight=25):
        super().__init__()

        self.in_features = in_size
        self.lr = lr
        self.l2 = l2
        self.f1_train = F1(num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.f1_val = F1(num_classes=1)
        self.f1_test = F1(num_classes=1)
        
        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.layers_hidden.append(nn.Dropout(0.25))
            self.layers_hidden.append(nn.ReLU())
            self.in_features = neurons

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], 1))
        self.layers_hidden = nn.Sequential(*self.layers_hidden)
        
        self.save_hyperparameters()

    def forward(self, s):
        return self.layers_hidden(s.float())

    def training_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s).squeeze()
        loss = self.criterion(a_logits, a.float())
        self.f1_train(torch.sigmoid(a_logits), a)
        self.log('Loss', loss, logger=True, sync_dist=True)
        self.log('F1 score', self.f1_train, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s).squeeze()
        self.f1_val(torch.sigmoid(a_logits), a)
        self.log('measure', self.f1_val, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        s, a = batch
        self.f1_test(torch.sigmoid(self(s).squeeze()), a)
        self.log('F1 score', self.f1_test, sync_dist=True)
        return {'F1 score': self.f1_test}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class BC_Area_s2(pl.LightningModule): # Step 2: Decide where to move, given that it was decided to move
    def __init__(self, in_out, hidden_layers=(100, 50), lr=1e-5, l2=1e-5):
        super().__init__()

        self.in_features = in_out[0]
        self.n_areas = in_out[1]
        self.lr = lr
        self.l2 = l2
        self.criterion = nn.BCEWithLogitsLoss()
         
        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.layers_hidden.append(nn.Dropout(0.25))
            self.layers_hidden.append(nn.ReLU())
            self.in_features = neurons

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], self.n_areas))
        self.layers_hidden = nn.Sequential(*self.layers_hidden)
        
        self.save_hyperparameters()

    def forward(self, s):
        return self.layers_hidden(s.float())

    def training_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s)
        loss = self.criterion(a_logits, a.float())
        dist = 1-torch.gather(F.softmax(a_logits, dim=1), 1, torch.argmax(a, dim=1).unsqueeze(1))
        self.log('Loss', loss, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        self.log('Distance', torch.sum(dist), on_epoch=True, on_step=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s)
        dist = 1-torch.gather(F.softmax(a_logits, dim=1), 1, torch.argmax(a, dim=1).unsqueeze(1))
        self.log('measure', -torch.sum(dist), logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s)
        acc = torch.argmax(a_logits, dim=1)==torch.argmax(a, dim=1) # Total accuracy
        dist = 1-torch.gather(F.softmax(a_logits, dim=1), 1, torch.argmax(a, dim=1).unsqueeze(1))
        self.log('Accuracy', acc.sum()/len(acc), sync_dist=True)
        self.log('Distance', torch.sum(dist), sync_dist=True)
        return {'Accuracy': acc.sum()/len(acc), 'Distance': torch.sum(dist)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class BCLSTM_Area_s1(pl.LightningModule): # Step 1: Decide to move or not
    def __init__(self, in_size, hidden_size=100, num_layers=3, lr=1e-5, l2=1e-5, pos_weight=1000):
        super().__init__()

        self.in_features = in_size
        self.lr = lr
        self.l2 = l2
        self.f1_train = F1(num_classes=1)
        self.f1_val = F1(num_classes=1)
        self.f1_test = F1(num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
        self.lstm = nn.LSTM(
            input_size=self.in_features, hidden_size=hidden_size, 
            num_layers=num_layers, dropout=0.25, batch_first=True)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(hidden_size, 1)

        self.save_hyperparameters()

    def forward(self, s):
        s, *_ = self.lstm(s.float())
        s = self.linear(s.squeeze())
        return s

    def training_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s).squeeze()
        loss = self.criterion(a_logits, a.float())
        self.f1_train(torch.sigmoid(a_logits.reshape(-1)), a.reshape(-1))
        self.log('Loss', loss, logger=True, sync_dist=True)
        self.log('F1 score', self.f1_train, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s).squeeze()
        self.f1_val(torch.sigmoid(a_logits.reshape(-1)), a.reshape(-1))
        self.log('measure', self.f1_val, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        s, a = batch
        self.f1_test(torch.sigmoid(self(s).reshape(-1)), a.reshape(-1))
        self.log('F1 score', self.f1_test, sync_dist=True)
        return {'F1 score': self.f1_test}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class Q(nn.Module):
    def __init__(self, in_out, hidden_layers=(100, 50)):
        super().__init__()
        self.in_features = in_out[0]
        
        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.in_features, neurons))
            self.layers_hidden.append(nn.ReLU())
            self.in_features = neurons

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], in_out[1]))
        self.layers_hidden = nn.Sequential(*self.layers_hidden)
    
    def forward(self, x):
        return self.layers_hidden(x.float())

class DQN(pl.LightningModule):
    def __init__(
        self, in_out, hidden_layers=[50, 20], buffer_capacity=1000000, warm_up=21936, sample_size=21936, batch_size=32,
        num_workers=0, lr=1e-3, l2=1e-8, gamma=0.999, sync_rate=10, eps_stop=1000, eps_start=1.0, eps_end=0.01, time_end=datetime(2020, 4, 1, 0, 0, 0),
        double_dqn=False):
        super().__init__()

        self.save_hyperparameters()
        self.create_buffer_agent()

        self.Q = Q(in_out, hidden_layers)
        self.target = Q(in_out, hidden_layers)

        self.episode_reward = 0
        self.total_reward = 0

    def forward(self, s):
        return self.Q(s)

    def loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        states = states.float()
        states.requires_grad_(True)
        state_action_values = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.hparams.double_dqn:
                next_outputs = self.Q(next_states)
                next_state_acts = next_outputs.max(1)[1].unsqueeze(-1)  # take action at the index with the highest value
                next_tgt_out = self.target(next_states)
                # Take the value of the action chosen by the train network
                next_state_values = next_tgt_out.gather(1, next_state_acts).squeeze(-1)
                next_state_values[dones] = 0.0
                next_state_values = next_state_values.detach()
            else:
                next_state_values = self.target(next_states).max(1)[0]
                next_state_values[dones] = 0.0
                next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return F.mse_loss(state_action_values, expected_state_action_values)

    def training_step(self, batch, batch_idx):
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_stop)

        reward, done = self.agent.play_step(self.Q, epsilon, self.device)
        self.episode_reward += reward

        loss = self.loss(batch)

        if self.trainer._distrib_type in {pl.utilities.DistributedType.DP, pl.utilities.DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        if self.global_step % self.hparams.sync_rate == 0:
            self.target.load_state_dict(self.Q.state_dict())

        status = {
            "steps": torch.tensor(self.global_step).to(self.device),
            "total_reward": torch.tensor(self.total_reward).to(self.device),
        }

        self.log("Total_reward", self.total_reward, sync_dist=True)
        self.log("Loss", loss, sync_dist=True)
        
        return {"loss": loss, "progress_bar": status}

    def configure_optimizers(self):
        return torch.optim.Adam(self.Q.parameters(), lr=self.hparams.lr)
    
    def train_dataloader(self):
        train_data = QDataset(self.buffer, self.hparams.sample_size)
        return DataLoader(train_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def create_buffer_agent(self):
        if not (Path('.') / 'data' / 'processed' / 'buffer.pkl').is_file():
            buffer = ReplayBuffer(self.hparams.buffer_capacity)
            agent = Agent(buffer, time_end=self.hparams.time_end)
            print('Populating buffer')
            for _ in tqdm(range(self.hparams.warm_up)):
                agent.play_step(net=None, epsilon=1.0)
            with open(str(Path('.') / 'data' / 'processed' / 'buffer.pkl'), 'wb') as f:
                    pickle.dump(buffer, f)
        else:
            with open(str(Path('.') / 'data' / 'processed' / 'buffer.pkl'), 'rb') as f:
                buffer = pickle.load(f)
            agent = Agent(buffer, time_end=self.hparams.time_end)
        self.buffer, self.agent = buffer, agent
