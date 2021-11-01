from pathlib import Path
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class BC_Fleet(pl.LightningModule):
    def __init__(self, hidden_layers=(100, 50), in_out=(603, 555), n_actions:int=5, lr=1e-5, l2=1e-5):
        super().__init__()

        self.in_features = in_out[0]
        self.lr = lr
        self.l2 = l2

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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class BC_Car(pl.LightningModule):
    def __init__(self, in_out, hidden_layers=(100, 50), lr=1e-5, l2=1e-5):
        super().__init__()

        self.in_features = in_out[0]
        self.n_areas = in_out[1]
        self.lr = lr
        self.l2 = l2
         
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
        loss = F.binary_cross_entropy_with_logits(a_logits, a.float())
        acc = torch.argmax(a_logits, dim=1)==torch.argmax(a, dim=1) # Total accuracy
        acc_dif = acc[torch.argmax(s[-len(self.n_areas):], dim=1)!=torch.argmax(a, dim=1)] # Accuracy only where cars were moved
        self.log('Loss', loss, on_epoch=True, logger=True)
        self.log('Accuracy total', acc.sum()/len(acc), on_epoch=True, logger=True)
        self.log('Accuracy moves only', acc_dif.sum()/len(acc_dif), on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        acc = torch.argmax(self(s), dim=1)==torch.argmax(a, dim=1) # Total accuracy
        acc_dif = acc[torch.argmax(s[-len(self.n_areas):], dim=1)!=torch.argmax(a, dim=1)] # Accuracy only where cars were moved
        self.log('Accuracy total', acc.sum()/len(acc), on_epoch=True, logger=True)
        self.log('Accuracy moves only', acc_dif.sum()/len(acc_dif), on_epoch=True, logger=True)
        return {'Accuracy total': acc.sum()/len(acc), 'Accuracy moves only': acc_dif.sum()/len(acc_dif)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class BalanceModel(pl.LightningModule):
    def __init__(self, hidden_layers_policy=[50, 20], hidden_layers_Q=[50, 20],
     lr_policy=1e-3, lr_Q=1e-3, gamma=0.999, tau=0.01):
        super().__init__()
        
        self.lr_policy = lr_policy
        self.lr_Q = lr_Q
        self.gamma = gamma
        self.tau = tau

        self.policy_net = Policy(hidden_layers=hidden_layers_policy)
        self.target_net = Policy(hidden_layers=hidden_layers_policy)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.freeze()

        self.Q_policy = Q(hidden_layers=hidden_layers_Q)
        self.Q_target = Q(hidden_layers=hidden_layers_Q)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.freeze()

        self.automatic_optimization = False # Manual optimization process required

    def forward(self, x):
        s, *_ = x
        return self.policy_net(s)

    def training_step(self, batch, batch_idx):
        s, a, s1, r = batch
        q_target = self.Q_target(s1, self.target_net(s1)) # Use the Q network to estimate the rewards from the target network's actions
        y = q_target * self.gamma + r # Compute the estimated return, by adding the return for the next action and the reward for the current action
        # Optimize both networks
        optimizer_policy, optimizer_Q = self.optimizes()

        # Compute mean-squared Bellman error for the Q network(MSBE)
        optimizer_Q.zero_grad()
        Qloss = F.mse_loss(self.Q_policy(s, a), y)
        self.manual_backward(Qloss, optimizer_Q)
        optimizer_Q.step()
        # The loss for the policy is just the negative value of the Q function. By doing this we look for the actions that maximise the return
        optimizer_policy.zero_grad()
        Piloss = -self.Q_policy(s, self.policy_net(s)).mean()
        self.manual_backward(Piloss, optimizer_policy)
        optimizer_policy.step()

        # Update the target networks with Polyak avergaing
        self.target_net.update_params(self.policy_net.state_dict(), self.tau)
        self.Q_target.update_params(self.Q_policy.state_dict(), self.tau)

        # Logging to TensorBoard by default
        self.log("Q_loss/y-1", Qloss/y-1)
        return Qloss/y-1

    def configure_optimizers(self):
        optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        optimizer_Q = torch.optim.Adam(self.Q_policy.parameters(), lr=self.lr_Q)
        return [optimizer_policy, optimizer_Q]

class Policy(nn.Module):
    def __init__(self, hidden_layers):
        super(Policy, self).__init__()
        
        self.hidden_in_features = 5
        
        self.layers_hidden = []
        for neurons in hidden_layers:
            self.layers_hidden.append(nn.Linear(self.hidden_in_features, neurons))
            self.hidden_in_features = neurons
            self.layers_hidden.append(nn.ReLU())

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], 1))
        self.layers_hidden = nn.Sequential(*self.layers_hidden)
    
    def forward(self, x):
        return self.layers_hidden(x)
        
    def update_params(self, new_params, tau):	# Polyak averaging to update the target network
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

class Q(nn.Module):
    def __init__(self, hidden_layers):
        super(Q, self).__init__()

        self.Lobs = nn.Linear(5, int(hidden_layers[0]/2))
        self.Lact = nn.Linear(1, int(hidden_layers[0]/2))
        self.hidden_in_features = hidden_layers[0]
        
        self.layers_hidden = []
        for neurons in hidden_layers[1:]:
            self.layers_hidden.append(nn.Linear(self.hidden_in_features, neurons))
            self.hidden_in_features = neurons
            self.layers_hidden.append(nn.ReLU())

        self.layers_hidden.append(nn.Linear(hidden_layers[-1], 1))

        self.layers_hidden = nn.Sequential(*self.layers_hidden)
    
    def forward(self, obs, act):
        xobs = F.relu(self.Lobs(obs))
        xact = F.relu(self.Lact(act))
        x = torch.cat((xobs, xact), 1)
        return self.layers_hidden(x)
        
    def update_params(self, new_params, tau):	# Polyak averaging to update the target network
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)