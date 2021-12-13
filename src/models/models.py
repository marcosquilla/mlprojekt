import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import F1, Precision

class BC_Car_s1(pl.LightningModule): # Step 1: Decide to move or not
    def __init__(self, in_size, hidden_layers=(100, 50), lr=1e-5, l2=1e-5, pos_weight=1000):
        super().__init__()

        self.in_features = in_size
        self.lr = lr
        self.l2 = l2
        self.f1_train = F1(num_classes=1)
        self.pre_train = Precision(num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
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
        self.f1_train(torch.round(torch.sigmoid(a_logits)), a)
        self.pre_train(torch.round(torch.sigmoid(a_logits)), a)
        self.log('Loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('F1 score', self.f1_train, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('Precision', self.pre_train, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        s, a = batch
        self.f1_test(torch.round(torch.sigmoid(self(s).squeeze())), a)
        self.log('F1 score', self.f1_test, sync_dist=True)
        return {'F1 score': self.f1_test}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class BC_Car_s2(pl.LightningModule): # Step 2: Decide where to move, given that it was decided to move
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
        self.log('Distance to target', torch.sum(dist), on_epoch=True, on_step=False, logger=True, sync_dist=True)
        return loss

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

class BCLSTM_Car_s1(pl.LightningModule): # Step 1: Decide to move or not
    def __init__(self, in_size, hidden_size=100, num_layers=3, lr=1e-5, l2=1e-5, pos_weight=1000):
        super().__init__()

        self.in_features = in_size
        self.lr = lr
        self.l2 = l2
        self.f1_train = F1(num_classes=1)
        self.f1_test = F1(num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
        self.lstm = nn.LSTM(
            input_size=self.in_features, hidden_size=hidden_size, 
            num_layers=num_layers, dropout=0.25, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

        self.save_hyperparameters()

    def forward(self, s):
        s, *_ = self.lstm(s.float())
        return self.linear(s.squeeze())

    def training_step(self, batch, batch_idx):
        s, a = batch
        a_logits = self(s).squeeze()
        loss = self.criterion(a_logits, a.float())
        self.f1_train(torch.round(torch.sigmoid(a_logits)), a)
        self.log('Loss', loss, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log('F1 score', self.f1_train, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        s, a = batch
        self.f1_test(torch.round(torch.sigmoid(self(s).squeeze())), a)
        return {'F1 score': self.f1_test}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)

class BalanceModel(pl.LightningModule):
    def __init__(self, hidden_layers_policy=[50, 20], hidden_layers_Q=[50, 20],
     lr_policy=1e-3, lr_Q=1e-3, gamma=0.999, tau=0.01):
        super().__init__()
        
        self.criterion = nn.MSELoss()
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
        Qloss = self.criterion(self.Q_policy(s, a), y)
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