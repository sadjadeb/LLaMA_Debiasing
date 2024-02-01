import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from utils import get_model_inputs, get_multiple_model_inputs


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, self.output_dim)
        )

    def forward(self, state):
        return self.fc(state)


class DQNAgent:
    def __init__(self, input_dim, dataset, learning_rate=3e-4, gamma=0.99, buffer=None, tau=0.999, pre_trained_model=None):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.model = DQN(input_dim, 1)
        if pre_trained_model:
            self.model = pre_trained_model
        base_opt = torch.optim.Adam(self.model.parameters())
        self.dataset = dataset
        self.MSE_loss = nn.MSELoss()
        self.replay_buffer = buffer
        self.optimizer = base_opt

    def get_action(self, state, dataset=None):
        if dataset is None:
            dataset = self.dataset
        inputs = get_multiple_model_inputs(state, state.remaining, dataset)
        model_inputs = autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0))
        expected_returns = self.model.forward(model_inputs)
        value, index = expected_returns.max(1)
        return state.remaining[index[0]]

    def compute_loss(self, batch, dataset, verbose=False):
        states, actions, rewards, next_states, dones = batch
        model_inputs = np.array([get_model_inputs(states[i], actions[i], dataset) for i in range(len(states))])
        model_inputs = torch.FloatTensor(model_inputs)

        rewards = np.array(rewards)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        curr_Q = self.model.forward(model_inputs)
        stacked_arrays = []
        for i in range(len(next_states[0].remaining)):
            temp = get_model_inputs(next_states[0], next_states[0].remaining[i], dataset)
            stacked_arrays.append(temp)
        if stacked_arrays:
            result_array = np.vstack(stacked_arrays)
            model_inputs = torch.FloatTensor(result_array)
            next_Q = self.model.forward(model_inputs)
            max_next_Q = torch.max(next_Q, 1)[0].max().item()
            expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q
        else:
            expected_Q = rewards.squeeze(1)
        loss = self.MSE_loss(curr_Q.squeeze(0), expected_Q.detach())
        return loss

    def update(self, batch_size, verbose=False):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch, self.dataset, verbose)
        train_loss = loss.float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_loss
