import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from models.qnet import QNet
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.01,
        buffer_size=100000,
        batch_size=512,
        target_update_freq=500,
        warmup_steps=1000,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps = warmup_steps

        self.device = (
            device if device else
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        # action_size is the output size, not QNet's second (hidden_size) argument.
        self.qnet = QNet(state_size, output_size=action_size).to(self.device)
        self.target_net = QNet(state_size, output_size=action_size).to(self.device)

        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)

        self.learn_step = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        valid_actions = self._valid_actions(state)
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        self.qnet.eval()

        with torch.no_grad():
            q_values = self.qnet(state).squeeze(0)
            invalid = [a for a in range(self.action_size) if a not in valid_actions]
            q_values[invalid] = -torch.inf

        self.qnet.train()

        return torch.argmax(q_values).item()

    def _valid_actions(self, state):
        # First 3 state entries are danger in exact action order.
        valid = [a for a in range(self.action_size) if state[a] < 0.5]
        if not valid:
            return list(range(self.action_size))

        # State entries 19..21 estimate reachable space after each action.
        # Prefer actions that do not squeeze the snake into a small pocket.
        length_ratio = state[26] if len(state) > 26 else 0.0
        min_space = max(0.08, min(0.85, length_ratio * 1.2 + 0.03))
        safe = [a for a in valid if state[19 + a] >= min_space]
        if safe:
            return safe

        return valid or list(range(self.action_size))

    def train_step(self):

        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None

        batch_size = min(len(self.memory), self.batch_size)

        states, actions, rewards, next_states, dones = \
            self.memory.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        current_q = self.qnet(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN: online network chooses; target network evaluates.
        with torch.no_grad():
            online_next_q = self.qnet(next_states)
            online_next_q = online_next_q.masked_fill(next_states[:, :self.action_size].bool(), -torch.inf)
            all_blocked = next_states[:, :self.action_size].bool().all(dim=1)
            online_next_q[all_blocked] = self.qnet(next_states[all_blocked])
            next_actions = online_next_q.argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)

            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1

        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()

    

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    def update_target_network(self):
        self.target_net.load_state_dict(
            self.qnet.state_dict()
        )

    def save(self, path):
        torch.save(
            {
                "qnet": self.qnet.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "learn_step": self.learn_step,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        if checkpoint.get("state_size", self.state_size) != self.state_size:
            raise ValueError(
                "Checkpoint uses a different state representation. "
                "Retrain with train.py after the environment upgrade."
            )
        try:
            self.qnet.load_state_dict(checkpoint["qnet"])
        except RuntimeError as exc:
            raise ValueError(
                "Legacy checkpoint has the old 3-neuron network. "
                "Retrain with train.py to create a compatible model."
            ) from exc
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint["epsilon"]
        self.learn_step = checkpoint.get("learn_step", 0)
