import numpy as np

from agents.replay_buffer import ReplayBuffer


buffer = ReplayBuffer(
    capacity=100,
)

for i in range(20):

    state = np.random.rand(28)

    action = np.random.randint(0,3)

    reward = np.random.randn()

    next_state = np.random.rand(28)

    done = np.random.choice([0,1])

    buffer.push(
        state,
        action,
        reward,
        next_state,
        done,
    )

print("Buffer size:", len(buffer))

states, actions, rewards, next_states, dones = buffer.sample(8)

print(states.shape)
print(actions.shape)
print(rewards.shape)
print(next_states.shape)
print(dones.shape)
