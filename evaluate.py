import glob
import os
import time

from environment.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent
try:
    from renderer.pygame_renderer import PygameRenderer
except ModuleNotFoundError:
    PygameRenderer = None


MODEL_PATH = os.getenv("SNAKE_EVAL_MODEL", "checkpoints/snake_dqn_best.pth")
MAX_STEPS = int(os.getenv("SNAKE_EVAL_MAX_STEPS", "10000"))
RENDER = os.getenv("SNAKE_EVAL_RENDER", "1") == "1"


def find_model_path(path):
    if os.path.exists(path):
        return path

    candidates = glob.glob("checkpoints/snake_dqn_*.pth")
    if not candidates:
        raise FileNotFoundError("No checkpoint found. Train first with python train.py.")

    return max(candidates, key=os.path.getmtime)


env = SnakeEnv()

state_size = env.reset().shape[0]
action_size = 3

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
)

# Checkpoint này được train với kiến trúc 28 -> 256 -> 256 -> 3.
# Chỉ cần online Q-network khi evaluate, không cần optimizer/target network.
model_path = find_model_path(MODEL_PATH)
agent.load(model_path)
agent.qnet.eval()

# Không random nữa
agent.epsilon = 0.0

renderer = None
if RENDER and PygameRenderer is not None:
    renderer = PygameRenderer(env)
elif RENDER:
    print("pygame is not installed; running console evaluation without rendering.")

state = env.reset()

done = False
steps = 0

while not done and steps < MAX_STEPS:

    if renderer is not None:
        renderer.render()

    action = agent.act(state)

    next_state, reward, done, _ = env.step(action)

    state = next_state
    steps += 1

    if renderer is not None:
        time.sleep(0.08)

print(f"Model: {model_path}")
print(f"Final Score: {env.score}")
print(f"Steps: {steps}")

if renderer is not None:
    renderer.close()
