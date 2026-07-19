import os
import csv
from datetime import datetime

from environment.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ======================
# Config
# ======================

NUM_EPISODES = int(os.getenv("SNAKE_EPISODES", "500"))
SAVE_EVERY = 100
PLOT_EVERY = int(os.getenv("SNAKE_PLOT_EVERY", "100"))
MOVING_AVG_WINDOW = int(os.getenv("SNAKE_MOVING_AVG", "100"))

# Giống train_long_memory
LONG_MEMORY_UPDATES = int(os.getenv("SNAKE_LONG_UPDATES", "50"))
BATCH_SIZE = int(os.getenv("SNAKE_BATCH_SIZE", "512"))

MODEL_DIR = os.getenv("SNAKE_MODEL_DIR", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_ROOT = os.getenv("SNAKE_LOG_DIR", "logs")
RUN_NAME = os.getenv("SNAKE_RUN_NAME", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
LOG_DIR = os.path.join(LOG_ROOT, RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)
METRICS_PATH = os.path.join(LOG_DIR, "metrics.csv")
SUMMARY_PATH = os.path.join(LOG_DIR, "summary.txt")
RESUME_MODEL = os.getenv("SNAKE_RESUME_MODEL", "")
if not RESUME_MODEL and os.getenv("SNAKE_RESUME", "0") == "1":
    RESUME_MODEL = os.path.join(MODEL_DIR, "snake_dqn_best.pth")


def moving_average(values, window):
    if not values:
        return []
    result = []
    running_sum = 0.0
    for i, value in enumerate(values):
        running_sum += value
        if i >= window:
            running_sum -= values[i - window]
        result.append(running_sum / min(i + 1, window))
    return result


def save_training_plots(history, log_dir, window):
    if plt is None or not history["episode"]:
        return

    episodes = history["episode"]
    score_ma = moving_average(history["score"], window)
    reward_ma = moving_average(history["reward"], window)
    loss_ma = moving_average(history["loss"], window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(episodes, history["score"], alpha=0.35, label="score")
    axes[0, 0].plot(episodes, score_ma, label=f"score MA{window}")
    axes[0, 0].set_title("Episode Score")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, history["reward"], alpha=0.35, label="reward")
    axes[0, 1].plot(episodes, reward_ma, label=f"reward MA{window}")
    axes[0, 1].set_title("Episode Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, history["loss"], alpha=0.35, label="loss")
    axes[1, 0].plot(episodes, loss_ma, label=f"loss MA{window}")
    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Huber Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, history["epsilon"], label="epsilon")
    axes[1, 1].set_title("Exploration Rate")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Snake DQN Training Metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "training_metrics.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, score_ma, label=f"score MA{window}")
    ax.plot(episodes, reward_ma, label=f"reward MA{window}")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Moving Average")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "learning_curve.png"), dpi=150)
    plt.close(fig)


def write_summary(history, path, best_score):
    if not history["episode"]:
        return

    window = min(MOVING_AVG_WINDOW, len(history["episode"]))
    score_ma = moving_average(history["score"], window)
    reward_ma = moving_average(history["reward"], window)
    loss_ma = moving_average(history["loss"], window)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Snake DQN Training Summary\n")
        f.write(f"Episodes: {history['episode'][-1]}\n")
        f.write(f"Best score: {best_score}\n")
        f.write(f"Final score MA{window}: {score_ma[-1]:.3f}\n")
        f.write(f"Final reward MA{window}: {reward_ma[-1]:.3f}\n")
        f.write(f"Final loss MA{window}: {loss_ma[-1]:.6f}\n")
        f.write(f"Final epsilon: {history['epsilon'][-1]:.6f}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Long memory updates: {LONG_MEMORY_UPDATES}\n")

# ======================
# Environment
# ======================

env = SnakeEnv()

state = env.reset()
state_size = state.shape[0]
action_size = 3

# ======================
# Agent
# ======================

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=5e-4,
    gamma=0.995,
    batch_size=BATCH_SIZE,
    epsilon_decay=0.997,
    epsilon_min=0.02,
)

if RESUME_MODEL:
    if os.path.exists(RESUME_MODEL):
        agent.load(RESUME_MODEL)
        print(f"Resumed model: {RESUME_MODEL}")
    else:
        print(f"Resume model not found, starting from scratch: {RESUME_MODEL}")

# ======================
# Training
# ======================

best_score = -1
history = {
    "episode": [],
    "score": [],
    "reward": [],
    "loss": [],
    "epsilon": [],
    "train_steps": [],
}

with open(METRICS_PATH, "w", newline="", encoding="utf-8") as metrics_file:
    writer = csv.writer(metrics_file)
    writer.writerow([
        "episode",
        "score",
        "reward",
        "loss",
        "epsilon",
        "train_steps",
        "best_score",
    ])

for episode in range(1, NUM_EPISODES + 1):

    state = env.reset()

    done = False

    total_reward = 0
    total_loss = 0
    train_steps = 0

    # ======================
    # Short Memory
    # ======================

    while not done:

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

        # train_short_memory
        loss = agent.train_step()

        if loss is not None:
            total_loss += loss
            train_steps += 1

        state = next_state
        total_reward += reward

    # ======================
    # Long Memory
    # ======================

    for _ in range(LONG_MEMORY_UPDATES):

        loss = agent.train_step()

        if loss is not None:
            total_loss += loss
            train_steps += 1

    # Giảm epsilon sau mỗi game
    agent.decay_epsilon()

    avg_loss = (
        total_loss / train_steps
        if train_steps > 0
        else 0
    )

    if env.score > best_score:
        best_score = env.score
        agent.save(os.path.join(MODEL_DIR, "snake_dqn_best.pth"))

    history["episode"].append(episode)
    history["score"].append(env.score)
    history["reward"].append(total_reward)
    history["loss"].append(avg_loss)
    history["epsilon"].append(agent.epsilon)
    history["train_steps"].append(train_steps)

    with open(METRICS_PATH, "a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow([
            episode,
            env.score,
            total_reward,
            avg_loss,
            agent.epsilon,
            train_steps,
            best_score,
        ])

    print(
        f"Episode {episode:4d} | "
        f"Score {env.score:3d} | "
        f"Best {best_score:3d} | "
        f"Reward {total_reward:6.1f} | "
        f"Loss {avg_loss:.4f} | "
        f"Epsilon {agent.epsilon:.3f}"
    )

    if episode % SAVE_EVERY == 0:

        save_path = os.path.join(
            MODEL_DIR,
            f"snake_dqn_{episode}.pth"
        )

        agent.save(save_path)

        print(f"Model saved: {save_path}")

    if episode % PLOT_EVERY == 0:
        save_training_plots(history, LOG_DIR, MOVING_AVG_WINDOW)
        write_summary(history, SUMMARY_PATH, best_score)

save_training_plots(history, LOG_DIR, MOVING_AVG_WINDOW)
write_summary(history, SUMMARY_PATH, best_score)
print(f"Metrics CSV: {METRICS_PATH}")
if plt is None:
    print("matplotlib not installed; plots were skipped.")
else:
    print(f"Plots saved: {LOG_DIR}")
print("Training Finished!")
