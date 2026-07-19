# AI_CONTEXT.md

## Project
- Project name: `snake-rl-old`
- Objective: Improve a Snake reinforcement learning project so the snake learns to eat apples, avoid dying, and ideally survive until filling the screen.
- Scope:
  - Python Snake environment.
  - DQN-based agent training and evaluation.
  - Focus on improving RL behavior, state representation, action selection, reward shaping, and training stability.

## Current Status
- Completed:
  - Improved environment state from 19 features to 28 features.
  - Fixed a critical QNet constructor usage bug in the DQN agent.
  - Added action masking to avoid immediately fatal actions.
  - Changed training target update logic to Double DQN.
  - Improved food spawning and full-board handling.
  - Adjusted reward shaping and timeout.
  - Updated best-checkpoint saving and evaluation default path.
  - Updated `models/qnet.py` default `input_size` from 19 to 28 for consistency.
- In Progress:
  - Long training has not yet been completed in the workspace.
  - Actual performance after many episodes is still unconfirmed.
- Next Steps:
  - Train from scratch with the updated state/model logic.
  - Evaluate the saved best checkpoint.
  - Tune map size, reward values, epsilon decay, and model parameters if performance is poor.

## Important Background
- The project originally used a DQN agent for Snake.
- The previous state representation had 19 features.
- The current environment returns 28 features.
- Old checkpoints created before the state/model changes should be treated as incompatible.
- A previous important bug was caused by calling `QNet(state_size, action_size)` while `QNet` signature is `(input_size=..., hidden_size=..., output_size=...)`.
  - This accidentally passed `action_size=3` as `hidden_size`, creating a very small network.
  - The fixed call is `QNet(state_size, output_size=action_size)`.
- `input_size=28` in `models/qnet.py` is now only a consistent default. Runtime training still passes `state_size` explicitly from the agent.

## Technical Decisions
- Decision: Use 28-dimensional state representation.
  - Reason: The agent needs more global safety information, not only local danger and food direction.
- Decision: Add reachable free-space ratios for each action.
  - Reason: Helps the snake avoid moving into areas that trap it.
- Decision: Use action masking during action selection.
  - Reason: Prevents the agent from choosing actions that are immediately fatal.
- Decision: Use Double DQN in `train_step()`.
  - Reason: Reduces Q-value overestimation by using the online network to choose the next action and the target network to evaluate it.
- Decision: Keep target network.
  - Reason: Stabilizes Q-learning targets.
- Decision: Save the best model separately as `snake_dqn_best.pth`.
  - Reason: Evaluation should use the best-performing checkpoint, not only the latest periodic checkpoint.
- Decision: Train from scratch after the state change.
  - Reason: Old checkpoints expect a different input representation and may have different model dimensions.

## Architecture
- Environment:
  - `SnakeEnv` controls board state, snake movement, food spawning, collision detection, reward calculation, and state extraction.
  - `reset()` returns the initial state.
  - `step(action)` applies one action and returns `(state, reward, done, info)`.
- Agent:
  - `DQNAgent` selects actions using epsilon-greedy policy with action masking.
  - Stores transitions in replay memory.
  - Trains from sampled mini-batches.
  - Uses an online Q-network and target Q-network.
- Model:
  - `QNet` is a feed-forward neural network.
  - Input size: 28.
  - Hidden size default: 256.
  - Output size: 3 actions.
- Training:
  - `train.py` creates environment and agent.
  - Runs episodes, stores transitions, trains agent, saves periodic and best checkpoints.
- Evaluation:
  - `evaluate.py` loads the best checkpoint and runs the trained policy.

## Experiments / Results
- Smoke verification passed after changes:
  - Environment state length is 28.
  - QNet hidden layer is 256 when constructed by the agent.
  - Masked Double DQN training step runs.
  - Full-board food spawning handles no-free-cell state by setting `food=None`.
- Short 3-episode training run succeeded in a temporary checkpoint directory.
  - Scores were 0 due to very short training and warmup.
  - This is not evidence of final model quality.
- Unconfirmed:
  - Performance after long training.
  - Ability to reliably fill the screen on larger boards.

## Important Files / Components
- `environment/snake_env.py`
  - Snake environment.
  - Current state extraction returns 28 features.
  - Contains collision, reward, food spawning, full-board win logic, reachable-space calculation.
- `agents/dqn_agent.py`
  - DQN agent.
  - Uses `QNet(state_size, output_size=action_size)`.
  - Contains action masking and Double DQN target calculation.
- `models/qnet.py`
  - Neural network model.
  - Default `input_size=28`, `hidden_size=256`, `output_size=3`.
- `train.py`
  - Main training script.
  - Uses environment variable `SNAKE_EPISODES`.
  - Uses environment variable `SNAKE_MODEL_DIR`.
  - Saves best checkpoint.
- `evaluate.py`
  - Evaluation script.
  - Default model path points to `checkpoints/snake_dqn_best.pth`.
- `agents/replay_buffer.py`
  - Experience replay buffer.

## Constraints
- Do not assume old checkpoints are compatible with the current model/state representation.
- Do not revert the QNet constructor fix.
- Do not change the action order without also updating danger flags and action masking.
  - Current action order: `0 = straight`, `1 = right`, `2 = left`.
  - The first three state values correspond to danger for those actions.
- Avoid broad or unrelated refactors.
- Preserve user changes in the working tree.
- Use `apply_patch` for manual file edits.

## Known Issues
- Long-run training quality is unconfirmed.
- Large boards such as 20x20 remain difficult for DQN because Snake requires long-horizon planning.
- DQN can still learn greedy food chasing and self-trapping behavior.
- `pygame` may be missing in the environment, causing renderer or some tests to fail until installed.
- Some sandboxed command runs may fail to write `__pycache__` or workspace checkpoints, but this may not affect local user execution.

## TODO
1. Train from scratch with updated code:
   - Example: set `SNAKE_EPISODES=10000` and run `python train.py`.
2. Evaluate:
   - Run `python evaluate.py` after training.
3. If training is unstable, test smaller boards first:
   - Recommended: 8x8 or 10x10 before 20x20.
4. Tune hyperparameters if needed:
   - Epsilon decay.
   - Learning rate.
   - Reward values.
   - Batch size.
   - Warmup steps.
5. Consider stronger improvements if DQN still self-traps:
   - Add Hamiltonian-cycle-inspired safety logic.
   - Add more explicit tail-reachability features.
   - Try dueling DQN or prioritized replay.
   - Add curriculum training from smaller boards to larger boards.

## Next Conversation
- Continue from the current implementation, not from the original project state.
- First verify that the local files contain the latest edits:
  - `models/qnet.py` has `input_size=28`.
  - `agents/dqn_agent.py` constructs networks with `QNet(state_size, output_size=action_size)`.
  - `environment/snake_env.py` returns a 28-feature state.
- Help run or interpret long training.
- If performance is weak, diagnose using training logs and tune reward/state/hyperparameters.
