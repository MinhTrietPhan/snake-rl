# Snake RL - Double DQN

Dự án huấn luyện AI chơi game Snake bằng Deep Reinforcement Learning. Mục tiêu là giúp rắn ăn táo, tránh chết, giảm hành vi tự nhốt mình và đạt score cao nhất có thể.

## Công nghệ chính

- Python: xây dựng môi trường Snake, train và evaluate.
- PyTorch: xây dựng và huấn luyện neural network.
- Deep Q-Network: học chính sách chọn hành động dựa trên Q-value.
- Double DQN: giảm lỗi overestimate Q-value bằng cách dùng online network để chọn hành động và target network để đánh giá.
- Replay Buffer: lưu kinh nghiệm cũ và sample batch ngẫu nhiên để train ổn định hơn.
- Target Network: mạng phụ giúp ổn định giá trị mục tiêu khi cập nhật Q-value.
- Epsilon-Greedy: cân bằng giữa khám phá hành động mới và khai thác hành động model cho là tốt.
- Reward Shaping: thiết kế điểm thưởng/phạt để rắn vừa ăn táo vừa tránh tự nhốt.
- Action Masking: loại bỏ hành động nguy hiểm hoặc dẫn vào vùng quá hẹp nếu còn lựa chọn an toàn.
- Flood Fill: tính vùng trống có thể di chuyển để đánh giá độ an toàn của hành động.
- Matplotlib: lưu biểu đồ score, reward, loss và epsilon trong quá trình train.
- Pygame: hiển thị game khi evaluate/demo.

## Cấu trúc thư mục

```text
agents/
  dqn_agent.py        # Agent DQN, Double DQN, action masking
  replay_buffer.py    # Replay memory
environment/
  snake_env.py        # Môi trường Snake, state, reward, collision, flood fill
models/
  qnet.py             # Neural network QNet
renderer/
  pygame_renderer.py  # Hiển thị game bằng pygame
train.py              # Train model
evaluate.py           # Test/evaluate checkpoint
checkpoints/          # Nơi lưu model
logs/                 # Nơi lưu metrics, summary và biểu đồ train
```

## Luồng hoạt động

1. `SnakeEnv` tạo trạng thái hiện tại của game.
2. `DQNAgent` chọn một trong ba hành động: đi thẳng, rẽ phải, rẽ trái.
3. Môi trường thực hiện hành động, trả về state mới, reward và trạng thái kết thúc.
4. Kinh nghiệm `(state, action, reward, next_state, done)` được lưu vào replay buffer.
5. Agent sample batch từ replay buffer để train QNet.
6. Target network được cập nhật định kỳ để ổn định quá trình học.
7. Model có score tốt nhất được lưu vào checkpoint.

## State

Môi trường trả về vector state mô tả tình huống hiện tại của rắn. Các nhóm thông tin chính:

- Nguy hiểm phía trước, bên phải, bên trái.
- Hướng hiện tại của rắn.
- Vị trí tương đối của táo so với đầu rắn.
- Vị trí tường.
- Vị trí thân/đuôi gần đầu rắn.
- Vùng trống có thể đi được sau mỗi hành động.
- Vị trí đầu rắn trên map.
- Tỉ lệ độ dài rắn so với màn hình.
- Khoảng cách tới táo.

## Reward

Reward được thiết kế để tránh việc rắn chỉ tham lam đi gần táo rồi tự nhốt:

- Chết: phạt nặng.
- Ăn táo: thưởng lớn.
- Mỗi bước đi: phạt nhẹ để tránh đi vòng vô hạn.
- Còn đủ vùng sống: thưởng nhẹ.
- Đi vào vùng quá hẹp: phạt.
- Còn đường tới đuôi: thưởng nhẹ.
- Mất đường tới đuôi: phạt.
- Tiến gần táo chỉ được thưởng nếu hành động đó vẫn giữ vùng sống an toàn.

## Cài đặt

Cài các thư viện cần thiết:

```powershell
pip install torch numpy matplotlib pygame
```

Nếu chỉ train không cần hiển thị game, `pygame` có thể thiếu. Khi đó `evaluate.py` vẫn có thể chạy ở chế độ không render.

## Train

Train mặc định:

```powershell
python train.py
```

Mặc định hiện tại:

- `SNAKE_EPISODES=500`
- `SNAKE_BATCH_SIZE=512`
- `SNAKE_LONG_UPDATES=50`
- `SNAKE_EXACT_SPACE=1`

Train 500 episode:

```powershell
$env:SNAKE_EPISODES="500"
$env:SNAKE_RUN_NAME="reward_safety_500eps"
python train.py
```

Train lâu hơn:

```powershell
$env:SNAKE_EPISODES="3000"
$env:SNAKE_RUN_NAME="reward_safety_3000eps"
python train.py
```

Train tiếp từ best checkpoint:

```powershell
$env:SNAKE_RESUME="1"
$env:SNAKE_EPISODES="1000"
python train.py
```

Train tiếp từ checkpoint cụ thể:

```powershell
$env:SNAKE_RESUME_MODEL="checkpoints/snake_dqn_1000.pth"
$env:SNAKE_EPISODES="1000"
python train.py
```

## Checkpoint

Model được lưu trong thư mục `checkpoints/`.

- `snake_dqn_best.pth`: model có score cao nhất trong lần train.
- `snake_dqn_100.pth`, `snake_dqn_200.pth`, ...: model được lưu định kỳ mỗi 100 episode.

`snake_dqn_best.pth` sẽ bị ghi đè khi có episode đạt score cao hơn best score hiện tại của run.

## Evaluate

Evaluate model mặc định:

```powershell
python evaluate.py
```

Evaluate không render:

```powershell
$env:SNAKE_EVAL_RENDER="0"
python evaluate.py
```

Evaluate checkpoint cụ thể:

```powershell
$env:SNAKE_EVAL_MODEL="checkpoints/snake_dqn_best.pth"
python evaluate.py
```

Giới hạn số step khi evaluate:

```powershell
$env:SNAKE_EVAL_MAX_STEPS="20000"
python evaluate.py
```

## Logs và biểu đồ

Mỗi lần train tạo một thư mục trong `logs/`:

```text
logs/run_YYYYMMDD_HHMMSS/
```

Các file được tạo:

- `metrics.csv`: log từng episode.
- `summary.txt`: tóm tắt kết quả cuối run.
- `training_metrics.png`: biểu đồ score, reward, loss, epsilon.
- `learning_curve.png`: biểu đồ moving average của score và reward.

Các cột chính trong `metrics.csv`:

- `episode`
- `score`
- `reward`
- `loss`
- `epsilon`
- `train_steps`
- `best_score`

## Biến môi trường quan trọng

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `SNAKE_EPISODES` | `500` | Số episode train trong một lần chạy |
| `SNAKE_BATCH_SIZE` | `512` | Batch size khi train |
| `SNAKE_LONG_UPDATES` | `50` | Số lần train thêm sau mỗi episode |
| `SNAKE_EXACT_SPACE` | `1` | `1` dùng flood fill chính xác, `0` dùng bản nhanh hơn |
| `SNAKE_RUN_NAME` | timestamp | Tên thư mục log |
| `SNAKE_LOG_DIR` | `logs` | Thư mục lưu log |
| `SNAKE_MODEL_DIR` | `checkpoints` | Thư mục lưu checkpoint |
| `SNAKE_RESUME` | `0` | `1` để resume từ `snake_dqn_best.pth` |
| `SNAKE_RESUME_MODEL` | rỗng | Đường dẫn checkpoint để train tiếp |
| `SNAKE_EVAL_MODEL` | `checkpoints/snake_dqn_best.pth` | Checkpoint dùng khi evaluate |
| `SNAKE_EVAL_RENDER` | `1` | `1` hiển thị pygame, `0` chạy console |
| `SNAKE_EVAL_MAX_STEPS` | `10000` | Số step tối đa khi evaluate |

## Ghi chú về epsilon

Epsilon là tỉ lệ agent chọn hành động random để khám phá.

```text
epsilon_new = epsilon_old * epsilon_decay
```

Nếu `epsilon_decay` càng gần `1.0`, agent khám phá lâu hơn nhưng train thấy chậm cải thiện hơn. Nếu decay nhỏ hơn, agent bớt random nhanh hơn nhưng dễ kẹt ở chiến thuật chưa tối ưu.

Với train ngắn 500 episode, có thể dùng decay nhanh hơn như `0.99`. Với train dài hơn, decay chậm như `0.997` thường ổn định hơn.

## Hạn chế

- Snake là bài toán có kế hoạch dài hạn, DQN vẫn có thể bị kẹt ở hành vi tham táo.
- Map 20x20 khó hơn nhiều so với map nhỏ vì rắn phải tránh tự nhốt khi thân dài.
- Flood fill giúp score tốt hơn nhưng làm train chậm hơn vì chạy nhiều logic trên CPU.
- Score cao nhất trong một episode không phản ánh toàn bộ chất lượng model; nên xem thêm moving average trong log.

## Gợi ý thuyết trình

Dự án sử dụng Deep Reinforcement Learning để huấn luyện agent chơi Snake. Agent quan sát trạng thái môi trường, dùng mạng QNet để dự đoán giá trị của từng hành động, sau đó học từ kinh nghiệm lưu trong replay buffer. Dự án cải thiện DQN bằng Double DQN, action masking, flood fill và reward shaping để rắn không chỉ đi tới táo mà còn giữ không gian sống, tránh tự nhốt và đạt score cao hơn.
