from enum import Enum
from dataclasses import dataclass
import os
import random
import numpy as np

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class SnakeEnv:

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        block_size: int = 20,
        exact_space_features: bool | None = None,
    ):
        self.width = width
        self.height = height
        self.block_size = block_size
        if exact_space_features is None:
            exact_space_features = os.getenv("SNAKE_EXACT_SPACE", "1") == "1"
        self.exact_space_features = exact_space_features

        self.snake = []
        self.food = None
        self.direction = Direction.RIGHT

        self.score = 0
        self.done = False

        self.reset()

    def reset(self):
        """Khởi tạo một game mới."""

        center_x = self.width // 2
        center_y = self.height // 2

        self.direction = Direction.RIGHT

        self.snake = [
            Point(center_x, center_y),
            Point(center_x - 1, center_y),
            Point(center_x - 2, center_y),
        ]

        self.score = 0
        self.done = False
        self.frame_iteration = 0

        self._spawn_food()

        return self._get_state()

    def step(self, action):

        self.frame_iteration += 1

        # Khoảng cách trước khi di chuyển
        head = self.snake[0]
        old_distance = self._food_distance(head)

        self._update_direction(action)

        ate_food = self._move_snake()

        # Khoảng cách sau khi di chuyển
        head = self.snake[0]
        new_distance = self._food_distance(head)

        collided = self._check_collision()

        reward = self._calculate_reward(
            ate_food,
            collided,
            old_distance,
            new_distance,
        )

        won = len(self.snake) == self.width * self.height
        if won:
            self.done = True
            reward = 100.0
        elif self.frame_iteration > 100 * len(self.snake):
            self.done = True
            reward = -10
        else:
            self.done = collided

        return (
            self._get_state(),
            reward,
            self.done,
            {"score": self.score, "won": won},
        )


    def _spawn_food(self):
        free_cells = [Point(x, y) for y in range(self.height)
                      for x in range(self.width) if Point(x, y) not in self.snake]
        self.food = random.choice(free_cells) if free_cells else None

    def _food_distance(self, point):
        if self.food is None:
            return 0
        return abs(point.x - self.food.x) + abs(point.y - self.food.y)

    def _get_state(self):

        head = self.snake[0]

        point_left = Point(head.x - 1, head.y)
        point_right = Point(head.x + 1, head.y)
        point_up = Point(head.x, head.y - 1)
        point_down = Point(head.x, head.y + 1)

        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN

        # Danger Straight
        danger_straight = (
            (dir_right and self._is_collision(point_right)) or
            (dir_left and self._is_collision(point_left)) or
            (dir_up and self._is_collision(point_up)) or
            (dir_down and self._is_collision(point_down))
        )

        # Danger Right (same order as actions: straight, right, left)
        danger_right = (
            (dir_up and self._is_collision(point_right)) or
            (dir_down and self._is_collision(point_left)) or
            (dir_left and self._is_collision(point_up)) or
            (dir_right and self._is_collision(point_down))
        )

        # Danger Left
        danger_left = (
            (dir_up and self._is_collision(point_left)) or
            (dir_down and self._is_collision(point_right)) or
            (dir_left and self._is_collision(point_down)) or
            (dir_right and self._is_collision(point_up))
        )
        head = self.snake[0]

        wall_left = head.x == 0
        wall_right = head.x == self.width - 1
        wall_up = head.y == 0
        wall_down = head.y == self.height - 1

        tail_left = Point(head.x - 1, head.y) in self.snake[1:]
        tail_right = Point(head.x + 1, head.y) in self.snake[1:]
        tail_up = Point(head.x, head.y - 1) in self.snake[1:]
        tail_down = Point(head.x, head.y + 1) in self.snake[1:]

        max_distance = max(1, self.width + self.height - 2)
        food_x = self.food.x if self.food is not None else head.x
        food_y = self.food.y if self.food is not None else head.y
        if self.exact_space_features:
            free_space = [self._reachable_ratio(action) for action in range(3)]
        else:
            free_space = [self._local_space_ratio(action) for action in range(3)]

        state = np.array([

            danger_straight,
            danger_right,
            danger_left,

            dir_left,
            dir_right,
            dir_up,
            dir_down,

            food_x < head.x,
            food_x > head.x,
            food_y < head.y,
            food_y > head.y,

            wall_left,
            wall_right,
            wall_up,
            wall_down,

            tail_left,
            tail_right,
            tail_up,
            tail_down,

            # Global information needed to avoid enclosing the snake.
            *free_space,
            head.x / max(1, self.width - 1),
            (self.width - 1 - head.x) / max(1, self.width - 1),
            head.y / max(1, self.height - 1),
            (self.height - 1 - head.y) / max(1, self.height - 1),
            len(self.snake) / (self.width * self.height),
            self._food_distance(head) / max_distance,
        ], dtype=np.float32)

        return state

    def _direction_for_action(self, action):
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = directions.index(self.direction)
        return directions[(idx + (1 if action == 1 else -1 if action == 2 else 0)) % 4]

    def _reachable_ratio(self, action):
        """Fraction of currently free cells reachable after an action (flood fill)."""
        old_direction = self.direction
        self.direction = self._direction_for_action(action)
        start = self._next_head()
        self.direction = old_direction
        blocked = set(self.snake[:-1])  # tail moves away on a normal move
        if self._is_wall(start) or start in blocked:
            return 0.0
        seen = self._flood_fill(start, blocked)
        return len(seen) / max(1, self.width * self.height - len(blocked))

    def _flood_fill(self, start, blocked):
        seen, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in (Point(p.x + 1, p.y), Point(p.x - 1, p.y),
                      Point(p.x, p.y + 1), Point(p.x, p.y - 1)):
                if not self._is_wall(q) and q not in blocked and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return seen

    def _current_reachable_ratio(self):
        head = self.snake[0]
        blocked = set(self.snake[1:-1])
        seen = self._flood_fill(head, blocked)
        return len(seen) / max(1, self.width * self.height - len(blocked))

    def _can_reach_tail(self):
        if len(self.snake) < 2:
            return True

        head = self.snake[0]
        tail = self.snake[-1]
        blocked = set(self.snake[1:-1])
        return tail in self._flood_fill(head, blocked)

    def _local_space_ratio(self, action):
        """Cheap local openness proxy used for faster training."""
        old_direction = self.direction
        self.direction = self._direction_for_action(action)
        start = self._next_head()
        self.direction = old_direction
        blocked = set(self.snake[:-1])
        if self._is_wall(start) or start in blocked:
            return 0.0

        neighbors = (
            Point(start.x + 1, start.y),
            Point(start.x - 1, start.y),
            Point(start.x, start.y + 1),
            Point(start.x, start.y - 1),
        )
        open_cells = sum(1 for p in neighbors if not self._is_wall(p) and p not in blocked)
        return open_cells / 4.0

    def _is_wall(self, point):
        return point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height
    
    def _update_direction(self, action):
        directions = [
            Direction.UP,
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
        ]
        idx = directions.index(self.direction)
        if action == 0:
            new_dir = directions[idx]
        elif action == 1:
            new_dir = directions[(idx + 1) % 4]
        elif action == 2:
            new_dir = directions[(idx - 1) % 4]
        else:
            raise ValueError("Invalid action")
        self.direction = new_dir

    def _next_head(self):
        head = self.snake[0]
        if self.direction == Direction.UP:
            return Point(head.x, head.y - 1)
        if self.direction == Direction.DOWN:
            return Point(head.x, head.y + 1)
        if self.direction == Direction.LEFT:
            return Point(head.x - 1, head.y)
        return Point(head.x + 1, head.y)
    
    def _move_snake(self):
        new_head = self._next_head()
        self.snake.insert(0, new_head)
        ate_food = (new_head == self.food)
        if ate_food:
            self.score += 1
            self._spawn_food()
        else:
            self.snake.pop()
        return ate_food
    
    def _check_collision(self):
        head = self.snake[0]
        if head.x < 0 or head.x >= self.width:
            return True
        if head.y < 0 or head.y >= self.height:
            return True
        if head in self.snake[1:]:
            return True
        return False
    
    def _is_collision(self, point):
        if self._is_wall(point):
            return True
        # Entering the current tail is legal because it moves away this frame.
        if point in self.snake[:-1]:
            return True
        return False
    
    def _calculate_reward(
        self,
        ate_food,
        collided,
        old_distance,
        new_distance,
    ):
        if collided:
            return -30

        space_ratio = self._current_reachable_ratio()
        length_ratio = len(self.snake) / (self.width * self.height)
        can_reach_tail = self._can_reach_tail()
        min_safe_space = min(0.85, length_ratio * 1.35 + 0.04)
        reward = -0.02

        if ate_food:
            reward += 25

        if space_ratio < min_safe_space:
            reward -= 2.0 * (min_safe_space - space_ratio + 0.05)
        else:
            reward += 0.03

        if can_reach_tail:
            reward += 0.05
        else:
            reward -= 1.5

        if new_distance < old_distance and space_ratio >= min_safe_space:
            reward += 0.05

        elif new_distance > old_distance:
            reward -= 0.03

        return reward
