import random

from environment.snake_env import SnakeEnv
from renderer.pygame_renderer import PygameRenderer


env = SnakeEnv()
renderer = PygameRenderer(env)

done = False
running = True

while running and not done:

    action = random.randint(0, 2)

    state, reward, done, _ = env.step(action)

    print(f"Danger Straight : {state[0]}")
    print(f"Danger Right    : {state[1]}")
    print(f"Danger Left     : {state[2]}")

    print(f"Moving Left     : {state[3]}")
    print(f"Moving Right    : {state[4]}")
    print(f"Moving Up       : {state[5]}")
    print(f"Moving Down     : {state[6]}")

    print(f"Food Left       : {state[7]}")
    print(f"Food Right      : {state[8]}")
    print(f"Food Up         : {state[9]}")
    print(f"Food Down       : {state[10]}")
    print(f"State Size      : {len(state)}")
    print("-" * 30)

    running = renderer.render()

renderer.close()
