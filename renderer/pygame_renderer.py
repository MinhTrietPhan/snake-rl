import pygame


class PygameRenderer:

    def __init__(self, env, fps=10):

        self.env = env
        self.fps = fps
        pygame.init()

        self.font = pygame.font.SysFont("Arial", 24)

        self.window = pygame.display.set_mode(
            (
                env.width * env.block_size,
                env.height * env.block_size,
            )
        )

        pygame.display.set_caption("Snake RL")

        self.clock = pygame.time.Clock()

        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.SNAKE_COLOR = (0, 255, 0)
        self.FOOD_COLOR = (255, 0, 0)
    def render(self):

        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.window.fill(self.BG_COLOR)

        # Draw food
        if self.env.food is not None:
            pygame.draw.rect(
                self.window,
                self.FOOD_COLOR,
                (
                    self.env.food.x * self.env.block_size,
                    self.env.food.y * self.env.block_size,
                    self.env.block_size,
                    self.env.block_size,
                ),
            )

        # Draw head
        head = self.env.snake[0]

        pygame.draw.rect(
            self.window,
            (0, 180, 0),      # Màu đầu
            (
                head.x * self.env.block_size,
                head.y * self.env.block_size,
                self.env.block_size,
                self.env.block_size,
            ),
        )

        # Draw body
        for segment in self.env.snake[1:]:

            pygame.draw.rect(
                self.window,
                (0, 255, 0),  # Màu thân
                (
                    segment.x * self.env.block_size,
                    segment.y * self.env.block_size,
                    self.env.block_size,
                    self.env.block_size,
                ),
            )
        # Count score
        score_text = self.font.render(
            f"Score: {self.env.score}",
            True,
            (255, 255, 255)
        )

        self.window.blit(score_text, (10, 10))
        pygame.display.flip()

        self.clock.tick(self.fps)

        return True
    def close(self):

        pygame.quit()
