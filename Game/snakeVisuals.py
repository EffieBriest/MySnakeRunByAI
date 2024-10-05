import pygame
import Game.snake as snake
class snakeVisuals:

    def __init__(self):
        pygame.init()         
        self.screen = pygame.display.set_mode((1200, 800)) 
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("bahnschrift", int(18 * 2))


    def drawFood(self, food):
        pygame.draw.rect(self.screen, (255, 0, 0), [food.x, food.y, 10, 10])


    def drawSnake(self, snake: snake.Snake):
        for i in range(snake.length):
            if i == 0:
                pygame.draw.rect(self.screen, (50, 150, 255), (snake.x[0], snake.y[0], 10, 10))
            else:
                pygame.draw.rect(self.screen, (255, 255, 255), (snake.x[i], snake.y[i], 10, 10))


    def updateScreen(self, food, snake, running, score, episode):
        self.screen.fill((0,0,0))
        self.drawFood(food)
        self.drawSnake(snake)
        self.print_episode(episode)
        self.print_score(score)
        pygame.display.flip()
        if not running:
            self.gameOverScreen()
            pygame.display.update()


    def print_score(self, score):
        value = self.font.render(f"Score: {score}", True, (255,255,255))
        self.screen.blit(value, [500 * 2, 10])


    def print_episode(self, episode):
            value = self.font.render(f"Episode: {episode}", True, (255,255,255))
            self.screen.blit(value, [10, 10])


    def gameOverScreen(self):
        mesg = self.font.render("Game over!", True, (255,255,255))
        self.screen.blit(mesg, [2 * int(600 * 2) / 5, 2 * int(400 * 2) / 5 + int(30 * 2)])