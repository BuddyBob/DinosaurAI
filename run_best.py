import sys
import pygame
import random
import os
import neat
import pickle

pygame.init()
screen_width = 1200
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Dinosaur Game - Best Player")
clock = pygame.time.Clock()

RUNNING = [
    pygame.image.load(os.path.join("assets", "DinoRun1.png")),
    pygame.image.load(os.path.join("assets", "DinoRun2.png"))
]
JUMPING = pygame.image.load(os.path.join("assets", "DinoJump.png"))
CACTUS_IMAGES = [
    pygame.image.load(os.path.join("assets", "LargeCactus1.png")),
    pygame.image.load(os.path.join("assets", "LargeCactus2.png")),
    pygame.image.load(os.path.join("assets", "LargeCactus3.png"))
]
TRACK = pygame.image.load(os.path.join("assets", "Track.png"))

class Dino:
    def __init__(self, x_pos=80, y_pos=310, jump_vel=8):
        self.X_POS = x_pos
        self.Y_POS = y_pos
        self.jump_vel = jump_vel
        self.gravity = jump_vel
        self.run_img = RUNNING
        self.jump_img = JUMPING
        self.step_index = 0
        self.dino_jump = False
        self.image = self.run_img[0]
        self.rect = self.image.get_rect()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS

    def update(self, jump):
        if jump and not self.dino_jump:
            self.dino_jump = True
        if self.dino_jump:
            self.image = self.jump_img
            self.rect.y -= self.gravity * 4
            self.gravity -= 0.8
            if self.gravity < -self.jump_vel:
                self.dino_jump = False
                self.rect.y = self.Y_POS
                self.gravity = self.jump_vel
        else:
            self.image = self.run_img[self.step_index // 5]
            self.step_index = (self.step_index + 1) % 10
            self.rect = self.image.get_rect()
            self.rect.x = self.X_POS
            self.rect.y = self.Y_POS

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

class Cactus:
    def __init__(self, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = screen_width
        self.rect.y = 325

    def update(self, speed):
        self.rect.x -= speed
        return self.rect.x < -self.image.get_width()

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))

def draw_background():
    screen.fill((255,255,255))
    for i in range(9):
        screen.blit(TRACK, (TRACK.get_width() * i, 340))

def main():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.txt"
    )
    with open("best_genome.pkl", "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    dino = Dino()
    cacti = [Cactus(random.choice(CACTUS_IMAGES))]
    spawn_distance = random.randint(300, 500)
    speed = 15
    score = 0

    font = pygame.font.Font('freesansbold.ttf', 20)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for cactus in list(cacti):
            if cactus.update(speed):
                cacti.remove(cactus)
                score += 1
                if score % 100 == 0:
                    speed += 1

        if cacti:
            last_x = cacti[-1].rect.x
            if last_x < screen_width - spawn_distance:
                cacti.append(Cactus(random.choice(CACTUS_IMAGES)))
                spawn_distance = random.randint(300, 500)

        nearest = min(cacti, key=lambda c: c.rect.x - dino.rect.x)
        inputs = (
            dino.rect.y,
            abs(dino.rect.x - nearest.rect.x),
            abs(dino.rect.y - nearest.rect.y)
        )
        jump = net.activate(inputs)[0] > 0.5
        dino.update(jump)

        draw_background()
        screen.blit(font.render(f"Score: {score}", True, (0,0,0)), (10,10))
        screen.blit(font.render(f"Speed: {speed}", True, (0,0,0)), (10,40))

        for cactus in cacti:
            cactus.draw(screen)
        dino.draw(screen)

        pygame.display.update()
        clock.tick(30)

if __name__ == "__main__":
    main()
