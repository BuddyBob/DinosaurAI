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
pygame.display.set_caption("Dinosaur Game")
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

score = 0

class Dino:
    def __init__(self, x_pos=80, y_pos=310, jump_vel=8):
        self.X_POS = x_pos
        self.Y_POS = y_pos
        self.jump_vel = jump_vel
        self.gravity = jump_vel
        self.run_img = RUNNING
        self.jump_img = JUMPING
        self.dino_run = True
        self.dino_jump = False
        self.step_index = 0
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, jump):
        if jump and not self.dino_jump:
            self.dino_jump = True
            self.dino_run = False
        if self.dino_jump:
            self.image = self.jump_img
            self.dino_rect.y -= self.gravity * 4
            self.gravity -= 0.8
            if self.gravity < -self.jump_vel:
                self.dino_jump = False
                self.dino_rect.y = self.Y_POS
                self.gravity = self.jump_vel
        else:
            self.image = self.run_img[self.step_index // 5]
            self.step_index = (self.step_index + 1) % 10
            self.dino_rect = self.image.get_rect()
            self.dino_rect.x = self.X_POS
            self.dino_rect.y = self.Y_POS

    def draw(self, screen):
        screen.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

class Cactus:
    def __init__(self, image):
        self.image = image
        self.cactus_rect = self.image.get_rect()
        self.cactus_rect.x = screen_width
        self.cactus_rect.y = 325

    def update(self, speed):
        self.cactus_rect.x -= speed
        return self.cactus_rect.x < -self.image.get_width()

    def draw(self, screen):
        screen.blit(self.image, (self.cactus_rect.x, self.cactus_rect.y))

def draw_background():
    screen.fill((255, 255, 255))
    for i in range(9):
        screen.blit(TRACK, (TRACK.get_width() * i, 340))

def evaluate_genomes(genomes, config):
    global score, p
    score = 0
    dinos, nets, ge = [], [], []
    cacti = [Cactus(random.choice(CACTUS_IMAGES))]
    spawn_distance = random.randint(300, 500)
    speed = 15

    for gid, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        dinos.append(Dino())
        genome.fitness = 0
        ge.append(genome)

    while dinos:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # update & remove cacti
        for cactus in list(cacti):
            if cactus.update(speed):
                cacti.remove(cactus)
                score += 2
                for genome in ge:
                    genome.fitness += 1
                if score % 100 == 0:
                    speed += 1
        
        # each frame increment fitness
        for genome in ge:
            genome.fitness += 0.05


        # spawn new cactus when last one passes threshold
        if cacti:
            last_x = cacti[-1].cactus_rect.x
            if last_x < screen_width - spawn_distance:
                cacti.append(Cactus(random.choice(CACTUS_IMAGES)))
                spawn_distance = random.randint(300, 500)

        # collision check
        for i, dino in enumerate(dinos):
            for cactus in cacti:
                if dino.dino_rect.colliderect(cactus.cactus_rect):
                    ge[i].fitness -= .5
                    dinos.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    break
            else:
                continue
            break

        # dino decisions
        for i, dino in enumerate(dinos):
            inputs = (
                dino.dino_rect.y,
                min(abs(dino.dino_rect.x - c.cactus_rect.x) for c in cacti),
                min(abs(dino.dino_rect.y - c.cactus_rect.y) for c in cacti)
            )
            jump = nets[i].activate(inputs)[0] > 0.5
            dino.update(jump)

        draw_background()
        font = pygame.font.Font('freesansbold.ttf', 20)
        screen.blit(font.render(f"Generation: {p.generation+1}", True, (0,0,0)), (10, 10))
        screen.blit(font.render(f"Score: {score}", True, (0,0,0)), (10, 70))
        highest = max((g.fitness for g in ge), default=0)
        avg = sum(g.fitness for g in ge)/len(ge) if ge else 0
        screen.blit(font.render(f"Highest Fitness: {highest}", True, (0,0,0)), (10, 130))
        screen.blit(font.render(f"Average Fitness: {avg:.2f}", True, (0,0,0)), (10, 160))

        for cactus in cacti:
            cactus.draw(screen)
        for dino in dinos:
            dino.draw(screen)

        pygame.display.update()
        clock.tick(30)

def run_neat(config_path):
    global p
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(evaluate_genomes, 50)
    print(f"\nBest genome:\n{winner}")

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    with open("best_genome.pkl", "rb") as f:
        best_genome = pickle.load(f)
    print(f"Best genome loaded:\n{best_genome}")

if __name__ == "__main__":
    run_neat('config.txt')
    pygame.quit()
