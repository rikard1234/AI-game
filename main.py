import pygame
import neat
import time
import os
import random
clock = pygame.time.Clock()
FPS = 60
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

bg = pygame.image.load('img/Background.png').convert_alpha()
enemy_img = pygame.image.load('img/Enemy.png').convert_alpha()
player_img = pygame.image.load('img/Player.png').convert_alpha()

#player = pygame.transform.scale(player, (int(p_width * 0.2), int(p_height * 0.2)))
enemy_group = pygame.sprite.Group()
class Enemy(pygame.sprite.Sprite):
    def __init__(self, image, x, y, scale, speed):
        pygame.sprite.Sprite.__init__(self)
        p_height = image.get_height()
        p_width = image.get_width()
        self.img = pygame.transform.scale(image, (int(p_width * scale), int(p_height * scale)))
        self.rect = self.img.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed
        self.update_time = pygame.time.get_ticks()
        self.alive = 1
        self.counted = 0


    def update(self, target):
        if self.alive == 1:
            self.rect.y += self.speed
            if(self.rect.y > 600):
                self.kill()
                self.alive = 0
            screen.blit(self.img, self.rect)

class Player(pygame.sprite.Sprite):
    def __init__(self, image, x, y, scale, speed):
        pygame.sprite.Sprite.__init__(self)
        p_height = image.get_height()
        p_width = image.get_width()
        self.img = pygame.transform.scale(image, (int(p_width * scale), int(p_height * scale)))
        self.rect = self.img.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = speed
        self.alive = 1
        self.moving = True

    def move(self):
        if self.alive == 1:
            key = pygame.key.get_pressed()
            if (key[pygame.K_a]):
                self.rect.x -= self.speed;
            if (key[pygame.K_d]):
                self.rect.x += self.speed;
    def moveleft(self):
        if self.alive == 1 and self.rect.x > 0:
            self.rect.x -= self.speed

    def moveright(self):
        if self.alive == 1 and self.rect.x < 250:
            self.rect.x += self.speed

    def moverandom(self):
        if(self.moving == True):
            x = random.randint(0, 1)
            if (x == 1) : self.moveleft()
            if (x == 0) : self.moveright()
    def draw(self):
        if(self.alive == 1):
            screen.blit(self.img, self.rect)

    def update(self, enemy_group):
        if(self.alive == 1):
            if pygame.sprite.spritecollide(self, enemy_group, False):
                self.alive = 0

    def stop(self):
        self.moving = False
player = Player(player_img, 400, 500, 0.2, 5)

def main(genomes, config):
    gametime = pygame.time.get_ticks()
    update_time = pygame.time.get_ticks()
    counter_right = 0
    counter_left = 0
    spawn_timer = 800
    bonus_timer = 5000
    nets = []
    ge = []
    players = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        x_random = random.randint(100, 200)
        players.append(Player(player_img, x_random, 500, 0.2, 10))
        g.fitness = 0
        ge.append(g)
    run = True
    while run:
        clock.tick(FPS)
        screen.blit(bg, (0, 0))
        if len(players) <= 0:
            run = False
            break
        if pygame.time.get_ticks() - gametime > bonus_timer:
            gametime =  pygame.time.get_ticks()
            print("bonus!")
            for x, player in enumerate(players):
                ge[x].fitness += 0.01

        for x, player in enumerate(players):
            player.moverandom()
            for y, enemy in enumerate(enemy_group):
                output = nets[x].activate(
                (player.rect.x, abs(player.rect.x - enemy.rect.x), abs(player.rect.x - 20 - enemy.rect.x)))
                output2 = nets[x].activate(
                (player.rect.x, abs(player.rect.x - enemy.rect.x), abs(player.rect.x + 20 - enemy.rect.x)))
                if output[0] > 0.8:
                    player.moveleft()
                    player.moving = False
                    player.update(enemy_group)
                    if (enemy.rect.y > 500 and enemy.rect.y < 600 and player.alive):
                        #print("kwa")
                        ge[x].fitness += 2
                if output2[0] > 0.8:
                    player.moveright()
                    player.update(enemy_group)
                    if (enemy.rect.y > 500 and enemy.rect.y < 600 and player.alive):
                        #print("kwa")
                        ge[x].fitness += 2


        for x, player in enumerate(players):
            player.draw()

        for x, player in enumerate(players):
            player.moving = True
            player.update(enemy_group)
            if player.alive == 0:
                ge[x].fitness -= 2
                players.pop(x)
                nets.pop(x)
                ge.pop(x)

        if pygame.time.get_ticks() - update_time > spawn_timer:
            update_time = pygame.time.get_ticks()
            x_random = random.randint(0, 300)
            enemy = Enemy(enemy_img, x_random, 0, 0.8, 15)
            enemy_group.add(enemy)

        enemy_group.update(player)
        counter_left = 0
        counter_right = 0






        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # update display window
        pygame.display.update()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main, 200)


if __name__== "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "../MojaGraAi/config-feedforward.txt")
    run(config_path)





#if __name__ == '__main__':
 #   print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
