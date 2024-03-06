import pygame
import random

pygame.init()

# Define the Star class
class Star:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), self.size)
