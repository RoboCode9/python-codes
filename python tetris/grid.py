import pygame

class Grid:
    def __init__(self):
        self.num_rows = 20
        self.num_cols = 10
        self.cell_size = 30
        self.grid = [[0 for j in range(self.num_cols)] for i in range(self.num_rows)]
        self.colors = self.get_cell_colors()

    def print_grid(self):
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                print(self.grid[row][column], end = " ")
            print()

    def get_cell_colors(self):

        grey = (127.5, 127.5, 127.5)
        green = (0, 255, 0)
        red = (255, 0, 0)
        black = (0, 0, 0)
        yellow = (255, 255, 0)
        purple = (255, 0, 255)
        aqua = (0, 255, 255)
        blue = (0, 0, 255)

        return [grey, green, red, black, yellow, purple, aqua, blue]    
    
    def draw(self, screen):
        for row in range(self.num_rows):
            for column in range(self.num_cols):
                cell_value = self.grid[row][column]
                cell_rect = pygame.Rect(column*self.cell_size +1, row*self.cell_size +1, self.cell_size -1, self.cell_size -1)
                pygame.draw.rect(screen, self.colors[cell_value], cell_rect)