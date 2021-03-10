import pygame
import numpy as np
import random

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)

class Grid:
    def __init__(self, screen, size, linewidth, borders):
        self.size = size  # Grid size
        self.screen = screen  # Pygame screen on which the grid will be displayed
        self.width, self.height = screen.get_size()  # Size of the screen
        self.linewidth = linewidth  # Width of the lines
        self.borders = borders  # Borders of the cells
        self.dict = np.zeros((size,size))  # Cells dictionary
        self.shapesCells = None  # Shapes positions on the grid
        self.shapesColors = None  # Shapes colors
        self.shapesPos = None  # Shapes positions on the cells
        self.shapesSize = None  # Shapes sizes

    def update(self, shapes=None):
        self.dict = np.zeros((self.size,self.size))  # Initialize empty grid
        if shapes:  # Add shapes if argument given
            self.shapesCells = []
            for i in range(len(shapes)):
                row, col = random.randrange(self.size), random.randrange(self.size)
                while (row, col) in self.shapesCells:
                    row, col = random.randrange(self.size), random.randrange(self.size)
                self.shapesCells.append((row, col))
                self.dict[row][col] = shapes[i]
            self.genShapes(len(shapes))

    # Generates the shapes characteristics
    def genShapes(self, nbShapes):
        cellDim = (self.width / self.size) - self.borders * 2
        self.shapesColors = []
        self.shapesPos = []
        self.shapesSize = []

        for i in range(nbShapes):
            self.shapesColors.append(tuple(np.random.choice(range(256), size=3)))
            self.shapesSize.append(random.randint(5, cellDim-1))
            cell = self.shapesCells[i]

            # Shape is a square
            if self.dict[cell[0]][cell[1]] == 1:
                x, y = random.randrange(cellDim - self.shapesSize[i]), random.randrange(
                    cellDim - self.shapesSize[i])
                self.shapesPos.append((x, y))

            # Shape is a circle
            elif self.dict[cell[0]][cell[1]] == 2:
                xCoeff, yCoeff = [-1, 1][random.randrange(2)], [-1, 1][random.randrange(2)]
                x, y = xCoeff * random.randrange((cellDim - self.shapesSize[i]) // 2 + 1), yCoeff * random.randrange(
                    (cellDim - self.shapesSize[i]) // 2 + 1)
                self.shapesPos.append((x, y))

            # Shape is a triangle
            elif self.dict[cell[0]][cell[1]] == 3:
                x, y = random.randrange(cellDim - self.shapesSize[i]), random.randrange(
                    cellDim - self.shapesSize[i])
                self.shapesPos.append((x, y))

    # Draw shapes
    def drawShapes(self):
        cellDimX = cellDimY = (self.width / self.size) - self.borders * 2

        for i in range(len(self.shapesCells)):
            cell = self.shapesCells[i]

            # Shape is a square
            if self.dict[cell[0]][cell[1]] == 1:
                posY = cellDimX * cell[0] + self.linewidth / 2 + self.borders + 2 * cell[0] * self.borders + \
                       self.shapesPos[i][0]
                posX = cellDimY * cell[1] + self.linewidth / 2 + self.borders + 2 * cell[1] * self.borders + \
                       self.shapesPos[i][1]
                pygame.draw.rect(self.screen, self.shapesColors[i],
                                 (posX, posY, self.shapesSize[i], self.shapesSize[i]))
                pygame.draw.rect(self.screen, BLACK, (posX, posY, self.shapesSize[i], self.shapesSize[i]),
                                 1)

            # Shape is a circle
            elif self.dict[cell[0]][cell[1]] == 2:
                posY = cellDimX * cell[0] + cellDimX / 2 + self.linewidth / 2 + self.borders + 2 * cell[0] * self.borders + \
                       self.shapesPos[i][0]
                posX = cellDimY * cell[1] + cellDimY / 2 + self.linewidth / 2 + self.borders + 2 * cell[1] * self.borders + \
                       self.shapesPos[i][1]
                pygame.draw.circle(self.screen, self.shapesColors[i], (posX, posY), self.shapesSize[i] / 2)
                pygame.draw.circle(self.screen, BLACK, (posX, posY), self.shapesSize[i] / 2, 1)

            # Shape is a triangle
            elif self.dict[cell[0]][cell[1]] == 3:
                posY = cellDimX * (cell[0] + 1) + 2 * cell[0] * self.borders + self.borders - self.shapesPos[i][0]
                posX = cellDimY * cell[1] + 2 * cell[1] * self.borders + self.borders + self.shapesPos[i][1]
                pygame.draw.polygon(self.screen, self.shapesColors[i],
                                    [(posX, posY), (posX + self.shapesSize[i] / 2, posY - self.shapesSize[i]),
                                     (posX + self.shapesSize[i], posY)])
                pygame.draw.polygon(self.screen, BLACK,
                                    [(posX, posY), (posX + self.shapesSize[i] / 2, posY - self.shapesSize[i]),
                                     (posX + self.shapesSize[i], posY)], 1)

    # Draw grid borders and cells divisions
    def drawGrid(self):
        self.screen.fill(GREY)
        # Get cell size
        cellSize = self.width / self.size

        for x in range(self.size):
            for y in range(self.size):
                pygame.draw.rect(self.screen, WHITE, (
                    cellSize * x + self.linewidth, cellSize * y + self.linewidth, cellSize - 2 * self.linewidth,
                    cellSize - 2 * self.linewidth),
                                 border_radius=3)

        # Draw shapes
        if self.shapesCells:
            self.drawShapes()

    # Generate random shapes numbers (1=square, 2=circle, 3=triangle)
    @staticmethod
    def genShapesTypes(oldShape=None):
        shapes = []

        # Add previously displayed shape if argument given
        if oldShape:
            shapes.append(oldShape)
            newShape = random.randrange(3) + 1
            while newShape == oldShape:
                newShape = random.randrange(3) + 1
            shapes.append(newShape)
        else:
            newShape = random.randrange(3) + 1
            shapes.append(newShape)

        return shapes


