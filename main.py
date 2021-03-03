import sys
import pygame
import keras
import tensorflow
from keras.layers import Conv2D, MaxPool2D, Flatten, TimeDistributed, LSTM, Dense
import grid
import numpy as np
from pygame.locals import KEYDOWN, K_q

# Constants
SIZE = (100, 100)
CHANNELS = 3
DELAYFRAMES = 1
EPOCHS = 10


def build_conv(shape=(SIZE[0], SIZE[1], 3)):
    model = keras.Sequential()
    model.add(Conv2D(36, (7, 7), input_shape=shape, padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(72, (6, 6), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(108, (5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())

    print(model.summary())
    return model


def build_model(shape=(DELAYFRAMES+3, SIZE[0], SIZE[1], 3), nbout=16):
    # Create the conv-pool layers
    conv = build_conv(shape[1:])
    # Create the model
    model = keras.Sequential()
    # Add the conv-pool layers
    model.add(TimeDistributed(conv, input_shape=shape))
    # Add LSTM layer
    model.add(LSTM(16, activation='softmax'))
    print(model.summary())
    return model


def main():
    # genTrainData(1000)
    train_inputs = np.load('train_inputs.npy')
    train_labels = np.load('train_labels.npy')
    train_labels = tensorflow.keras.utils.to_categorical(train_labels, 16)

    #print(train_inputs.shape)

    model = build_model()
    opt = keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    model.fit(
        train_inputs[:200],
        train_labels[:200],
        validation_split=0.25,
        verbose=1,
        epochs=EPOCHS,
        batch_size=10
    )
    model.save('model2.keras')
    # example()


# Displays a visual example of a classification with trained model
# Frame rate is voluntarily slowered for better visualization
def example():
    # Initialization of the pygame
    pygame.init()

    # Create the screen
    screen = pygame.display.set_mode(SIZE)

    # Create the clock
    clock = pygame.time.Clock()

    # Set title and icon
    # Icon made by Freepik (https://www.freepik.com) from Flaticon (https://www.flaticon.com)
    pygame.display.set_caption("Grid Shapes")
    icon = pygame.image.load('blocks.png')
    pygame.display.set_icon(icon)

    # Initialize the grid
    minigrid = grid.Grid(screen, 4, 1, 2)

    for i in range(1):
        # Empty grid for first frame
        checkEvents()
        minigrid.drawGrid()
        pygame.display.update()
        clock.tick(1)

        # One shape for second frame
        shape = grid.Grid.genShapesTypes()
        minigrid.update(shape)
        checkEvents()
        minigrid.drawGrid()
        pygame.display.update()
        img = pygame.surfarray.array3d(screen).flatten()
        clock.tick(1)

        # Empty grid again for a random number of frames
        minigrid.update()
        for j in range(DELAYFRAMES):
            checkEvents()
            minigrid.drawGrid()
            pygame.display.update()
            clock.tick(1)

        # New shapes for last frame
        minigrid.update(grid.Grid.genShapesTypes(shape[0]))
        checkEvents()
        minigrid.drawGrid()
        pygame.display.update()
        clock.tick(1)
        print(minigrid.shapesCells[0][0] * minigrid.size + minigrid.shapesCells[0][1])

        # train_data = pd.DataFrame(pd.read_csv('train.csv')).to_numpy()
        # img = train_data[0,0][0]
        # print(img)
        # pygame.surfarray.blit_array(screen,img)
        # pygame.display.update()
        # clock.tick(1)


# Generate training set of given size and save it as a .csv file for later use
# The set consists of sequences of images, followed by the correct cell selection (label)
def genTrainData(samples):
    # Initialization of the pygame
    pygame.init()

    # Create the screen
    screen = pygame.display.set_mode(SIZE)

    # Create the clock
    clock = pygame.time.Clock()

    # Set title and icon
    # Icon made by Freepik (https://www.freepik.com) from Flaticon (https://www.flaticon.com)
    pygame.display.set_caption("Grid Shapes")
    icon = pygame.image.load('blocks.png')
    pygame.display.set_icon(icon)

    # Initialize the grid
    minigrid = grid.Grid(screen, 4, 1, 2)

    train_inputs = []
    train_labels = []
    for i in range(samples):
        sequence = []
        # Empty grid for first frame
        minigrid.drawGrid()
        pygame.display.update()
        sequence.append(np.array(pygame.surfarray.array3d(screen)))

        # One shape for second frame
        shape = grid.Grid.genShapesTypes()
        minigrid.update(shape)
        minigrid.drawGrid()
        pygame.display.update()
        sequence.append(pygame.surfarray.array3d(screen))

        # Empty grid again for a random number of frames
        minigrid.update()
        for j in range(DELAYFRAMES):
            minigrid.drawGrid()
            pygame.display.update()
            sequence.append(pygame.surfarray.array3d(screen))

        # New shapes for last frame
        minigrid.update(grid.Grid.genShapesTypes(shape[0]))
        minigrid.drawGrid()
        pygame.display.update()
        sequence.append(pygame.surfarray.array3d(screen))

        # Save sequence of images and correct cell to train data
        train_inputs.append(sequence)
        train_labels.append(minigrid.shapesCells[0][0]*minigrid.size+minigrid.shapesCells[0][1])

    np.save('train_inputs.npy', train_inputs)
    np.save('train_labels.npy', train_labels)



# Check if any events are detected in the game (only useful for visualization)
def checkEvents():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == KEYDOWN and event.key == K_q:
            pygame.quit()
            sys.exit()

if __name__ == '__main__':
    main()