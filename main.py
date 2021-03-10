import sys
import pygame
import keras
import tensorflow
from keras.layers import Conv2D, MaxPool2D, Flatten, TimeDistributed, LSTM, Dense, BatchNormalization
import grid
import random
import numpy as np
from pygame.locals import KEYDOWN, K_q

# Constants
SIZE = (36, 36)
CHANNELS = 3
NBOUT = 9
EPOCHS = 50
MAXDELAYFRAMES = 10

def build_model(shape=(None, SIZE[0], SIZE[1], CHANNELS), nbout=NBOUT):
    # Build cnn
    cnn = keras.Sequential()
    cnn.add(Conv2D(8, (3, 3), input_shape=shape[1:], padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPool2D())
    cnn.add(Conv2D(8, (5, 5), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPool2D())
    cnn.add(Conv2D(64, (7, 7), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPool2D())
    cnn.add(Flatten())

    # Build rnn
    rnn = keras.Sequential()
    rnn = LSTM(nbout, input_shape=shape[1:], activation='softmax')


    # Combine both
    main_input = keras.Input(shape=shape)
    model = TimeDistributed(cnn)(main_input)
    model = rnn(model)

    # Build final model
    final_model = keras.Model(inputs=main_input, outputs=model)
    print(final_model.summary())
    return final_model


def main():
    # genTrainData(3000)
    train_inputs = np.load('train_inputs.npy', allow_pickle=True)
    train_labels = np.load('train_labels.npy', allow_pickle=True)
    train_labels = tensorflow.keras.utils.to_categorical(train_labels, 9)

    #print(train_inputs.shape)

    model = build_model()
    opt = keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_inputs[:1000],
        train_labels[:1000],
        validation_split=0.1,
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
    minigrid = grid.Grid(screen, 3, 1, 1)

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
        delay_frames = random.randint(1, MAXDELAYFRAMES)
        for j in range(delay_frames):
            minigrid.drawGrid()
            pygame.display.update()
            clock.tick(1)

        # New shapes for one frame
        minigrid.update(grid.Grid.genShapesTypes(shape[0]))
        minigrid.drawGrid()
        pygame.display.update()
        clock.tick(1)

        minigrid.update()
        for j in range(MAXDELAYFRAMES - delay_frames):
            minigrid.drawGrid()
            pygame.display.update()
            clock.tick(1)
        print(minigrid.shapesCells[0][0] * minigrid.size + minigrid.shapesCells[0][1])


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
    minigrid = grid.Grid(screen, 3, 1, 1)

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
        delay_frames = random.randint(1,MAXDELAYFRAMES)
        for j in range(delay_frames):
            minigrid.drawGrid()
            pygame.display.update()
            sequence.append(pygame.surfarray.array3d(screen))

        # New shapes for one frame
        minigrid.update(grid.Grid.genShapesTypes(shape[0]))
        minigrid.drawGrid()
        pygame.display.update()
        sequence.append(pygame.surfarray.array3d(screen))

        minigrid.update()
        for j in range(MAXDELAYFRAMES-delay_frames):
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