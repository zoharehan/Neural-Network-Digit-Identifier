import sys
import pygame
from pygame.locals import *
import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

pygame.init()


class Screen:
    def __init__(self, s_width, s_height):
        self.width = s_width
        self.height = s_height
        self.screen = pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption("Draw Your Number!")
        self.screen.fill((0,0,0)) #black colored screen

    def draw(self, br, a, b):
        self.screen.blit(br, (a,b))

    def save_screen(self):
        pygame.image.save(self.screen, "TEST.png")


class Brush:
    def __init__(self, b_width, b_height):
        self.width = b_width
        self.height = b_height
        brush = pygame.image.load("brush.png") #pngs have transparent layers
        self.brush = pygame.transform.scale(brush, (self.width,self.height)) #resize brush

    def draw(self):
        x, y = pygame.mouse.get_pos()
        screen.draw(self.brush, x - 25, y - 25)  # draw brush object to screen, - half of brush size
        pygame.display.update()


# creating screen and brush objects
screen = Screen(560,560)
brush = Brush(50,50)


# THE NEURAL NETWORK TRAINING
# load data
data = keras.datasets.mnist
# train-test split
(train_images, train_labels), (test_images, test_labels) = data.load_data()


# initialising the display and the clock
pygame.display.update()
clock = pygame.time.Clock()  # to prevent program from running more than 60f/sec


# The MAIN GAME LOOP
play = False

while True:
    clock.tick(60)  # no more than 60 frames per sec
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            play = True
        elif event.type == MOUSEBUTTONUP:
            play = False
        elif event.type == KEYDOWN:
            #pygame.image.save(screen, "TEST.png")
            screen.save_screen()
            model = keras.models.load_model("project.h5")

            image = cv.imread("TEST.png", 0)
            image = image / 255.0
            # print(image.shape)
            image = cv.resize(image, (28, 28), interpolation=cv.INTER_LINEAR_EXACT) #reconsider
            # print(image)
            image.shape = (1, 28, 28)
            predict = model.predict([image])
            # print(predict[0])
            number = np.argmax(predict[0])
            plt.grid(False)
            plt.imshow(image[0], cmap=plt.cm.binary)
            plt.title("Prediction: " + str(number))
            plt.show()
            print("i predict this number is ", number)
            sys.exit()

        if play:
            brush.draw()

