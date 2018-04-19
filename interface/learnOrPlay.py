from Tkinter import *
from Flashcards import Flashcards
from Playing import Game

class learnOrPlay(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        learnOrPlay = Label(self, text="Are you going to learn or play?")

        flashButton = Button(self, text="Review Flashcards", command=lambda:controller.show_frame(Flashcards))
        gameButton = Button(self, text="Play Game", command=lambda:controller.show_frame(Game))

        returnButton = Button(self, text="Back to Categories")

        learnOrPlay.pack(fill=X)
        flashButton.pack(side="left")
        gameButton.pack(side="left")

        returnButton.pack()