from Tkinter import *
from learnOrPlay import learnOrPlay

class Level(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Widgets
        pickLabel = Label(self, text="Pick level:")

        button1 = Button(self, text="Level 1", command=lambda:controller.show_frame(learnOrPlay))
        button2 = Button(self, text="Level 2", command=lambda:controller.show_frame(learnOrPlay))
        button3 = Button(self, text="Level 3", command=lambda:controller.show_frame(learnOrPlay))
        button4 = Button(self, text="Level 4", command=lambda:controller.show_frame(learnOrPlay))
        button5 = Button(self, text="Level 5", command=lambda:controller.show_frame(learnOrPlay))

        cardOrGame = Button(self, text="Back to Play or Cards")
        categories = Button(self, text="Back to Categories")

        # Organization
        pickLabel.pack(fill=X)

        button1.pack(fill=X)
        button2.pack(fill=X)
        button3.pack(fill=X)
        button4.pack(fill=X)
        button5.pack(fill=X)

        cardOrGame.pack(fill=X)
        categories.pack(fill=X)