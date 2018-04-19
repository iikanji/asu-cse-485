from Tkinter import *

class Game(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        #Widgets
        topFrame = Frame(self)
        bottomFrame = Frame(self)

        playLabel = Label(topFrame, text="You are playing", fg="red")
        imageLabel = Label(topFrame, text="Pretend this is an image")

        cardOrGame = Button(bottomFrame, text="Back to Play or Cards")
        categories = Button(bottomFrame, text="Back to Categories")


        #Organization
        topFrame.pack()
        bottomFrame.pack(side=BOTTOM)

        playLabel.pack()
        imageLabel.pack(side=BOTTOM)

        cardOrGame.pack()
        categories.pack()