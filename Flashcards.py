from Tkinter import *

class Flashcards(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        #Widgets
        topFrame = Frame(self)
        bottomFrame = Frame(self)

        reviewLabel = Label(topFrame, text="You are reviewing", fg="red")
        imageLabel = Label(topFrame, text="Pretend this is an image")

        cardOrGame = Button(bottomFrame, text="Back to Play or Cards")
        categories = Button(bottomFrame, text="Back to Categories")


        #Organization
        topFrame.pack()
        bottomFrame.pack(side=BOTTOM)

        reviewLabel.pack()
        imageLabel.pack(side=BOTTOM)

        cardOrGame.pack()
        categories.pack()