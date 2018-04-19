from Tkinter import *
from learnOrPlay import learnOrPlay
import csv

class Level(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Add the level to the second row of the CSV file
        def addCSV1():
            myFile = open("game.csv", "a")

            with myFile:
                writer = csv.writer(myFile)
                writer.writerows([["1"]])

            myFile.close()

        def addCSV2():
            myFile = open("game.csv", "a")

            with myFile:
                writer = csv.writer(myFile)
                writer.writerows([["2"]])

            myFile.close()

        def addCSV3():
            myFile = open("game.csv", "a")

            with myFile:
                writer = csv.writer(myFile)
                writer.writerows([["3"]])

            myFile.close()

        def addCSV4():
            myFile = open("game.csv", "a")

            with myFile:
                writer = csv.writer(myFile)
                writer.writerows([["4"]])

            myFile.close()

        def addCSV5():
            myFile = open("game.csv", "a")

            with myFile:
                writer = csv.writer(myFile)
                writer.writerows([["5"]])

            myFile.close()


        # Widgets
        pickLabel = Label(self, text="Pick level:")

        button1 = Button(self, text="Level 1", command=lambda:addCSV1())
        button2 = Button(self, text="Level 2", command=lambda:addCSV2())
        button3 = Button(self, text="Level 3", command=lambda:addCSV3())
        button4 = Button(self, text="Level 4", command=lambda:addCSV4())
        button5 = Button(self, text="Level 5", command=lambda:addCSV5())

        cardOrGame = Button(self, text="Play or Cards", command=lambda:controller.show_frame(learnOrPlay))
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

