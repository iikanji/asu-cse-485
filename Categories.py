from Tkinter import *
from Level import Level
import csv

class Categories(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        #Widgets
        label = Label(self, text="Pick Category:")

        #Add the category to the first row of the CSV file
        def addNumCSV():
            #Read in info from numbers.csv
            with open("numbers.csv") as File:
                reader = csv.reader(File)
                rows = list(reader)
                words = rows[0]
            File.close


            gameFile = open("game.csv", "a")

            with gameFile:
                writer = csv.writer(gameFile)
                #Add "Category"
                writer.writerows([["Numbers"]])
                #Add list of words
                writer.writerows([words])

            #Close the file
            gameFile.close()

        def addColCSV():
            myFile = open("game.csv", "a")

            with myFile:
                writer = csv.writer(myFile)
                writer.writerows([["Colors"]])

            myFile.close()

        button1 = Button(self, text="1. Numbers", command=lambda:addNumCSV())
        custButton1 = Button(self, text="Custom 1")

        button2 = Button(self, text="2. Colors and Shapes", command=lambda:addColCSV())
        custButton2 = Button(self, text="Custom 2")

        button3 = Button(self, text="3. Fruits and Vegetables")
        custButton3 = Button(self, text="Custom 3")

        button4 = Button(self, text="4. Feelings")
        custButton4 = Button(self, text="Custom 4")

        button5 = Button(self, text="5. Alphabet")

        button6 = Button(self, text="6. Family")

        levelButton = Button(self, text="CHOOSE LEVEL", command=lambda:controller.show_frame(Level))

        #Organization
        label.grid(columnspan=2)

        button1.grid(row=1)
        custButton1.grid(row=1, column=1)

        button2.grid(row=2)
        custButton2.grid(row=2, column=1)

        button3.grid(row=3)
        custButton3.grid(row=3, column=1)

        button4.grid(row=4)
        custButton4.grid(row=4, column=1)

        button5.grid(row=5)

        button6.grid(row=6)

        levelButton.grid(row=5, column=1)