from Tkinter import *
from Categories import Categories
from Level import Level
from learnOrPlay import learnOrPlay
from Playing import Game
from Flashcards import Flashcards


class App(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        #Display the main menu at the top of the screen
        MainMenu(self)

        #Set up Frame
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for screen in (Categories, Level, learnOrPlay, Game, Flashcards):
            frame = screen(container, self)
            self.frames[screen] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Categories)

    def show_frame(self, context):
        frame = self.frames[context]
        frame.tkraise()


class MainMenu:
    def __init__(self, master):
        menubar = Menu(master)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=master.quit)
        menubar.add_cascade(label="Menu", menu=filemenu)
        master.config(menu=menubar)

interface = App()

interface.mainloop()