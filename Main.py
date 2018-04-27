from Tkinter import *
from Categories import Categories
from Level import Level
from learnOrPlay import learnOrPlay
from Flashcards import Flashcards


class App(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)


        #Display the main menu at the top of the screen
        MainMenu(self)

        self.wm_minsize(width=800,height=480)
        self.resizable(width=False,height=False)


        #Set up Frame
        container = Frame(self,width=480,height=100)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        #container.size()

        self.frames = {}

        for screen in (StartScreen, Categories, Level, learnOrPlay, Flashcards):
            frame = screen(container, self)
            self.frames[screen] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartScreen)

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

class StartScreen(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Widgets
        label = Label(self, text="Welcome to the game")

        # Organization
        label.grid(columnspan=2)

        startButton = Button(self, text="Enter", command=lambda: controller.show_frame(Categories))
        startButton.grid(row=1)

interface = App()

interface.mainloop()