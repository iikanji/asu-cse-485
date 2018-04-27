import pygame
import random, os, sys
import csv
#from nlp import speech

pygame.init()

def callGame():


        ## Game settings
        gameScreen = pygame.display.set_mode((800, 480), 0, 32)  # The 512, 512 screen dimensions may need to be modified with Pi screen size
        pygame.display.set_caption('Gunma Pronunciation Game')
        backgroundTexture = 'AllBackground.png'
        backgroundImage = pygame.image.load(backgroundTexture)
        backgroundImage = pygame.transform.scale(backgroundImage, (800, 480))
        secondBackground = pygame.image.load(backgroundTexture)
        secondBackground = pygame.transform.scale(backgroundImage, (800, 480))

        ##Spritesheet animation class
        class Sprite():

            def __init__(self, xPos, yPos, spritesheetPng, numSprites):
                self.imageNum = 0
                self.xPos = xPos
                self.yPos = yPos
                self.spritesheet = spritesheetPng
                self.numSprites = numSprites
                self.imageWidth = pygame.image.load(self.spritesheet).get_width()
                self.imageHeight = pygame.image.load(self.spritesheet).get_height()
                self.spriteWidth = (self.imageWidth) / (self.numSprites)
                self.spriteHeight = self.imageHeight
                self.sheet = pygame.image.load(self.spritesheet).convert_alpha()
                self.rect = pygame.Rect((self.imageNum, 0), (self.spriteWidth, self.spriteHeight))
                self.ima = pygame.Surface((self.rect).size).convert()
                self.ima.blit(self.sheet, (0, 0), self.rect)

            def animate(self):
                self.imageNum += self.spriteWidth
                if (self.imageNum >= self.imageWidth):
                    self.imageNum = 0
                self.rect = pygame.Rect((self.imageNum, 0), (self.spriteWidth, self.spriteHeight))
                self.ima.blit(self.sheet, (0, 0), self.rect)

            def draw(self, screen):
                screen.blit(self.sheet, (self.xPos, self.yPos), self.rect)

        ##Information should be passed from interface
        words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        wordCategory = 'Numbers'
        lvl = 0


        # with open("game.csv", 'r') as File:
        #     reader = csv.reader(File)
        #
        #     rows = list(reader)
        #
        #     wordCategory = " ".join(rows[0])
        #
        #     words = rows[1]
        #
        #     lvl = " ".join(rows[2])
        #
        # File.close

        numEnemies = len(words)
        enemyCounter = numEnemies - 1
        currWord = words[0]
        words.pop(0)

        enemy1Texture = 'enemy1.png'
        enemy2Texture = 'enemy2.png'
        enemy3Texture = 'enemy3.png'
        enemy1 = pygame.image.load(enemy1Texture)
        enemy2 = pygame.image.load(enemy2Texture)
        enemy3 = pygame.image.load(enemy3Texture)

        basicfont = pygame.font.SysFont(None, 40)
        guiText0 = 'Category: ' + wordCategory
        text0 = basicfont.render(guiText0, True, (0, 0, 0), None)
        textrect0 = text0.get_rect()
        textrect0.centerx = 135
        textrect0.centery = 15

        guiText = 'Level: ' + str(lvl)
        text = basicfont.render(guiText, True, (0, 0, 0), None)
        textrect = text.get_rect()
        textrect.centerx = 60
        textrect.centery = 45

        clock = pygame.time.Clock()
        pygame.mouse.set_visible(False)
        run = True

        wrongList = []
        correctList = []

        ## CREATE enemies and Gunma-chan using spritesheet class
        # enemy1 =  Sprite(x, y, "spritesheet.png", numSprites)   #x,y (where you want it), image file name, number of sprites in sheet
        # enemy2 =  Sprite(x, y, "spritesheet.png", numSprites)   #x,y (where you want it), image file name, number of sprites in sheet
        # enemy3 =  Sprite(x, y, "spritesheet.png", numSprites)   #x,y (where you want it), image file name, number of sprites in sheet
        enemyArray = [enemy1, enemy2, enemy3]
        currEnemy = random.choice(enemyArray)
        # gunmaAnim = GunmaSprite()
        gunmaAnim = Sprite(75, 325, "testgunmaspritesheet.png",
                           8)  # x,y (where you want it), image file name, number of sprites in sheet

        currEnemyX = 800
        currEnemyY = 400
        enemyStopperX = 250
        enemyStopperY = 400

        isStopped = False

        currEnemySpeed = 5

        basicfont4 = pygame.font.SysFont(None, 52)
        wordText = basicfont4.render(currWord, True, (0, 0, 0), None)
        wordRect = wordText.get_rect()
        wordRect.centerx = gameScreen.get_rect().centerx
        wordRect.centery = gameScreen.get_rect().centery

        pauseText = basicfont.render('Temp Pause Screen', True, (0, 0, 0), None)
        pauseRect = pauseText.get_rect()
        pauseRect.centerx = gameScreen.get_rect().centerx
        pauseRect.centery = gameScreen.get_rect().centery

        speakText = basicfont.render('', True, (0, 0, 0), None)
        speakRect = speakText.get_rect()
        speakRect.centerx = 550  # gameScreen.get_rect().centerx
        speakRect.centery = 120

        # s = speech.Speech(lvl)

        # Paralax Background Initialization
        yBackgroundStart = secondBackground.get_width()
        yBackgroundCurr = yBackgroundStart
        yBackgroundOne = 0

        endGame = False
        runGame = True

        while runGame:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # if event.key == pygame.K_p:
                    # isPaused = not isPaused
                    if event.key == pygame.K_c:
                        if isStopped:
                            correctList.append(currWord)
                            isStopped = False
                            currEnemyX -= 3
                    if event.key == pygame.K_i:
                        if isStopped:
                            wrongList.append(currWord)
                            isStopped = False
                            currEnemyX -= 3
                if event.type == pygame.QUIT:
                    #File.close;
                    pygame.quit()
                    quit()

            gameScreen.blit(backgroundImage, (yBackgroundOne, 0))
            gameScreen.blit(backgroundImage, (yBackgroundCurr, 0))

            screenboundx, screenboundy = gameScreen.get_size()

            if currEnemyX <= -25 and enemyCounter > 0:  # Move enemy R -> L
                currEnemy = random.choice(enemyArray)
                currEnemyX = screenboundx
                currEnemyY = 400
                currWord = words.pop(0)
                wordText = basicfont4.render(currWord, True, (0, 0, 0), None)
                wordRect = wordText.get_rect()
                wordRect = wordText.get_rect()
                wordRect.centerx = gameScreen.get_rect().centerx
                wordRect.centery = gameScreen.get_rect().centery
                enemyCounter -= 1

            if (currEnemyX == enemyStopperX):
                currEnemySpeed = 0
                cloudSpeed = 0
                isStopped = True
                """
                #call NLP
                if s.record_and_validate(currWord):
                    correctList.append(currWord)		
                    isStopped = False
                    currEnemyX -= 3
                else:
                    wrongList.append(currWord)
                    isStopped = False
                    currEnemyX -= 3	
                """

            if enemyCounter > 0:  # Run game
                gameScreen.blit(wordText, wordRect)
                gameScreen.blit(text, textrect)
                gameScreen.blit(text0, textrect0)
                gunmaAnim.draw(gameScreen)
                gameScreen.blit(currEnemy, (currEnemyX, currEnemyY))
            else:  # End game info
                endGame = True
                isStopped = True
                endText = basicfont4.render('Game Complete', True, (0, 0, 0), None)
                endRect = endText.get_rect()
                endRect.centerx = gameScreen.get_rect().centerx
                endRect.centery = gameScreen.get_rect().centery - 125
                gameScreen.blit(endText, endRect)
                incList = 'Incorrect List: ' + ' , '.join(wrongList)
                incText = basicfont.render(incList, True, (255, 255, 255), None)
                incRect = incText.get_rect()
                incRect.centerx = gameScreen.get_rect().centerx
                incRect.centery = gameScreen.get_rect().centery - 25
                gameScreen.blit(incText, incRect)
                corrList = 'Correct List: ' + ' , '.join(correctList)
                corrText = basicfont.render(corrList, True, (255, 255, 255), None)
                corrRect = corrText.get_rect()
                corrRect.centerx = gameScreen.get_rect().centerx
                corrRect.centery = gameScreen.get_rect().centery + 75
                gameScreen.blit(corrText, corrRect)

            if not isStopped:  # If the enemy is not stopped
                currEnemySpeed = 5
                gunmaAnim.animate()
                if yBackgroundOne == -800:
                    yBackgroundOne = yBackgroundStart
                else:
                    yBackgroundOne -= 2

                if (yBackgroundCurr > -800):
                    yBackgroundCurr -= 2
                else:
                    yBackgroundCurr = yBackgroundStart

                currEnemyX -= currEnemySpeed
                speakText = basicfont.render('', True, (0, 0, 0), None)
                gameScreen.blit(speakText, speakRect)
            else:  # Enemy is stopped
                if not endGame:
                    speakText = basicfont.render('Speak Now', True, (0, 0, 0), None)
                    gameScreen.blit(speakText, speakRect)

            """
            if currEnemyX == enemyStopperX + 5:
                speakText = basicfont.render('Speak Now', True, (0, 0, 0), None)
                gameScreen.blit(speakText, speakRect)
            elif currEnemyX < enemyStopperX:
                speakText = basicfont.render('', True, (0, 0, 0), None)
                gameScreen.blit(speakText, speakRect)
            """

            """
            if not isPaused:														
                currEnemyX -= currEnemySpeed
                wordRect.centerx = gameScreen.get_rect().centerx 
                wordRect.centery = gameScreen.get_rect().centery 
            else:
                gameScreen.blit(pauseText, pauseRect)	
            """

            clock.tick(20)
            pygame.display.update()

