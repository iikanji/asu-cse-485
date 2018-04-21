import pygame
import random, os, sys
#from nlp import speech

pygame.init()

## Game settings
gameScreen = pygame.display.set_mode((800,480), 0, 32) 		#The 512, 512 screen dimensions may need to be modified with Pi screen size
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
		self.spriteWidth = (self.imageWidth)/(self.numSprites)
		self.spriteHeight = self.imageHeight
		self.sheet = pygame.image.load(self.spritesheet).convert_alpha()
		self.rect = pygame.Rect((self.imageNum,0),(self.spriteWidth, self.spriteHeight))
		self.ima = pygame.Surface((self.rect).size).convert()
		self.ima.blit(self.sheet,(0,0),self.rect)
	
	
	#sheet = pygame.image.load(self.spritesheet).convert_alpha()
	#rect = pygame.Rect((imageNum,0),(self.spriteWidth,self.spriteHeight))
	
	#ima = pygame.Surface(rect.size).convert()
	#ima.blit(sheet,(0,0),rect)

	def animate(self):
		self.imageNum += self.spriteWidth											
		if(self.imageNum >= self.imageWidth):  										
			self.imageNum = 0
		self.rect = pygame.Rect((self.imageNum, 0),(self.spriteWidth, self.spriteHeight))			
		self.ima.blit(self.sheet, (0,0), self.rect)
	
	def draw(self, screen):
		screen.blit(self.sheet, (self.xPos, self.yPos), self.rect)


##Information should be passed from interface
words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
wordCategory = 'Numbers'
lvl = 0

numEnemies = len(words)		#need to determine based on the words array length
enemyCounter = numEnemies + 1


enemy2Texture = 'enemy2.png'
enemy3Texture = 'enemy3.png'
groundTexture = 'BrownGround.png'
enemy1Texture = 'enemy1.png'
skyTexture = 'sky.png'
cloudTexture = 'BigCloud.png'
sky = pygame.image.load(skyTexture)
cloud = pygame.image.load(cloudTexture)
ground = pygame.image.load(groundTexture)
enemy1 = pygame.image.load(enemy1Texture)
enemy2 = pygame.image.load(enemy2Texture)
enemy3 = pygame.image.load(enemy3Texture)

basicfont = pygame.font.SysFont(None, 40)
guiText0 = 'Category: ' + wordCategory
text0 = basicfont.render(guiText0, True, (0, 0, 0), None)
textrect0 = text0.get_rect()
textrect0.centerx = 130 	
textrect0.centery = 15 		

guiText = 'Level: ' + str(lvl)
text = basicfont.render(guiText, True, (0, 0, 0), None)
textrect = text.get_rect()
textrect.centerx = 60 		#gameScreen.get_rect().centerx
textrect.centery = 45 		#gameScreen.get_rect().centery

correctText = basicfont.render(' Correct - C key', True, (0, 255, 0), None)		
correctTextRect = correctText.get_rect()
correctTextRect.centerx = 100
correctTextRect.centery = 450

incorrectText = basicfont.render('Incorrect - I key', True, (255, 0, 0), None)		
incorrectTextRect = incorrectText.get_rect()
incorrectTextRect.centerx = 100
incorrectTextRect.centery = 480

clock = pygame.time.Clock()
pygame.mouse.set_visible(False)
run = True

wrongList = []
correctList = []

## CREATE enemies and Gunma-chan using spritesheet class
#enemy1 =  Sprite(x, y, "spritesheet.png", numSprites)   #x,y (where you want it), image file name, number of sprites in sheet
#enemy2 =  Sprite(x, y, "spritesheet.png", numSprites)   #x,y (where you want it), image file name, number of sprites in sheet
#enemy3 =  Sprite(x, y, "spritesheet.png", numSprites)   #x,y (where you want it), image file name, number of sprites in sheet
enemyArray = [enemy1, enemy2, enemy3]
currEnemy = random.choice(enemyArray)
#gunmaAnim = GunmaSprite()
gunmaAnim = Sprite(75, 325, "testgunmaspritesheet.png", 8)   #x,y (where you want it), image file name, number of sprites in sheet


groundX = 0 
groundY = 300
cloud1X = 25
cloud1Y = 25
cloud2X = 150
cloud2Y = 50
cloud3X = 300
cloud3Y = 150
currEnemyX = 800
currEnemyY = 400
enemyStopperX = 250
enemyStopperY = 400

isPaused = False
isStopped = False

if not isPaused:
	currEnemySpeed = 5
	cloudSpeed = 5

currWord = words[0]
del words[0]
basicfont4 = pygame.font.SysFont(None, 48)
wordText = basicfont4.render(currWord, True, (0, 0, 0), None)
wordRect = wordText.get_rect()
wordRect.centerx = gameScreen.get_rect().centerx #currEnemyX + 15
wordRect.centery = gameScreen.get_rect().centery #currEnemyY - 25

pauseText = basicfont.render('Temp Pause Screen', True, (0, 0, 0), None)
pauseRect = pauseText.get_rect()
pauseRect.centerx = gameScreen.get_rect().centerx
pauseRect.centery = gameScreen.get_rect().centery

speakText = basicfont.render('', True, (0, 0, 0), None)
speakRect = speakText.get_rect()
speakRect.centerx = gameScreen.get_rect().centerx 
speakRect.centery = 384

#s = speech.Speech(lvl)


yBackgroundStart = secondBackground.get_width()
yBackgroundCurr = yBackgroundStart
yBackgroundOne = 0
# = 0

while True:
	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_p:
				isPaused = not isPaused
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
		   pygame.quit()
		   quit()
	
	gameScreen.blit(backgroundImage, (yBackgroundOne, 0))
	gameScreen.blit(backgroundImage, (yBackgroundCurr, 0))
	#gameScreen.blit(sky, (0,0))
	
	
	
	screenboundx, screenboundy = gameScreen.get_size()

	if currEnemyX <= -25 and len(words) + 1 > 0:		#Move enemy R -> L
		currEnemy = random.choice(enemyArray)
		currEnemyX = screenboundx
		currEnemyY = 400
		currWord = words.pop(0)
		wordText = basicfont4.render(currWord, True, (0, 0, 0), None)
		wordRect = wordText.get_rect()
		wordRect = wordText.get_rect()
		wordRect.centerx = gameScreen.get_rect().centerx#currEnemyX + 15
		wordRect.centery = gameScreen.get_rect().centery#currEnemyY - 25	

	if not isStopped:
		currEnemySpeed = 5
		cloudSpeed = 5
		gunmaAnim.animate()
		if yBackgroundOne == -800:
			yBackgroundOne = yBackgroundStart
		else:
			yBackgroundOne -= 2
	
		if(yBackgroundCurr > -800): 
			yBackgroundCurr -= 2
		else:
			yBackgroundCurr = yBackgroundStart
			
	if cloud1X <= -50:								#Move cloud1 R -> L
		cloud1X = screenboundx
		cloud1Y = 25 + random.randrange(100)
	
	if cloud2X <= -50:								#Move cloud2 R -> L
		cloud2X = screenboundx
		cloud2Y = 50 + random.randrange(150)
	
	if cloud3X <= -50:								#Move cloud2 R -> L
		cloud3X = screenboundx
		cloud3Y = 10 + random.randrange(50)
	
	if not len(words) == 0:
		#gameScreen.blit(ground, (groundX, groundY))
		gunmaAnim.draw(gameScreen)
		gameScreen.blit(currEnemy, (currEnemyX, currEnemyY))
		gameScreen.blit(wordText, wordRect)
		#gameScreen.blit(cloud, (cloud1X, cloud1Y))
		#gameScreen.blit(cloud, (cloud2X, cloud2Y))
		#gameScreen.blit(cloud, (cloud3X, cloud3Y))
		gameScreen.blit(text, textrect)
		gameScreen.blit(text0, textrect0)
		#gameScreen.blit(correctText, correctTextRect)
		#gameScreen.blit(incorrectText, incorrectTextRect)
	
	else:											#End game info
		stringEndText = 'Temp End Screen'
		endText = basicfont.render(stringEndText, True, (0, 0, 0), None)
		endRect = endText.get_rect()
		endRect.centerx = gameScreen.get_rect().centerx 
		endRect.centery = gameScreen.get_rect().centery - 100
		gameScreen.blit(endText, endRect)
		incList = 'Incorrect List: ' + ' , '.join(wrongList)
		incText = basicfont4.render(incList, True, (0, 0, 0), None)
		incRect = incText.get_rect()
		incRect.centerx = gameScreen.get_rect().centerx
		incRect.centery = gameScreen.get_rect().centery
		gameScreen.blit(incText, incRect)
		corrList = 'Correct List: ' + ' , '.join(correctList)
		corrText = basicfont4.render(corrList, True, (0, 0, 0), None)
		corrRect = corrText.get_rect()
		corrRect.centerx = gameScreen.get_rect().centerx
		corrRect.centery = gameScreen.get_rect().centery + 100
		gameScreen.blit(corrText, corrRect)
	
	if currEnemyX == enemyStopperX + 5:
		speakText = basicfont.render('Speak Now', True, (0, 0, 0), None)
		gameScreen.blit(speakText, speakRect)
	else:
		speakText = basicfont.render('', True, (0, 0, 0), None)
		gameScreen.blit(speakText, speakRect)

	if(currEnemyX == enemyStopperX):
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
	if not isPaused:														
		currEnemyX -= currEnemySpeed
		cloud1X -= cloudSpeed
		cloud2X -= cloudSpeed
		cloud3X -= cloudSpeed
		wordRect.centerx = gameScreen.get_rect().centerx #currEnemyX + 15
		wordRect.centery = gameScreen.get_rect().centery #currEnemyY - 25	
	else:
		gameScreen.blit(pauseText, pauseRect)	
		
	clock.tick(20)
	pygame.display.update()

