#import
import sys
import time
import random
import os

import numpy as np

import pygame as pg
from pygame.locals import Color, QUIT, MOUSEBUTTONDOWN, USEREVENT, USEREVENT
from pygame import *
from time import sleep


#使用變數先指定參數
WINDOW_WIDTH = 300  #遊戲畫面寬和高
WINDOW_HEIGHT = 300
WHITE = (255, 255, 255)
IMAGEWIDTH = 64
IMAGEHEIGHT = 64
FPS = 60

#pygame初始化
pg.init()


#指定全域變數，物件
table =[[0 for i in range(4)] for i in range(4)]
newItemTable=[[0 for i in range(4)] for i in range(4)]
moveTable = [[0 for i in range(4)] for i in range(4)]
oldTable=[[0 for i in range(4)] for i in range(4)]
spriteTable = [[] for i in range(4)]
allSprite = pg.sprite.Group()
font = pg.font.Font(os.path.join('2048ByNCC', 'HanyiSentyCrayon.ttf'), 24)

class Background(pg.sprite.Sprite): #background的精靈類別
    def __init__(self, image_file, location):
        super().__init__()  #call Sprite initializer
        self.raw_image = pg.image.load(image_file).convert_alpha()
        self.image = pg.transform.scale(self.raw_image, (300, 300))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location
        self.width = 300
        self.height = 300
        

class block(pg.sprite.Sprite): #每個Block的精靈類別
    def __init__(self, width, height, x, y, image_file):
        super().__init__()
        # 載入圖片
        self.raw_image = pg.image.load(image_file).convert_alpha()
        # 縮小圖片
        self.image = pg.transform.scale(self.raw_image, (width, height))
        # 回傳位置
        self.rect = self.image.get_rect()
        # 定位
        self.rect.topleft = (x, y)
        self.width = width
        self.height = height
    def update(self, i, j, direction,finishAnimation,finishMoveAnimation):
        global moveTable
        global table
        global newItemTable
        global oldTable
        bx,by=self.rect.topleft
        if finishAnimation:
            self.rect.topleft=(10+64*j+8*j+1, 10+64*i+8*i)
            #先移回來 10+64*j+8*j, 10+64*i+8*i
            image_file=os.path.join('2048ByNCC', str(table[i][j])+'.png')
            # 載入圖片
            self.raw_image = pg.image.load(image_file).convert_alpha()
            # 縮小圖片
            self.image = pg.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
        else:
            if not finishMoveAnimation:
                if moveTable[i][j]!=0:
                    if direction == 1: #up
                        self.rect.topleft=(bx,by-18)
                    elif direction == 2: #down
                        self.rect.topleft=(bx,by+18)
                    elif direction == 3: #left
                        self.rect.topleft=(bx-18,by)
                    else : #right
                        self.rect.topleft=(bx+18,by)
                    moveTable[i][j]-=0.25
#                    if moveTable[i][j]==0:
                    image_file=os.path.join('2048byncc', str(oldTable[i][j])+'.png')
                    self.raw_image = pg.image.load(image_file).convert_alpha()
                    # 縮小圖片
                    self.image = pg.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
            else:
                if moveTable[i][j]==0 :
                    if newItemTable[i][j] != 0:
                        newItemTable[i][j]-=0.25
                        wah=newItemTable[i][j]/0.25
                        self.rect.topleft=(10+64*j+8*j+(32-int(4**(3-wah)/2)), 10+64*i+8*i+(32-int(4**(3-wah)/2)))
                        #先移回來 10+64*j+8*j, 10+64*i+8*i
                        image_file=os.path.join('2048ByNCC', str(table[i][j])+'.png')
                        # 載入圖片
                        self.raw_image = pg.image.load(image_file).convert_alpha()
                        # 縮小圖片
                        self.image = pg.transform.scale(self.raw_image, (int(pow(4,3-wah)), int(pow(4,3-wah))))
                            #做縮放
                    else:
                        self.rect.topleft=(10+64*j+8*j+1, 10+64*i+8*i)
                        #先移回來 10+64*j+8*j, 10+64*i+8*i
                        image_file=os.path.join('2048ByNCC', str(table[i][j])+'.png')
                        # 載入圖片
                        self.raw_image = pg.image.load(image_file).convert_alpha()
                        # 縮小圖片
                        self.image = pg.transform.scale(self.raw_image, (IMAGEWIDTH, IMAGEHEIGHT))
            
            #一格花4次跑到，最後花8次放大縮小圖片，一次移動18px，
            #新增一個oldtable~~


#設定視窗
main_clock = pg.time.Clock()        
screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))   #依設定顯示視窗
pg.display.set_caption("2048 by ncc")           #設定程式標題
BackGround = Background(os.path.join('2048ByNCC', 'bg.png'), [0,0]) #指定background物件


def RandomNewItem():
    while True:
        for i in range(3,-1,-1):#y
            for j in range(4):#x
                if table[i][j] == 0:
                    if random.randint(1,16) == 1 :
                        if random.random() > 0.5:
                            table[i][j] = 2
                        else:
                            table[i][j] = 4
                        return i,j
                        
def init_train():
    init()
    for i in range(4):#y        
        for j in range(4):#x\
            spriteTable[i][j].update(i,j,-1,True,True)
    screen.blit(BackGround.image, BackGround.rect)
    allSprite.draw(screen)
    pg.display.update()
    
def init():
    for i in range(4):#y        
        for j in range(4):#x\
            table[i][j]=0
    # 亂數兩格來
    rn=random.randint(1,4) # randomNumber 三種機緣任君挑選
    if rn==1:
        position=random.randint(8,16)-1
        table[position%4][position//4]=2
        position=random.randint(1,position)-1
        table[position%4][position//4]=2
    elif rn==2:
        position=random.randint(8,16)-1
        table[position%4][position//4]=4
        position=random.randint(1,position)-1
        table[position%4][position//4]=2
    else:
        position=random.randint(8,16)-1
        table[position%4][position//4]=2
        position=random.randint(1,position)-1
        table[position%4][position//4]=4
    
    #print("table:") #debug 用
    #for i in range(4):#y        
    #        for j in range(4):#x
    #            print(table[i][j],end=" ")
    #        print()
            
    for i in range(4):
        for j in range(4):
            spriteTable[i].append(block(IMAGEWIDTH, IMAGEHEIGHT, 10+64*j+8*j+1, 10+64*i+8*i, os.path.join('2048ByNCC', str(table[i][j])+'.png')))
            allSprite.add(spriteTable[i][j])
    # TheIndexLikeBelow !
    # 0,0 0,1 0,2 0,3
    # 1,0 1,1 1,2 1,3
    # 2,0 2,1 2,2 2,3
    # 3,0 3,1 3,2 3,3

def showGameOver():
    #加畫字
    print("GameOver!")
    while True:
        background2= Background(os.path.join('2048ByNCC', 'background.png'), [0,0]) #指定background2物件
        text = font.render("哈哈，你輸惹，偶是念誠", True, (255, 255, 255))
        text2 = font.render("按[ENTER]重玩、[ESC]關掉", True, (255, 255, 255))
        
        for event in pg.event.get():
            if event.type == pg.QUIT: #關閉程式的判斷
                pg.quit()  #關閉程式的程式碼
                sys.exit()
        keys=pg.key.get_pressed()
        if keys[K_ESCAPE]:
            print("exiting")
            pg.quit()  #關閉程式的程式碼
            sys.exit()
        if keys[K_RETURN]:
            print("Restarting")
            os.execl(sys.executable, sys.executable, *sys.argv)
        screen.blit(background2.image,background2.rect)
        screen.blit(text, (20, 20))
        screen.blit(text2, (20, 80))
        pg.display.update()
        main_clock.tick(FPS)
    #os.execl(sys.executable, sys.executable, *sys.argv)
def showRestart():
    fnt = pg.font.Font(os.path.join('2048ByNCC', 'HanyiSentyCrayon.ttf'), 50)
    text = fnt.render("failed...restart", True, (255, 255, 255))
    screen.blit(text, (20, 120))
    sleep(1)
    #os.execl(sys.executable, sys.executable, *sys.argv)
def noMoreStep(table):
    fulled=True
    for i in range(4):#y
        for j in range(4):#x
            if table[i][j]==0:
                fulled=True
                break
    if fulled:
        for i in range(3,0,-1):#y
            for j in range(4):#x
                if table[i][j]!=0 :
                    if table[i-1][j] ==table[i][j] :
                        return False
                    if table[i-1][j]==0:
                        return False
        for i in range(3):#y
            for j in range(4):#x
                if table[i][j]!=0 :
                    if table[i+1][j] ==table[i][j] :
                        return False
                    if table[i+1][j]==0:
                        return False
        for i in range(3,0,-1):#x
            for j in range(4):#y
                if table[j][i]!=0:
                    if table[j][i-1] ==table[j][i] :
                        return False
                    if table[j][i-1]==0:
                        return False
        for i in range(3):#x
            for j in range(4):#y
               if table[j][i]!=0:
                    if table[j][i+1] ==table[j][i] :
                        return False
                    if table[j][i+1]==0:
                        return False
        return True

def movable(table,direction):
    if direction==1:
        for i in range(3,0,-1):#y
            for j in range(4):#x
                if table[i][j]!=0 :
                    if table[i-1][j] ==table[i][j] :
                        return True
                    if table[i-1][j]==0:
                        return True

    elif direction==2:
        for i in range(3):#y
            for j in range(4):#x
                if table[i][j]!=0 :
                    if table[i+1][j] ==table[i][j] :
                        return True
                    if table[i+1][j]==0:
                        return True
    elif direction==3:
        for i in range(3,0,-1):#x
            for j in range(4):#y
                if table[j][i]!=0:
                    if table[j][i-1] ==table[j][i] :
                        return True
                    if table[j][i-1]==0:
                        return True
    else:
        for i in range(3):#x
            for j in range(4):#y
               if table[j][i]!=0:
                    if table[j][i+1] ==table[j][i] :
                        return True
                    if table[j][i+1]==0:
                        return True
    return False

def move(direction):
    combineCount = 0
    score = 0
    global moveTable
    global table
    global oldTable
    for i in range(4):
        for j in range(4):
            newItemTable[i][j]=0
            moveTable[i][j]=0
            oldTable[i][j]=table[i][j]
    needMove=[[True for i in range(4)]for j in range(4)]
    if direction==1: #up
        #由上到下，左到右，先排好
        for i in range(1,4,1):#y
            for j in range(4):#x
                if (table[i][j]!=0):
                    for k in range(i-1,-1,-1): # 由下到上
                        if table[k][j] == 0:
                            table[k][j]=table[k+1][j]
                            table[k+1][j]=0
                            if k==0:
                                moveTable[i][j]+=i-k
                        elif table[k][j]!=0:
                            moveTable[i][j]+=i-k-1
                            break
        
        #print("up")
        #由下到上，左到右，作檢查
        for i in range(3,0,-1):#y
            for j in range(4):#x
                if needMove[i][j]:
                    if table[i][j] ==table[i-1][j] and table[i][j]!=0 :
                        #moveTable[i][j]+=1
                        table[i-1][j] *=2
                        score = score + table[i-1][j]
                        newItemTable[i-1][j] = 1
                        needMove[i-1][j]=False
                        table[i][j] = 0
                        combineCount +=1
                        
        #由上到下，左到右，整理好
        for i in range(1,4,1):#y
            for j in range(4):#x
                if (table[i][j]!=0):
                    if table[i-1][j] == 0: 
                        table[i-1][j]=table[i][j]
                        table[i][j]=0
                        #moveTable[i][j]+=i-1-k
                        if newItemTable[i][j] == 1: #排改變位置的表格
                            newItemTable[i-1][j] = 1
                            newItemTable[i][j] =0

    elif direction==2:
        #由下到上，左到右，先排好
        for i in range(2,-1,-1):#y
            for j in range(4):#x
                if (table[i][j]!=0):
                    for k in range(i+1,4,1): # 由上到下
                        if table[k][j] == 0:
                            table[k][j]=table[k-1][j]
                            table[k-1][j]=0
                            if k==3:
                                moveTable[i][j]+=k-i
                        elif table[k][j]!=0:
                            moveTable[i][j]+=k-i-1
                            break
        
        #print("down")
        #由上到下，左到右，作檢查
        for i in range(3):#y
            for j in range(4):#x
                if needMove[i][j]:
                    if table[i][j] ==table[i+1][j] and table[i][j]!=0 :
                        #moveTable[i][j]+=1
                        table[i+1][j] *=2
                        score = score + table[i+1][j]
                        newItemTable[i+1][j] = 1
                        needMove[i+1][j]=False
                        table[i][j] = 0
                        combineCount +=1
                        
        #由下到上，左到右，整理好
        for i in range(2,-1,-1):#y
            for j in range(4):#x
                if (table[i][j]!=0):
                    if table[i+1][j] == 0: 
                        table[i+1][j]=table[i][j]
                        table[i][j]=0
                        #moveTable[i][j]+=i-1-k
                        if newItemTable[i][j] == 1: #排改變位置的表格
                            newItemTable[i+1][j] = 1
                            newItemTable[i][j] =0

    elif direction==3:
        #由左到又，由上到下，先排好
        for i in range(1,4,1): #x
            for j in range(4): #y
                if (table[j][i]!=0):
                    for k in range(i-1,-1,-1): # 由又到左 k=>x
                        if table[j][k] == 0:
                            table[j][k]=table[j][k+1]
                            table[j][k+1]=0
                            if k==0:
                                moveTable[j][i]+=i-k
                        elif table[j][k]!=0:
                            moveTable[j][i]+=i-k-1
                            break
        
        #print("left")
        #由又到左，由上到下，作檢查
        for i in range(3,0,-1):#x
            for j in range(4):#y
                if needMove[j][i]:
                    if table[j][i] ==table[j][i-1] and table[j][i]!=0 :
                        #moveTable[i][j]+=1
                        table[j][i-1] *=2
                        score = score + table[j][i-1]
                        newItemTable[j][i-1] = 1
                        needMove[j][i-1]=False
                        table[j][i] = 0
                        combineCount +=1
                        
        #由左到又，由上到下，整理好
        for i in range(1,4,1):#x
            for j in range(4):#y
                if (table[j][i]!=0):
                    if table[j][i-1] == 0: 
                        table[j][i-1]=table[j][i]
                        table[j][i]=0
                        #moveTable[i][j]+=i-1-k
                        if newItemTable[j][i] == 1: #排改變位置的表格
                            newItemTable[j][i-1] = 1
                            newItemTable[j][i] =0
                                             
    else:
        #由又到左，由上到下，先排好
        for i in range(2,-1,-1): #x
            for j in range(4): #y
                if (table[j][i]!=0):
                    for k in range(i+1,4,1): # 由左到又移動 k=>x
                        if table[j][k] == 0:
                            table[j][k]=table[j][k-1]
                            table[j][k-1]=0
                            if k==3:
                                moveTable[j][i]+=k-i
                        elif table[j][k]!=0:
                            moveTable[j][i]+=k-i-1
                            break
        
        #print("right")
        #由左到又，由上到下，作檢查
        for i in range(3):#x
            for j in range(4):#y
                if needMove[j][i]:
                    if table[j][i] ==table[j][i+1] and table[j][i]!=0 :
                        #moveTable[i][j]+=1
                        table[j][i+1] *=2
                        score = score + table[j][i+1]
                        newItemTable[j][i+1] = 1
                        needMove[j][i+1]=False
                        table[j][i] = 0
                        combineCount +=1
                        
        #由又到左，由上到下，整理好
        for i in range(2,-1,-1):#x
            for j in range(4):#y
                if (table[j][i]!=0):
                    if table[j][i+1] == 0: 
                        table[j][i+1]=table[j][i]
                        table[j][i]=0
                        #moveTable[i][j]+=i-1-k
                        if newItemTable[j][i] == 1: #排改變位置的表格
                            newItemTable[j][i+1] = 1
                            newItemTable[j][i] =0
    
    x,y=RandomNewItem()
    newItemTable[x][y]=1
    return score
    
def MOVE(direction):    
    score = 0
    global oldTable
    global table
    
    for i in range(4):
        for j in range(4):
            oldTable[i][j]=table[i][j]
            
    not_merged = [[True,True,True,True],
                  [True,True,True,True],
                  [True,True,True,True],
                  [True,True,True,True]]
    if direction == 1: # up
        for i in range(1,4,1):
            for j in range(4):
                # move up one by one, start from top
                for k in range(i,-1,-1):
                    if k > 0:
                        if table[k-1][j] == 0:
                            table[k-1][j] = table[k][j]
                            table[k][j] = 0
                        elif table[k-1][j] == table[k][j] and not_merged[k-1][j] and not_merged[k][j]:
                            table[k-1][j] *=2
                            score += table[k-1][j]
                            table[k][j] = 0
                            not_merged[k-1][j] = False
                            break
    if direction == 2: # down
        for i in range(2,-1,-1):
            for j in range(4):
                # move down one by one, start from bottom
                for k in range(i,4,1):
                    if k < 3:
                        if table[k+1][j] == 0:
                            table[k+1][j] = table[k][j]
                            table[k][j] = 0
                        elif table[k+1][j] == table[k][j] and not_merged[k+1][j] and not_merged[k][j]:
                            table[k+1][j] *=2
                            score += table[k+1][j]
                            table[k][j] = 0
                            not_merged[k+1][j] = False
                            break
    if direction == 3: # left
        for j in range(1,4,1):
            for i in range(4):
                # move left one by one, start from left
                for k in range(j,-1,-1):
                    if k > 0:
                        if table[i][k-1] == 0:
                            table[i][k-1] = table[i][k]
                            table[i][k] = 0
                        elif table[i][k-1] == table[i][k] and not_merged[i][k-1] and not_merged[i][k]:
                            table[i][k-1] *=2
                            score += table[i][k-1]
                            table[i][k] = 0
                            not_merged[i][k-1] = False
                            break                
    if direction == 4: # right
        for j in range(2,-1,-1):
            for i in range(4):
                # move right one by one, start from right
                for k in range(j,4,1):
                    if k < 3:
                        if table[i][k+1] == 0:
                            table[i][k+1] = table[i][k]
                            table[i][k] = 0
                        elif table[i][k+1] == table[i][k] and not_merged[i][k+1] and not_merged[i][k]:
                            table[i][k+1] *=2
                            score += table[i][k+1]
                            table[i][k] = 0
                            not_merged[i][k+1] = False
                            break
    #x,y=RandomNewItem()
    return score
def showMoving(direction):
    need_move_before = [[0 for i in range(4)] for i in range(4)]
    need_move_after  = [[0 for i in range(4)] for i in range(4)]
    move_steps  = [[0 for i in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            # compare current table with old table
            # find only where differences occur
            if table[i][j] != oldTable[i][j]:
                if oldTable[i][j] > 0:
                    need_move_before[i][j] = oldTable[i][j]
                if table[i][j] > 0:
                    need_move_after[i][j] = table[i][j]
    if direction == 1:#up
        for i in range(3,0,-1):
            for j in range(4):
                if need_move_before[i][j] > 0:
                    for k in range(i-1,-1,-1):
                        if need_move_after[k][j] > 0:
                            move_steps[i][j] = i-k
                            break
    if direction == 2:#down
        for i in range(3):
            for j in range(4):
                if need_move_before[i][j] > 0:
                    for k in range(i+1,4,1):
                        if need_move_after[k][j] > 0:
                            move_steps[i][j] = k-i
                            break
    if direction == 3:#left
        for j in range(3,0,-1):
            for i in range(4):
                if need_move_before[i][j] > 0:
                    for k in range(j-1,-1,-1):
                        if need_move_after[i][k] > 0:
                            move_steps[i][j] = j-k
                            break                            
    if direction == 4:#right
        for j in range(3):
            for i in range(4):
                if need_move_before[i][j] > 0:
                    for k in range(j+1,4,1):
                        if need_move_after[i][k] > 0:
                            move_steps[i][j] = k-j
                            break
    #print(np.array(oldTable))
    #print(np.array(table))
    #print('--- needMove before ----')
    #print(np.array(need_move_before))
    #print('--- needMove after ----')
    #print(np.array(need_move_after))
    #print('--- steps ---')
    #print(np.array(move_steps))
    for t in range(10):
        k = t/10.0
        for i in range(4):
            for j in range(4):
                bx,by=spriteTable[i][j].rect.topleft
                w = 16#spriteTable[i][j].rect.width
                d = move_steps[i][j]

                if direction ==1:#up
                    spriteTable[i][j].rect.topleft = (bx,by-w*d*k)
                elif direction ==2:#down
                    spriteTable[i][j].rect.topleft = (bx,by+w*d*k)
                elif direction ==3:#left
                    spriteTable[i][j].rect.topleft = (bx-w*d*k,by)
                else :#right
                    spriteTable[i][j].rect.topleft = (bx+w*d*k,by)                    
        screen.blit(BackGround.image, BackGround.rect)
        allSprite.draw(screen)
        pg.display.update()
        main_clock.tick(FPS)
def main():
    init()
    global Background
    global table
    direction =0 
    running = True
    finishAnimation=True
    finishMoveAnimation=False
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT: #關閉程式的判斷
                running = False
        if not finishAnimation:
            for i in range(4):#y        
                for j in range(4):#x
                    spriteTable[i][j].update(i,j,direction,finishAnimation,finishMoveAnimation)
                    
            AMAF = False
            AAF = False
            finishMoveAnimation=False
            for i in range(3,-1,-1):#y
                for j in range(4):#x
                    if moveTable[i][j] == 0 :
                        AMAF = True
                    else :
                        AMAF = False
                        break
                if not AMAF:
                    break
            for i in range(3,-1,-1):#y
                for j in range(4):#x
                    if newItemTable[i][j] == 0 and moveTable[i][j] == 0:
                        AAF = True
                    else :
                        AAF = False
                        break
                if not AAF:
                    break
            if AMAF:
                finishMoveAnimation=True
            if AAF :
                finishAnimation=True
            
#            
#            print("MoveTable:")
#            for k in range(4):#y        
#                for w in range(4):#x
#                    print(moveTable[k][w],end=" ")
#                print()
#            print(" Table:")
#            for i in range(4):#y        
#                for j in range(4):#x
#                    print(table[i][j],end=" ")
#                print()
#            print("newItemTable")
#            for i in range(4):#y        
#                for j in range(4):#x
#                    print(newItemTable[i][j],end=" ")
#                print()
                        
        elif finishMoveAnimation and finishAnimation:
            for i in range(4):#y        
                for j in range(4):#x
                    spriteTable[i][j].update(i,j,direction,finishAnimation,finishMoveAnimation)
            finishMoveAnimation=False
            if noMoreStep(table):
                sleep(1)
                for i in range(4):#y        
                    for j in range(4):#x\
                        table[i][j]=0
                        spriteTable[i][j].update(i,j,direction,True,True)
                #Background.update(os.path.join('2048ByNCC', 'background.png')) #指定background物件
                showGameOver()
        else :
            keys=pg.key.get_pressed()
            if keys[K_UP]:
                direction =1
            if keys[K_DOWN]:
                direction =2
            if keys[K_LEFT]:
                direction =3
            if keys[K_RIGHT]:
                direction =4
            
            if keys[K_UP] or keys[K_DOWN] or keys[K_LEFT] or keys[K_RIGHT]:
                if movable(table,direction):
                    move(direction)
                    finishAnimation=False
                else:
                    print("can't move this step!")
                
                
        screen.blit(BackGround.image, BackGround.rect)
        allSprite.draw(screen)
        pg.display.update()
        main_clock.tick(FPS)
    pg.quit()  #關閉程式的程式碼
    
class GameState:
    global Background
    global table
    def __init__(self):
        self.score = 0
        self.direction = 0
        self.running = True
        self.finishAnimation=True
        self.finishMoveAnimation=False
        init_train()
    def reset(self):
        init_train()

    def frame_step(self, input_actions):
        pg.event.pump()
        
        reward = 0.1
        terminal = False
        
        if sum(input_actions) != 1:
            raise ValueError("Multiple input actions!")
            
        self.running = True
            
        while self.running:     
            if not self.finishAnimation:
                for i in range(4):#y        
                    for j in range(4):#x
                        spriteTable[i][j].update(i,j,self.direction,self.finishAnimation,self.finishMoveAnimation)
                        
                AMAF = False
                AAF = False
                self.finishMoveAnimation=False
                for i in range(3,-1,-1):#y
                    for j in range(4):#x
                        if moveTable[i][j] == 0 :
                            AMAF = True
                        else :
                            AMAF = False
                            break
                    if not AMAF:
                        break
                for i in range(3,-1,-1):#y
                    for j in range(4):#x
                        if newItemTable[i][j] == 0 and moveTable[i][j] == 0:
                            AAF = True
                        else :
                            AAF = False
                            break
                    if not AAF:
                        break
                if AMAF:
                    self.finishMoveAnimation=True
                if AAF :
                    self.finishAnimation=True
                
            elif self.finishMoveAnimation and self.finishAnimation:
                for i in range(4):#y        
                    for j in range(4):#x
                        spriteTable[i][j].update(i,j,self.direction,self.finishAnimation,self.finishMoveAnimation)
                self.finishMoveAnimation=False
                if noMoreStep(table):
                    sleep(1)
                    for i in range(4):#y        
                        for j in range(4):#x\
                            table[i][j]=0
                            spriteTable[i][j].update(i,j,self.direction,True,True)
                    #Background.update(os.path.join('2048ByNCC', 'background.png')) #指定background物件
                    showGameOver()
            else :
                if input_actions[0] ==1:
                    self.direction =1 #up
                if input_actions[1] ==1:
                    self.direction =2 #down
                if input_actions[2] ==1:
                    self.direction =3
                if input_actions[3] ==1:
                    self.direction =4
                
                if movable(table,self.direction):
                    move(self.direction)
                    self.finishAnimation=False
                else:
                    print("can't move this step!")
                    self.finishAnimation=True
                    self.finishMoveAnimation=True
                    
                    
            screen.blit(BackGround.image, BackGround.rect)
            allSprite.draw(screen)
            pg.display.update()
            main_clock.tick(FPS)
            
            if self.finishAnimation and self.finishMoveAnimation :
                self.running = False
                
    def frame_step_quick(self, input_actions):
        
        reward = 0.1
        terminal = False
        
        if sum(input_actions) != 1:
            raise ValueError("Multiple input actions!")

        if input_actions[0] ==1:
            self.direction =1 #up
        if input_actions[1] ==1:
            self.direction =2 #down
        if input_actions[2] ==1:
            self.direction =3
        if input_actions[3] ==1:
            self.direction =4
        
        if movable(table,self.direction):
            reward = MOVE(self.direction)
            #showMoving(self.direction)
            RandomNewItem()
            for i in range(4):#y        
                for j in range(4):#x
                    spriteTable[i][j].update(i,j,self.direction,True,True)
        else:
            print("can't move this step!")
            reward = -10
           

        if noMoreStep(table):
            #sleep(1)
            #Background.update(os.path.join('2048ByNCC', 'background.png')) #指定background物件
            
            reward = -128
            terminal = True
            
                    
        screen.blit(BackGround.image, BackGround.rect)
        allSprite.draw(screen)
        if terminal:
            showRestart()
        else:
            zero_count = 16 - np.count_nonzero(table)
            reward = reward * (zero_count/16.0)

        pg.display.update()
        #main_clock.tick(FPS)
        pg.event.pump()
        
        return table, reward, terminal
        

            

        
if __name__ == '__main__':
    main()
