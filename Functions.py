import cv2
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt

#cria imagem b&w a partir de uma altura e largura dada
def createImg(h,w):
    array = (np.random.rand(h, w) * 256).astype(np.uint8)
    img = Image.fromarray(array)
    img.save('teste.png')

#aplica o filtro sobel para encontrar os contornos da imagem
def filtro_sobel(img):

    sobel_v = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
    sobel_h = [[-1, -2, -1],[0, 0, 0],[1, 2, 1]]
    dImg = img.shape
    newImg = np.zeros(dImg)

    for i in range(1,dImg[0]-1):
        for j in range(1,dImg[1]-1):
            p = sobel_v[0][0]*img[i-1][j-1]+sobel_v[0][1]*img[i-1][j]+sobel_v[0][2]*img[i-1][j+1]+sobel_v[1][0]*img[i][j-1]+sobel_v[1][1]*img[i][j]+sobel_v[1][2]*img[i][j+1]+sobel_v[2][0]*img[i+1][j-1]+sobel_v[2][1]*img[i+1][j]+sobel_v[2][2]*img[i+1][j+1]
            q = sobel_h[0][0]*img[i-1][j-1]+sobel_h[0][1]*img[i-1][j]+sobel_h[0][2]*img[i-1][j+1]+sobel_h[1][0]*img[i][j-1]+sobel_h[1][1]*img[i][j]+sobel_h[1][2]*img[i][j+1]+sobel_h[2][0]*img[i+1][j-1]+sobel_h[2][1]*img[i+1][j]+sobel_h[2][2]*img[i+1][j+1]
            newImg[i][j] = np.sqrt(p**2+ q**2)    

    return newImg

#retorna a equação da reta dado 2 pontos
def equacao_reta(p1, p2,x):
    
    a = p2[1] - p1[1] 
    b = p1[0] - p2[0] 
    c = a*(p1[0]) + b*(p1[1])  
    
    ''''
    if(b<0):  
        print("The line passing through points P and Q is:", 
              a ,"x ",b ,"y = ",c ,"\n")  
              
    else: 
        print("The line passing through points P and Q is: ", 
              a ,"x + " ,b ,"y = ",c ,"\n")  
    '''
    if b == 0:
        return 0
    return (c-a*x)/b

#calcula o valor de y dado uma equação da reta e um valor x
def linear(img, p1,p2):
    pImg = img.shape
    p0 = (0,0)
    for i in range((pImg[0])):
        for j in range(pImg[1]):
            if img[i][j] < p1[0]:
                img[i][j] = equacao_reta(p0, p1, img[i][j])

            if img[i][j] >= p1[0] and img[i][j] < p2[0]:
                img[i][j] = equacao_reta(p1, p2, img[i][j])

            if img[i][j] >= p2[0] and img[i][j] <= len(img):
                img[i][j] = equacao_reta(p2, pImg, img[i][j])
    return img

#
def limiarizacao(img, limiar):
    dImg = img.shape
    for i in range(dImg[0]):
        for j in range(dImg[1]):
            if img[i][j] <= limiar:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

#Aplica uma transformação logaritma a cada byte da imagem
def transformLoga(imgc,c,h,w):
    for i in range(h-1):
        for j in range(w-1):
            imgc[i][j] = c*np.log10(1 + imgc[i][j])
    return imgc

#Aplica uma potencia de um numero P a cada byte da imagem
def transformPot(imgc,c,h,w, p):
    for i in range(h-1):
        for j in range(w-1):
            imgc[i][j] = c*(imgc[i][j]**p)
    return imgc

#Retorna o negativo de uma imagem
def negative(img):
    return 255 - img

#load image and use
img = cv2.imread("./estrada.jpeg")
dImg = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgp = limiarizacao(img.copy(),160)

plt.subplot(2,1,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(imgp,cmap = 'gray')
plt.title('exemplo'), plt.xticks([]), plt.yticks([])
plt.show()


'''
h, w = img.shape[:2]
cv2.imshow("Pic", img)
imc = img
imp = img
cv2.imshow("pic2", (transformPot(imc, 2, h, w, 0.5)+img))
cv2.imshow("pic3", (transformLoga(imc, 2, h, w)+img))
'''
#...............TESTES..................
#print(h, w)
#momentos = cv2.moments(img)
#momHu = cv2.HuMoments(momentos)
#grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#grayImage = 255 - img
#cv2.imshow("Pic1", grayImage)
#np.log10(np.abs(momHu))
#print(-np.sign(momHu))
#imgc = img
#print("aqui é log\n")
#imp = transformPot(imgc, 2, h, w, 0.7)
#imc = transformaLoga(imgc, 2, h, w)
#imr = imp + img
#cv2.imshow("Pic2", imr)
#cv2.imshow("Pic3", imr)


cv2.waitKey(0)
cv2.destroyAllWindows()




