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
    return (c-a*x)/b

def linear(img, p1,p2):
    newImg = []
    for i in range(p1[0]):
        y = equacao_reta((0,0), p1, i)
        newImg.append(y)

    for i in range(p1[0], p2[0]):
        y = equacao_reta(p1, p2, i)
        newImg.append(y)

    for i in range(p2[0], 256):
        y = equacao_reta(p2, (255,255), i)
        newImg.append(y)

    print(newImg)

linear((3,4), (6,7))
print(0-3)