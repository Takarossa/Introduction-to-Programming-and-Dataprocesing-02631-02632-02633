# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:34:59 2020

@author: Takarossa Shimashi
"""
#Generic test function. Takes a list and function as input
#The number of inputs must be at most 3.
#The number of outputs must be at most 2.
def testInOutList(expList,functionToTest,numInputs,numOutputs):
    failCoun = 0
    setFactor = numInputs+numOutputs
    testNum = int(len(expList)/setFactor)
    for i in range(testNum):
        inputList = expList[i*setFactor:i*setFactor+numInputs]
        #print(inputList)
        outputList = expList[i*setFactor+numInputs:i*setFactor+numInputs+numOutputs]
        #print(outputList)      
        if numInputs==1:
            funcOutput = functionToTest(inputList[0])
            if numOutputs==1:
                result = funcOutput == outputList[0]
            if numOutputs==2:
                result = funcOutput[0] == outputList[0] and funcOutput[1] == outputList[1]
        elif numInputs==2:
            funcOutPut = functionToTest(inputList[0],inputList[1])
            if numOutputs==1:
                result = funcOutPut == outputList[0]
            if numOutputs==2:
                result = funcOutput[0] == outputList[0] and funcOutput[1] == outputList[1]
        elif numInputs==3:
            funcOutPut = functionToTest(inputList[0],inputList[1],inputList[2])
            if numOutputs==1:
                result = funcOutPut == outputList[0]
            if numOutputs==2:
                result = funcOutput[0] == outputList[0] and funcOutput[1] == outputList[1]
        if result == False:
            print("Failure in set: "+str(i)+" with input(s) "+str(inputList)+"\n Expected output: "+str(outputList)+" but function returned "+str(funcOutput))
            failCoun +=1
    return  print("\nTest executed. "+str(failCoun)+" failures out of "+str(testNum))



######################### 2016 January
def bookPages(pagesContent):
    if pagesContent % 4 == 0:
        return pagesContent
    else:
        return 4 - pagesContent % 4 + pagesContent

#print(bookPages(17))
#print(bookPages(21))
#print(bookPages(49))


####
def loadBalance(runtime):
    totSum = sum(runtime)
    diff = float("inf")
    for i in range(len(runtime)-1):
        sum1 = sum(runtime[0:i+1])
        sum2 = totSum-sum1
        if diff > abs(sum1-sum2):
            diff = abs(sum1-sum2)
            k = i+1
    return  k

#print(loadBalance(np.array([5, 2.5, 17, 1.5, 22, 3.5])))


####

def nearestColor(r, g, b):
    COLORS = (("White",100,100,100),("Grey",50,50,50),("Black",0,0,0),("Red",100,0,0),("Maroon",50,0,0),("Yellow",100,100,0),("Olive",50,50,0),("Lime",0,100,0),("Green",0,50,0),("Aqua",0,100,100),("Teal",0,50,50),("Blue",0,0,100),("Navy",0,0,50),("Fuchsia",100,0,100),("Purple",50,0,50))
    closest = float("inf")
    for i in range(len(COLORS)):
        distance = max(abs(r-COLORS[i][1]),abs(g-COLORS[i][2]),abs(b-COLORS[i][3]))
        if distance < closest:
            closest = distance
            color = COLORS[i][0]
    return color

#print(nearestColor(75, 0, 0))


####
def climbCategorization(distance, grade):
    D,G = distance, grade
    if G > 8:
        if D < 2:
            return "Beginner"
        elif D < 4:
            return "Easy"
        elif D < 8:
            return "Medium"
        elif D < 12:
            return "Difficult"
        elif D >= 12:
            return "Extreme"
    elif D > 16:
        if G < 1:
            return "Beginner"
        elif G < 2:
            return "Easy"
        elif G < 4:
            return "Medium"
        elif G < 6:
            return "Difficult"
        elif G >= 6:
            return "Extreme"
    else:
        DG = D*G
        if DG < 16:
            return "Beginner"
        elif DG < 32:
            return "Easy"
        elif DG < 64:
            return "Medium"
        elif DG < 96:
            return "Difficult"
        elif DG >=96:
            return "Extreme"
        else:
            return "Critical Error"

#print(climbCategorization(8, 6))


####
RELATIONS = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
def romanToValue(roman):
    roman = [RELATIONS[elements] for elements in list(roman)]
    runTotal = 0
    maxSymbol = 0
    roman.reverse()
    for i in range(len(roman)):
        if maxSymbol <= roman[i]:
            maxSymbol = roman[i]
            runTotal += roman[i]
        else:
            runTotal -= roman[i]
    return runTotal  
        
#print(romanToValue("XCIV"))





############################ 2016, February(a)
import numpy as np
def weeklyAverage(dayTemp):
    return np.mean(np.sort(dayTemp)[2:5])

#import numpy as np
#print(weeklyAverage(np.array([17.3, 18.2, 31.2, 14.2, -12.5, 16.5, 14.2])))


####
RELATIONS = {"M":0,"E":1,"C":2,"S":3,"F":4,"L":5}
def carsAvailable(fleet, category):
    category = RELATIONS.get(category[0],-1)
    if category == -1:
        return -1
    else:
        return fleet[category]

#import numpy as np
#print(carsAvailable(np.array([5, 0, 17, 13, 11, 1]), "Lux"))


####
import numpy as np
def speedOfLight(f, wavelengths):
    return np.average(wavelengths)*f

#print(speedOfLight(2.45e9, np.array([0.122, 0.125, 0.123])))


####
import numpy as np
import math
def similarityMatrix(x, y, delta):
    S = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            S[i][j] = math.exp(-delta * math.pow((x[i] - y[j]),2))
    return S

#print(similarityMatrix(np.array([1.1, 1.2]), np.array([1.3, 1.4, 1.5]), 2))


####
import numpy as np
def coinReturn(coinsInMachine, amount):
    paid = 0
    nextCoin = 0
    coinsInMachine = np.sort(coinsInMachine)
    amountToPay = amount
    coinsToReturn = list()
    while True:
        if amountToPay < coinsInMachine[0]/2:
            return np.array(coinsToReturn)
        for i in range(len(coinsInMachine)):
            if amountToPay >= coinsInMachine[i]:
                nextCoin = coinsInMachine[i]
        coinsToReturn.append(nextCoin)
        amountToPay -= nextCoin

#print(coinReturn(np.array([0.5, 1, 2, 5, 10, 20]), 34.60))


################################ 2016, May
import math
def eseries(x, N):
    f = 0
    for i in range(N):
        f += math.pow(x,i)/math.factorial(i)
    return f

#print(eseries(1.23, 5))


####
RELATIONS = {'A':2,'B':2,'C':2,'D':3,'E':3,'F':3,'G':4,'H':4,'I':4,'J':5,'K':5,'L':5,'M':6,'N':6,'O':6,'P':7,'Q':7,'R':7,'S':7,'T':8,'U':8,'V':8,'W':9,'X':9,'Y':9,'Z':9}
def alphaToPhone(alpha):
    return ''.join([str(RELATIONS.get(elements,elements)) for elements in list(alpha)])

#print(alphaToPhone("4525DTU1"))


####
import numpy as np
RELATIONS = {'a':1,'e':1,'i':1,'o':1,'u':1}
def genderGuess(name):
    if RELATIONS.get(name[-1],0):
        if sum(np.array(list(name))=='f')>=2:
            return 0.35
        else:
            return 0.87
    else:
        if name[0] == 'k':
            return 0.69
        else:
            return 0.25

#print(genderGuess("affonso"))
#print(genderGuess("afks"))
#print(genderGuess("kfks"))
#print(genderGuess("allonso"))


####
import math            
def birthday(n):
    return 1 - math.exp(math.lgamma(365+1) - math.lgamma(365-n+1) - n*math.log(365))

#print(birthday(23))


####
import numpy as np
def matrixSearch(A, x):
    height = len(A[:,0])
    width = len(A[0,:])
    i = 0
    j = width-1
    while True:
        if A[i,j] == x:
            return [i+1,j+1]
        elif A[i,j] > x:
            j -= 1
        elif A[i,j] < x:
            i += 1
        if j < 0 or i == height:
            return [0,0]

#print(matrixSearch(np.array([[1,2,6,10],[3,7,7,13],[7,9,11,14]]), 7))
#print(matrixSearch(np.array([[1,2,6,10],[3,7,7,13],[7,9,11,14]]), 8))
#print(matrixSearch(np.array([[1,2,6,10],[3,7,7,13],[7,9,11,14]]), 10))
#print(matrixSearch(np.array([[1,2,6,10],[3,7,7,13],[7,9,11,14]]), 9))


#################################################### 2016, June
import math
def normalWeight(h):
    return [math.ceil(18.5*h*h),math.floor(25*h*h)]

#print(normalWeight(1.73))


####

def danceClass(v):
    male = v == 1
    female = v == 0
    if len(v) < 10 or len(v) > 30 or abs(sum(male)-sum(female)) >3:
        return "invalid"
    else:
        return "valid"

#print(danceClass(np.array([1,1,1,0,0,1,0,1,1,1,0])))
#print(danceClass(np.array([1,1,1,0,1,0])))
#print(danceClass(np.array([1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,0])))
#print(danceClass(np.array([1,1,1,1,1,1,1,1,1,1,1])))
#print(danceClass(np.array([0,0,0,0,0,0,0,0,0,0,0])))


####
import numpy as np
def survivalTemperature(M,g):
    survivalMatrix = np.zeros((len(M),len(g)))
    if np.any(M < 50) or np.any(M > 500) or np.any(g < 0.04) or np.any(g > 0.45):
        return "RangeError"
    for i in range(len(M)):
        for j in range(len(g)):
            survivalMatrix[i,j] = 36 - ( ( 0.9*M[i] - 12 )*( g[j]+0.95 ) ) / ( 27.8*g[j] )
    return survivalMatrix
        
#print(survivalTemperature(np.array([50,200,300]),np.array([0.20,0.14])))


####
RELATIONS = {"brbr":19,"brbl":0,"blbr":0,"brgr":38,"grbr":38,"blbl":1,"blgr":50,"grbl":50,"grgr":75}
def greenEyes(p1,p2):
    return RELATIONS.get(''.join(p1[0:2]+p2[0:2]))

#print(greenEyes('green','brown'))

####

def bestBuy(p,m):
    money = m
    n = 0
    for i in range(len(p)):
        if money >= p[i]:
            money -= p[i]
            n += 1
        else:
            return n
    return n
    
import numpy as np
#print(bestBuy(np.array([5,4,6,2,9,1,1,4]),16))


########################################### 2016, August
import numpy as np
import math
def confidence(x):
    m = np.mean(x)
    partSum = 0
    for i in range(len(x)):
        partSum += math.pow(x[i]-m,2)
    s = math.sqrt(partSum/(len(x)-1))
    return [m-2*(s/math.sqrt(len(x))),m+2*(s/math.sqrt(len(x)))]

#print(confidence(np.array([1, 2, 4, 3, 1])))


####
import math
C_DICTIONARY = {1:6,2:2,3:2,4:5,5:0,6:3,7:5,8:1,9:4,10:6,11:2,12:4}
W_DICTIONARY = {0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
def weekday(d, m, y):
    return W_DICTIONARY.get((d+C_DICTIONARY.get(m)+y+math.floor(y/4))%7)

#print(weekday(21, 8, 16))
#print(weekday(21, 8, 17))
#print(weekday(21, 8, 18))


####
import numpy as np
def symmetrize(x):
    height = len(x[:,0])
    width = len(x[1,:])
    y = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if i == j:
                y[i,j] = x[i,j]
            else:
                y[i,j] = x[i,j]+x[j,i]
    return y

#print(symmetrize(np.array([[1.2, 2.3, 3.4],[4.5, 5.6, 6.7],[7.8, 8.9, 10.0]])))


####
import math
def voldif(R, n):
    top = math.pow(math.pi,n/2)
    bot = math.gamma(n/2+1)
    volSphere = ( top / bot ) * math.pow(R,n)
    volCube = math.pow(2*R,n)
    return volCube-volSphere

#print(voldif(5, 2))


####
import numpy as np
def stringcompare(string1, string2):
    s1 = np.unique(list(string1))
    s2 = np.unique(list(string2))
    s1Same,s1Dif,s2Same,s2Dif = 0,0,0,0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                s1Same +=1
                j = len(s2)
    for i in range(len(s2)):
        for j in range(len(s1)):
            if s2[i] == s1[j]:
                s2Same +=1
                j = len(s1)
    s1Dif = len(s1)-s1Same
    s2Dif = len(s2)-s2Same
    return s1Dif+s2Dif

#print(stringcompare("aardvark", "artwork"))


################### 2016, December
import numpy as np
def cvrchecksum(cvr):
    cvrArr = np.array([int(elements) for elements in list(str(cvr))])
    weight = np.array([2,7,6,5,4,3,2])
    return 11-sum(cvrArr*weight)%11

#print(cvrchecksum(3006094))


####
    
def nextleapyear(y):
    rem = y%4
    nextCand = y+(4-rem)*(rem!=0)+4*(rem==0)
    while True:
        if nextCand%400 == 0:
            return nextCand
        elif nextCand%100 == 0:
            nextCand += 4
        else:
            return nextCand

#print(nextleapyear(1896))
#print(nextleapyear(1899))
#print(nextleapyear(1999))
#print(nextleapyear(2004))


####

def capitalize(text):
    textList = list(text)
    textList[0] = textList[0].upper()
    for i in range(len(textList)-2):
        if textList[i] == '!' or textList[i] == '?' or textList[i] == '.':
            if textList[i+1] != ' ':
                textList[i+1] = textList[i+1].upper()
            elif textList[i+1] == ' ':
                textList[i+2] = textList[i+2].upper()
    return ''.join(textList)

#print(capitalize("hello! how are you? please remember capitalization. EVERY time."))

 
####
import numpy as np 
def submatrix(M, r, c):
    height = len(M[:,0])
    width  = len(M[0,:])
    removeRow, removeCol = True, True
    if r < 1 or r > height:
        removeRow = False
    if c < 1 or c > width:
        removeCol = False
    r -=1
    c -=1
    if removeRow:
        M = np.delete(M,r,0)
    if removeCol:
        M = np.delete(M,c,1)
    return M

#print(submatrix(np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12])), 3, 2))
#print(submatrix(np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12])), -1, 2))
#print(submatrix(np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12])), -1, -1))
#print(submatrix(np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12])), 3, -1))


#### [Difficult]
import numpy as np
def approximatepi(N):
    #Figure out how many points are in the system
    pointNum = N**2
    #The points have the same coords over and over, but combination of x and y change
    #Calculate the possible values an x or y coord can have
    cords = list()
    for pointCoun in range(N):
        cords.append(1/(N*2)+pointCoun/N)
    #Calculate the distance of each possible combination of cords
    allPoints = np.zeros(pointNum)
    for i in range(N):
        for j in range(N):
            allPoints[i*N+j] = np.sqrt(cords[i]**2 + cords[j]**2)
    #Find the amount of points within the circle
    K = len(allPoints[np.where(allPoints < 1)[0]])
    return (K/N**2)*4

#print(approximatepi(2))











