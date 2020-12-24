# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 08:55:46 2020

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


##################################### 2017, January
import math
def trianglearea(a, b, c):
    if a+b<=c or a+c<=b or b+c<=a:
        return 0
    p = (1/2)*(a+b+c)
    return math.sqrt(p*(p-a)*(p-b)*(p-c))

#print(trianglearea(4.5, 6, 7.5))
#print(trianglearea(1,2,3))
#print(trianglearea(1,3,2))
#print(trianglearea(3,2,1))


####
import numpy as np
def polygonarea(x, y):
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    detSum = 0
    for i in range(len(x)-1):
        detSum += x[i]*y[i+1]-y[i]*x[i+1]
    return (1/2)*detSum

#print(polygonarea(np.array([1, 3, 3, 4, 7, 6, 1]), np.array([1, 1, 2, 3, 3, 5, 5])))


####
import numpy as np
import math
def fitpolynomial(x, y, n):
    k = len(x)
    coefMatrix = np.zeros((k,n+1))
    for i in range(k):
        for j in range(n+1):
            coefMatrix[i,j] = math.pow(x[i],j)
    wTw = np.matmul( np.transpose(coefMatrix) , coefMatrix )
    wInv = np.linalg.inv(wTw)
    wInvwTran = np.matmul( wInv , np.transpose(coefMatrix) )
    return np.matmul(wInvwTran , y)

#print(fitpolynomial(np.array([-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]), np.array([2.1, 0.6, -1.5, -1.6, -1.9, -2, -1.1, 0.3, 2.7]), 3))


#### [difficult]
import numpy as np
def predictability(x):
    q = np.zeros(5)
    for i in range(len(x)):
        q[x[i]-1] += 1
    
    if len(x) == 1:
        return q[x[0]-1]/len(x)
    
    R = np.zeros((5,5))
    for i in range(len(x)-1):
        R[x[i]-1,x[i+1]-1] += 1
        
    metaSum, N = 0, 1
    while N < len(x):
        metaSum += R[x[N-1]-1,x[N]-1] / q[x[N-1]-1]
        N += 1
        
    return (1/N)*(q[0]/N+ metaSum)

#print(predictability(np.array([1, 5, 5, 3, 5, 1, 3, 5, 4, 5, 4, 1, 5, 5, 4])))
#print(predictability(np.array([5])))


####
vowelSet = {'a','e','i','o','u','y'}
def syllables(word):
    charList = list(word)
    vowelFound = False
    syl = 0
    for char in charList:
        if char in vowelSet:
            vowelFound = True
        elif vowelFound:
            if char not in vowelSet:
                syl += 1
                vowelFound = False
        else:  
            syl +=0
    return syl
    
#print(syllables("cheesecake"))


################################################# 2017, May
import numpy as np
def salePrice(prices):
    prices = np.sort(prices)[::-1]
    for i in range(int(len(prices)/2)):
        prices[i*2+1] = 0
    return sum(prices)

#print(salePrice(np.array([9.95, 129.50, 9.95, 40.00, 17.75])))


####
    
def ISBNCheckDigit(isbn):
    isbnSum = 0
    for i in range(len(isbn)):
        isbnSum += isbn[i]*(i+1)
    if isbnSum%11 == 10:
        return 'X'
    else:
        return isbnSum%11

#print(ISBNCheckDigit(np.array([1, 4, 9, 1, 9, 3, 9, 3, 6])))



####
import numpy as np
import math
def partialCorrelation(X):
    N = len(X[:,0])
    K = len(X[0,:])
    C = np.linalg.inv( 1/N * np.matmul ( np.transpose ( X ) , X ) )
    P = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i == j:
                P[i,j] = 1
            else:
                P[i,j] = ( -C[i,j] ) / ( math.sqrt( C[i,i] * C[j,j] ) )
    return P

#print(partialCorrelation(np.array([[-.1, .4, -1],[3.4, 1.8, -1],[-1.9, -1.5, .7],[.8, 1.6, 4.4],[.7, 1, 3]])))


####

def zeroCrossing(signal):
    count = 0
    signal = signal[signal != 0.0]
    for i in range(len(signal)):
        if i == 0:
            count += 0
        elif signal[i-1] > 0 and signal[i] < 0:
            count += 1
        elif signal[i-1] < 0 and signal[i] > 0:
            count += 1
    return count

#print(zeroCrossing(np.array([0.6, 0.2, -0.1, 0.1, 0.3, 0.3, -0.1, -0.8, -0.9, -0.5, 0.4, 1.0, 0.8, 0.2, -0.4, -0.5, -0.1, 0.1, -0.1, -0.5])))
#print(zeroCrossing(np.array([0.6,0.0,-0.1])))


####
import numpy as np
def pairwiseRank(P):
    K = len(P)
    omega = np.zeros(K)
    N = np.zeros((P.shape))
    for i in range(len(P)):    
        omega[i] = sum(P[i,:])
        for j in range(len(P[0])):
            N[i,j] = P[i,j]+P[j,i]  

    r = np.ones(K)   
    rho = np.zeros(K)
    divisor = 0
    for _ in range(100):
        for counI in range(K):
            for counJ in range(K):
                divisor +=  N[counI,counJ]/(r[counI]+r[counJ])
               
            rho[counI] = omega[counI] / divisor 
            divisor = 0
            
        for counI in range(K):
            r[counI] = rho[counI] / sum(rho)  
           
    return r


#print(pairwiseRank(np.array([[0, 1, 0, 1],[5, 0, 1, 6],[8, 8, 0, 5],[7, 3, 1, 0]])))


##################################### 2017, August
    
def rgb2hue(R, G, B):
    delta = max(R,G,B)-min(R,G,B)
    estimate = 0
    if R>=G and R>=B:
        estimate = 60*(G-B)/delta
    elif G>=R and G>=B:
        estimate = 120 + 60*(B-R)/delta
    elif B>=R and B>=G:
        estimate = 240 + 60*(R-G)/delta
    if estimate < 0:
        estimate += 360
    return estimate

#print(rgb2hue(0.6, 0.2, 0.3))
#print(rgb2hue(0.5, 0.1, 0.6))


####
RELATIONS = { '0':'0', '1':'2', '2':'4', '3':'6', '4':'8', '5':'1', '6':'3', '7':'5', '8':'7', '9':'9'}
def cardValidation(cardnumber):
    cardList = list(cardnumber)
    for i in range(int(len(cardList)/2)):
        cardList[i*2] = RELATIONS.get(cardList[i*2])
        
    cardArray = [int(i) for i in cardList]
    return sum(cardArray)

#print(cardValidation("4024007156748096"))


####
import numpy as np
import math
def polygonPerimeter(x, y):
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    P = 0
    for i in range(len(x)-1):
        P += math.sqrt( math.pow( x[i]-x[i+1] , 2 ) + math.pow( y[i]-y[i+1] , 2 )  )
    return P

#print(polygonPerimeter(np.array([1, 3, 3, 4, 7, 6, 1]), np.array([1, 1, 2, 3, 3, 5, 5])))


####
import math
import numpy as np
def rotateScale(coordinates, center, theta, scale):
    trigMatrix = np.reshape(np.array([math.cos(theta),-math.sin(theta),math.sin(theta),math.cos(theta)]),(2,2))
    newCord = np.zeros((2,len(coordinates[0,:])))
    for i in range(len(coordinates[0,:])):
        secMatrix = np.reshape(np.array([coordinates[0,i]-center[0], coordinates[1,i]-center[1]]),(2,1))
        newCord[0,i] = (np.matmul (trigMatrix,secMatrix) * scale)[0]+center[0]
        newCord[1,i] = (np.matmul (trigMatrix,secMatrix) * scale)[1]+center[1]
    return newCord

#print(rotateScale(np.array([[1,3,3,4,7,6,1],[1,1,2,3,3,5,5]]), np.array([3,1]), -math.pi/3, 2))


#### [Impossible to do during an exam. 
#This is called a knapsack problem and is not introductory, anyway here is my solution]
#To solve this, use binary numbers to compute every possible option
import numpy as np
def costliestCar(maxPrice):
    trimArr = np.array([0,22000,44000])
    trimStringArr = np.array(["Access, ","Comfort, ","Sport, "])
    extraArr = np.array([4000,8000,13000,7000])
    extraStringArr = np.array(["Cruise control, ","Air conditioning, ","Alloy wheels, ","Chrome spoiler, "])
    baseCost = 159000
    closeCost = 0
    #    For one option, range is len
    for i in range(len(trimArr)):
        binArr = np.zeros(len(trimArr)+len(extraArr))
        binArr[i] = 1
#   For multi options of max 1, the range is 2^len
        combinations = 1<<len(extraArr)
        for j in range(combinations):
            binStr = bin(j)[2:]
            binLen = len(binStr)
            for k in range(binLen):
                binArr[len(binArr)-binLen+k] = int(binStr[k])
            
            trimString = ""
            trimCost = 0
            for k in range(len(trimArr)):
                trimCost = trimCost + binArr[k]*trimArr[k]
                trimString = trimString + trimStringArr[k]*int((binArr[k]==1))
                
            extraString = ""
            extraCost = 0
            for k in range(len(extraArr)):
                binIndex = len(trimArr)+k
                extraCost = extraCost + extraArr[k]*binArr[binIndex]
                extraString = extraString + extraStringArr[k]*int((binArr[binIndex]==1))
            
           
            totalCost =  baseCost + trimCost + extraCost
            if  totalCost > closeCost and maxPrice >= totalCost:
                closeCost = totalCost
                closeString = (trimString+ extraString)[0:-2]
            
    return closeString    

print(costliestCar(170500))




































