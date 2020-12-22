# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:23:25 2020

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




##################################### 2015, June
    
def computeAgeGroup(age):
    ageGroups = ((0,"Infant",1),(1,"Toddler",3),(3,"Child",13),(13,"Teenager",20),(20,"Adult",float("inf")))
    for i in range(len(ageGroups)):
        if age >= ageGroups[i][0] and age < ageGroups[i][2]:
            ageGroup = ageGroups[i][1]
    return ageGroup

#print(computeAgeGroup(2.9))
#computeAgeGroupTestList = (0,"Infant",1,"Toddler",3,"Child",13,"Teenager",20,"Adult",100,"Adult")
#testInOutList(computeAgeGroupTestList,computeAgeGroup,1,1)


####
def computeShirtSize(chest, waist):
    if chest >=38 and chest <=42 and waist >=30 and waist <=35:
        return "Small"
    elif chest >=40 and chest <=44 and waist >=32 and waist <=37:
        return "Medium"
    elif chest >=42 and chest <=46 and waist >=34 and waist <=39:
        return "Large"
    elif chest >=44 and chest <=48 and waist >=36 and waist <=41:
        return "X-Large"
    elif chest >=46 and chest <=50 and waist >=38 and waist <=43:
        return "XX-Large"
    else: 
        return "Not available"

#print(computeShirtSize(43.5, 34.2))
#computeShirtSizeTestList = (41,33,"Small",43,35,"Medium",45,37,"Large",47,39,"X-Large",49,41,"XX-Large",400,500,"Not available")
#testInOutList(computeShirtSizeTestList,computeShirtSize,2,1)


####
import numpy as np
def acState(state, timeStamp):
    stateTime = np.zeros(5)
    for i in range(len(state)-1):
        if state[i] == 0:
            stateTime[0] = stateTime[0] + timeStamp[i+1]-timeStamp[i]
        elif state[i] == 1:
            stateTime[1] = stateTime[1] + timeStamp[i+1]-timeStamp[i]
        elif state[i] == 2:
            stateTime[2] = stateTime[2] + timeStamp[i+1]-timeStamp[i]
        elif state[i] == 3:
            stateTime[3] = stateTime[3] + timeStamp[i+1]-timeStamp[i]
        elif state[i] == 4:
            stateTime[4] = stateTime[4] + timeStamp[i+1]-timeStamp[i]
    return stateTime

#print(acState(np.array([0, 1, 2, 3, 2, 3, 1]), np.array([0, 486, 849, 1250, 2340, 3560, 7045])))


####
RELATIONS = {'.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E','..-.':'F','--.':'G','....':'H','..':'I','.---':'J','-.-':'K','.-..':'L','--':'M','-.':'N','---':'O','.--.':'P','--.-':'Q','.-.':'R','...':'S','-':'T','..-':'U','...-':'V','.--':'W','-..-':'X','-.--':'Y','--..':'Z', '':' '}     
def morseToText(morseCode):
    return ''.join(RELATIONS[element] for element in morseCode.split(' '))

#print(morseToText("-- --- .-. ... .  -.-. --- -.. ."))


####
import numpy as np
def dhondt(votes, seats):
    seatsAllocated = np.zeros(len(votes))
    quotientArray = np.zeros(len(votes))
    for i in range(seats):
        for j in range(len(votes)):
            quotientArray[j] = votes[j] / (seatsAllocated[j]+1)
            
        candidateArray = quotientArray==max(quotientArray)
        if sum(candidateArray) == 1:
            seatsAllocated = seatsAllocated + candidateArray
        else:
            maxVotes = 0
            maxIndex = 0
            for j in range(len(votes)):
                if candidateArray[j] == True:
                    if votes[j] > maxVotes:
                        maxVotes = votes[j]
                        maxIndex = j
                        
            seatsAllocated[maxIndex] += 1
    return seatsAllocated

#print(dhondt(np.array([340000, 280000, 160000, 60000, 15000]), 7))
#print(dhondt(np.array([340000, 170000, 160000, 60000, 15000]), 7))




############################# 2015, August

def classifyBMI(height, weight):
    bmi = weight / (height*height)
    if bmi<16:
        return "Severely underweight"
    elif bmi<18.5:
        return "Underweight"
    elif bmi<25:
        return "Normal"
    elif bmi<30:
        return "Overweight"
    elif bmi<40:
        return "Obese"
    else:
        return "Severely obese"

#print(classifyBMI(1.85, 88))


####
import numpy as np
def cumulativeStock(transactions):
    stock = np.zeros(len(transactions))
    for i in range(len(transactions)):
        if i == 0:
            stock[i] = transactions[i]
        elif transactions[i]==0:
            stock[i] = 0
        else:
            stock[i] = stock[i-1]+transactions[i]
    return stock

#print(cumulativeStock(np.array([10, -4, -3, 10, -12, 0, 8])))
#print(cumulativeStock(np.array([10, -10, -10, -5, -5, 50, 10])))
#print(cumulativeStock(np.array([100,0,100,0,-5])))


####

def RPNCalculator(commands):
    stack = list()
    commands = commands.split(' ')
    for i in range(len(commands)):     
        if commands[i] == '-':
            stackLast = stack.pop()
            stackSecondLast = stack.pop()
            stack.append(stackSecondLast-stackLast)
        elif commands[i] == 's':
            stackLast = stack.pop()
            stackSecondLast = stack.pop()
            stack.append(stackLast)
            stack.append(stackSecondLast)
        elif commands[i] == '+':
            stackLast = stack.pop()
            stackSecondLast = stack.pop()
            stack.append(stackSecondLast+stackLast)
        else:
            stack.append(int(commands[i]))
    return stack

#print(RPNCalculator("2 10 17 - s"))
#print(RPNCalculator("4 6 - 2 +"))


####
import math
def starPoints(a, b, maxArea):
    n = 3
    while True:
        A = n * a * b * math.sin(math.pi/n)
        if A > maxArea:
            return n-1   
        else:
            n += 1

#print(starPoints(2, 1, 6.1))


####
import numpy as np        
def matrixCleanup(M):
    rowsToKeep = ~np.any(M == 0, axis=1)
    colsToKeep = ~np.any(M == 0, axis=0)
    M = M[rowsToKeep,:]
    M = M[:,colsToKeep]
    return M

#print(matrixCleanup(np.array(([1,2,3,4,5],[6,0,8,9,10],[11,12,13,14,15],[16,0,0,19,20]))))



############################### 2015, December
    
def convertGrade(grade, scale):
    if grade > 10:
        return 'A'
    elif grade == 10:
        return 'B'
    
    if scale == "7-point":
        if grade == 7:
            return 'C'
        elif grade == 4:
            return 'D'
        elif grade == 2:
            return 'E'
        elif grade == 0:
            return 'Fx'
        elif grade == -3:
            return 'F'
        
    if scale == "13-scale":
        if grade == 9 or grade == 8:
            return 'C'
        elif grade == 7:
            return 'D'
        elif grade == 6:
            return 'E'
        elif grade == 5 or grade == 3:
            return 'Fx'
        elif grade == 0:
            return 'F'
    return "Error"

#print(convertGrade(7, '13-scale'))
#convertGradeTestList = (11,'13-scale','A',10,'13-scale','B',9,'13-scale','C',8,'13-scale','C',7,'13-scale','D',6,'13-scale','E',5,'13-scale','Fx',3,'13-scale','Fx',0,'13-scale','F',11,'7-point','A',10,'7-point','B',7,'7-point','C',4,'7-point','D',2,'7-point','E',0,'7-point','Fx',-3,'7-point','F')
#testInOutList(convertGradeTestList,convertGrade,2,1)


####

def hireApplicant(skill, pay):
    x,y = skill, pay
    if x < 5:
        return "No go"
    elif x >= 5 and y > 0.9*x+1:
        return "Too expensive"
    elif x >= 5 and x < 8 and y <= 0.9*x+1:
        return "Hire"
    elif x >= 8 and y > 4 and y <= 0.9*x+1:
        return "Long term contract"
    elif x >= 8 and y <= 4:
        return "Unicorn"
    else:
        return "Critical error"

#print(hireApplicant(8,7))
#print(hireApplicant(10,3))
#print(hireApplicant(3,3))


####
import numpy as np
def averagedB(M):
    rowsToKeep = ~np.any(M > 70, axis=1)
    M = M[rowsToKeep,:]
    return np.mean(M,axis=0)

#print(averagedB(np.array(([25, 25, 30, 40, 40, 40, 45],[50, 55, 55, 65, 65, 70, 75],[75, 70, 70, 70, 50, 50, 55],[25, 30, 35, 40, 50, 55, 60]))))


####

def buildLego(h, l, w):
    volPerBrick = h*l*w
    n = 1
    while True:
        if n*volPerBrick > 1000:
            return n-1
        else:
            n += 1

#print(buildLego(1,8,1.6))


####
import numpy as np
def sudokuCheck(sudokuBoard):
    rowErrors = sum(np.sum(sudokuBoard,axis=1)!=45)
    colErrors = sum(np.sum(sudokuBoard,axis=0)!=45)
    allErrors = rowErrors+colErrors
    for i in range(3):
        ia = i*3
        for j in range(3):
            ja = j*3
            allErrors += np.sum(np.sum(sudokuBoard[ia:ia+3,ja:ja+3],axis=1))!=45
    return allErrors

#print(sudokuCheck(np.array(([5,3,4,6,7,8,9,1,2],[6,7,2,1,9,5,3,5,8],[1,9,8,3,4,2,4,6,7],[8,5,9,7,6,1,4,2,3],[4,2,6,8,5,3,7,9,1],[7,1,3,9,2,4,8,5,6],[9,6,1,5,3,7,2,8,4],[2,8,7,4,1,9,6,3,5],[3,4,5,2,8,6,1,7,9]))))


