import numpy as np
import random
from game_logic.game import Game, LENGTH

NUM_TILES = 16
ROW = 0
COLUMN = 1
SQUARE = 2


class TDMetaAgent(object):
    def __init__(self, a=0.0025, g=0.99, e=0.01, name='td_meta'):
        self.tuples = np.zeros(NUM_TILES**6, dtype=float)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        self.name = name

    def tdChooseAction(self, state, actions):
        if (random.random() < self.epsilon):
            return actions[random.randint(0, np.size(actions) - 1)]
        max_actions = np.zeros(4, dtype=np.int)
        tempGame = Game(np.copy(state))
        reward = tempGame.do_action(actions[0])
        maxValue = reward + self.getVofS(tempGame.state())
        max_actions[0] = actions[0]
        numMax = 1
        for i in range(np.size(actions) - 1):
            tempGame = Game(np.copy(state))
            action = actions[i + 1]
            reward = tempGame.do_action(action)
            value = reward + self.getVofS(tempGame.state())
            if(value > maxValue):
                maxValue = value
                max_actions[0] = action
                numMax = 1
            elif(value == maxValue):
                max_actions[numMax] = action
                numMax += 1
        return max_actions[random.randint(0, numMax - 1)]

    def tdLearn(self, prevMetaState, prevState, metaState, state, reward): 
        tupleNums = self.getTupleNums(prevMetaState)
        tdError = self.alpha * (reward + self.gamma * self.getVofS(metaState) - self.getVofTupleNums(tupleNums))
        for num in tupleNums:
            self.tuples[num] += tdError

    def getVofS(self, state):
        return self.getVofTupleNums(self.getTupleNums(state))

    def getVofTupleNums(self, tupleNums):
        return sum([self.tuples[num] for num in tupleNums])

    def getTupleNums(self, state):
        tupleNum = np.zeros(17, dtype=np.int)
        index = 0
        for type in range(3):
            maxNum = LENGTH if type <= COLUMN else (LENGTH - 1)**2
            for num in range(maxNum):
                tupleNum[index] = self.stateToTupleNum(state, type, num)
                index += 1
        return tupleNum

    def stateToTupleNum(self, state, tupleType, tupleNum):
        hexString = ''.join(['{:x}'.format(tupleType), '{:x}'.format(tupleNum)])
        if(tupleType == ROW):
            hexString += ''.join(['{:x}'.format(state[tupleNum][col]) for col in range(LENGTH)])
        elif(tupleType == COLUMN):
            hexString += ''.join(['{:x}'.format(state[row][tupleNum]) for row in range(LENGTH)])
        elif(tupleType == SQUARE):
            basex = tupleNum % (LENGTH - 1)
            basey = int(tupleNum / (LENGTH - 1))
            hexString += ''.join(['{:x}'.format(state[basex + i][basey + j])
                                  for i in range(2) for j in range(2)])
        return int(hexString, base=16)

    def getTag(self):
        return self.name
