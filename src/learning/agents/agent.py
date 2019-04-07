import numpy
import random
import pickle
from game_logic.game import Game
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, mask, name):
        self.mask = mask
        self.name = name
        self.tuples = numpy.zeros(self.mask.getMaxTupleNum(), dtype=float)

    @abstractmethod
    def learn(self, prevMetaState, prevState, metaState, state, reward):
        pass

    def chooseAction(self, state, actions):
        if (random.random() < self.epsilon):
            return actions[random.randint(0, numpy.size(actions) - 1)]
        max_actions = numpy.zeros(4, dtype=numpy.int)
        tempGame = Game(numpy.copy(state))
        reward = tempGame.do_action(actions[0])
        maxValue = reward + self.getVofS(tempGame.state())
        max_actions[0] = actions[0]
        numMax = 1
        for i in range(numpy.size(actions) - 1):
            tempGame = Game(numpy.copy(state))
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

    def getVofS(self, state):
        return self.getVofTupleNums(self.mask.getTupleNums(state))

    def getVofTupleNums(self, tupleNums):
        return sum([self.tuples[num] for num in tupleNums])

    def getTag(self):
        return self.name + '_' + self.mask.getTag()

    def save(self, fileName):
        pickleFile = open(fileName, 'wb')
        pickle.dump(self.tuples, pickleFile)
        pickleFile.close()

    def load(self, fileName):
        pickleFile = open(fileName, 'rb')
        self.tuples = pickle.load(pickleFile)
        pickleFile.close()
