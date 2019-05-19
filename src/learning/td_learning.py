import numpy
import random
from game_logic.game import Game
from learning.agent import Agent

class TD0Agent(Agent):

    def __init__(self, mask, a=0.0025, g=0.99, e=0.01, name='td0'):
        super().__init__(mask, name)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        self.tuples = numpy.zeros(self.mask.getMaxTupleNum(), dtype=float)

    def learn(self, prevState, action, state, reward): 
        tupleNums = self.mask.getTupleNums(prevState)
        tdError = self.alpha * (reward + self.gamma *
                                self.getVofS(state) - 
                                self.getVofTupleNums(tupleNums))
        for num in tupleNums:
            self.tuples[num] += tdError
            if self.tuples[num] < 0:
                self.tuples[num] = 0

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
        tag = super().getTag()
        tag += '_a'+str(self.alpha).split('.')[1]
        tag += 'e'+str(self.epsilon).split('.')[1]
        tag += 'g'+str(self.gamma).split('.')[1]
        return tag
