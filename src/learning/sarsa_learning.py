import numpy
import random
from learning.agent import Agent

class SARSAAgent(Agent):

    def __init__(self, mask, a=0.0025, g=0.99, e=0.01, name='SARSA'):
        super().__init__(mask, name)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        self.tuples = numpy.zeros((self.mask.getMaxTupleNum(), 4), dtype=float)

    def learn(self, prevState, action, state, reward): 
        tupleNums = self.mask.getTupleNums(prevState)
        next_action = self.chooseAction(state, [0, 1, 2, 3])
        sarsaError = self.alpha*(reward+self.gamma*self.getVofSA(state, next_action)-self.getVofSA(prevState, action))
        for num in tupleNums:
            self.tuples[num, action] += sarsaError
            if self.tuples[num, action] < 0:
                self.tuples[num, action] = 0
        
    def chooseAction(self, state, actions):
        if (random.random() < self.epsilon):
            return actions[random.randint(0, numpy.size(actions) - 1)]
        max_actions = numpy.zeros(4, dtype=numpy.int)
        max_actions[0] = actions[0]
        maxValue = self.getVofSA(state, actions[0])
        numMax = 1
        for i in range(numpy.size(actions) - 1):
            action = actions[i + 1]
            value = self.getVofSA(state, action)
            if(value > maxValue):
                maxValue = value
                max_actions[0] = action
                numMax = 1
            elif(value == maxValue):
                max_actions[numMax] = action
                numMax += 1
        return max_actions[random.randint(0, numMax - 1)]

    def getVofSA(self, state, action):
       return self.getVofTupleNumsA(self.mask.getTupleNums(state), action) 

    def getVofTupleNumsA(self, tupleNums, action):
        return sum([self.tuples[num, action] for num in tupleNums])

    def getTag(self):
        tag = super().getTag()
        tag += '_a'+str(self.alpha).split('.')[1]
        tag += 'e'+str(self.epsilon).split('.')[1]
        tag += 'g'+str(self.gamma).split('.')[1]
        return tag
