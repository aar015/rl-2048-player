from learning.algorithms.agent import Agent

class TDMetaAgent(Agent):

    def __init__(self, mask, a=0.0025, g=0.99, e=0.01, name='td_meta'):
        super().__init__(mask, name)
        self.alpha = a
        self.gamma = g
        self.epsilon = e

    
    def learn(self, prevMetaState, prevState, metaState, state, reward): 
        tupleNums = self.mask.getTupleNums(prevMetaState)
        tdError = self.alpha * (reward + self.gamma * self.getVofS(metaState) -
                                self.getVofTupleNums(tupleNums))
        for num in tupleNums:
            self.tuples[num] += tdError

   
