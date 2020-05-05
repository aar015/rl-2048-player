'''Code relating to the mask learning agents use to understand the game'''
import numpy
from abc import ABC, abstractmethod


class Mask(ABC):
    '''Abstract class identifing the functions a mask class needs'''

    def __init__(self, boardSize, maxTile):
        '''Init the mask class
        input:
            boardSize: Size of the board
            maxTile: log2 of max tile that can appear on board'''
        self.boardSize = boardSize
        self.maxTile = maxTile

    @abstractmethod
    def getNumTuples(self):
        '''Return the number of tuples used to describe each state'''
        pass

    @abstractmethod
    def getNumTupleStates(self):
        '''Return the number of states any tuple can take'''
        pass

    @abstractmethod
    def getTupleStates(self, state):
        '''Transforms a state into its tuple state representation
        input:
            state: state to transform
        output: array of tuple states corresponding to state'''
        pass

    def getBoardSize(self):
        '''Return the boardSize'''
        return self.boardSize


class Mask_rxcx4(Mask):
    '''Mask that analyzes the states by looking at each row, column, and 4x4 square.'''

    def __init__(self, boardSize=4, maxTile=15):
        '''Init the mask class
        input:
            boardSize: Size of the board
            maxTile: log2 of max tile that can appear on board'''
        super().__init__(boardSize, maxTile)
        self.row_flag = 0
        self.column_flag = 1
        self.square_flag = 2

    def getNumTuples(self):
        '''Return the number of tuples used to describe each state'''
        return 2 * self.boardSize + (self.boardSize-1)**2

    def getNumTupleStates(self):
        '''Return the number of states any tuple can take'''
        return 2 * self.boardSize * (self.maxTile + 1)**self.boardSize + (self.boardSize-1)**2 * (self.maxTile + 1)**4

    def getTupleStates(self, state):
        '''Transforms a state into its tuple state representation
        input:
            state: state to transform
        output: array of tuple states corresponding to state'''
        tupleStates = numpy.zeros(self.getNumTuples(), dtype=numpy.int)
        index = 0
        for tupleType in range(self.square_flag+1):
            if tupleType < self.square_flag:
                maxNum = self.boardSize
            else:
                maxNum = (self.boardSize - 1)**2
            for tupleIndex in range(maxNum):
                tupleStates[index] = self.getTupleState(state, tupleType, tupleIndex)
                index += 1
        return tupleStates

    def getTupleState(self, state, tupleType, tupleIndex):
        '''Get specific tuple state for a state.
        input:
            state: State to transform
            tupleType: Type of tuple given by flag
            tupleIndex: Which tuple of this type we are looking for
        output: specific tuple state of state'''
        num = tupleType * self.boardSize * (self.maxTile + 1)**self.boardSize
        if(tupleType < self.square_flag):
            num += tupleIndex * (self.maxTile + 1)**self.boardSize
            if(tupleType == self.row_flag):
                for col in range(self.boardSize):
                    num += state[tupleIndex][col] * (self.maxTile + 1)**col
            elif(tupleType == self.column_flag):
                for row in range(self.boardSize):
                    num += state[row][tupleIndex] * (self.maxTile + 1)**row
        elif(tupleType == self.square_flag):
            num += tupleIndex * (self.maxTile + 1)**4
            basex = tupleIndex % (self.boardSize - 1)
            basey = int(tupleIndex / (self.boardSize - 1))
            for i in range(2):
                for j in range(2):
                    num += state[basex + i][basey + j] * (self.maxTile + 1)**(i + j*2)
        return num
