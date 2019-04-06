from abc import ABC, abstractmethod


class Mask(ABC):

    def __init__(self, name, boardSize=4, maxTile=15):
        self.name = name
        self.boardSize = boardSize
        self.maxTile = maxTile

    @abstractmethod
    def getNumTuples(self):
        pass

    @abstractmethod
    def getMaxTupleNum(self):
        pass

    @abstractmethod
    def getTupleNums(self, state):
        pass

    def getTag(self):
        return self.name


