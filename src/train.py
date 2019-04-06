'''Train rl agent'''
from game_logic.play import play
from learning.algorithms.td_meta_agent import TDMetaAgent
from learning.masks.mask_rxcx4 import Mask_rxcx4
import numpy
import csv
import os

def train(agent, numIterations=1000, logFile=None):
    scores = numpy.zeros(numIterations, dtype=numpy.int32)
    for i in range(10):
        scores[i] = play(agent.chooseAction, agent.learn)
    if logFile is not None:
        with open(logFile, mode='w') as log_File:
            writer = csv.writer(log_File, delimiter='\n', lineterminator='', quoting=csv.QUOTE_NONE)
            writer.writerow(scores)


def main():
    mask = Mask_rxcx4()
    agent = TDMetaAgent(mask, 0.0025, 0.95, .001)
    log_File = os.path.join(os.getcwd(),'logs', agent.getTag() + '.csv')
    train(agent, 10, log_File)


if __name__ == "__main__":
    main()
