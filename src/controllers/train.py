'''Train rl agent'''
from game_logic.play import play
import numpy
import csv


def train(agent, numIterations=1000, logFile=None):
    scores = numpy.zeros(numIterations, dtype=numpy.int32)
    for i in range(10):
        scores[i] = play(agent.chooseAction, agent.learn)
    if logFile is not None:
        with open(logFile, mode='w') as log_File:
            writer = csv.writer(log_File, delimiter='\n', lineterminator='', quoting=csv.QUOTE_NONE)
            writer.writerow(scores)
