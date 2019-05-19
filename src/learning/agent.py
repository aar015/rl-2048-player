import numpy
import pickle
import csv
import cv2
import imageio
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from game_logic.game import Game


class Agent(ABC):

    def __init__(self, mask, name):
        self.mask = mask
        self.name = name

    @abstractmethod
    def learn(self,prevState, state, reward):
        pass

    @abstractmethod
    def chooseAction(self, state, actions):
        pass

    def play(self, verbose=False):
        """Agent plays a single game
           Based on the code from georgwiese:https://github.com/georgwiese/2048-rl
        Args:
            verbose: If verbose is true also return game states and scores
        Returns:
            final score and log if verbose is set to true"""
        game = Game()
        # record previous state to update learning algorithm
        prevState = game.state().copy()
        # whether or not game has reached a gameover state
        game_over = game.game_over()
        # If verbose record a log of game states and scores
        if verbose:
            log = []
            log.append([game.score(), game.state().copy()])
        while not game_over:
            # Choose next action
            next_action = self.chooseAction(game.state().copy(),
                                            game.available_actions())
            # Perform action and recieve a reward
            reward = game.do_action(next_action)
            # Add random tile here for full states
            # game.add_random_tile()
            # Update learning algorithm
            self.learn(prevState, next_action, game.state().copy(), reward)
            # Update prevState
            prevState = game.state().copy()
            # Add random tile here for metastates
            game.add_random_tile()
            # If verbose add new state and score to log
            if verbose:
                log.append([game.score(), game.state().copy()])
            # Check if game is over
            game_over = game.game_over()
        # If verbose return final score and log
        if verbose:
            return game.score(), log
        # Else return just final score of game
        else:
            return game.score()

    def train(self, numIterations=1000, logFile=None, _mode='w'):
        scores = numpy.zeros(numIterations, dtype=numpy.int32)
        for i in range(numIterations):
            scores[i] = self.play(verbose=False)
        if logFile is not None:
            with open(logFile, mode=_mode) as log_File:
                writer = csv.writer(log_File, delimiter='\n',
                                    lineterminator='\n', quoting=csv.QUOTE_NONE)
                writer.writerow(scores)

    def makeGif(self, gif_file, num_trials=10, board_size=4, graphic_size=750,
                top_margin=40, seperator_width=12, end_pause=50):
        bestFinalScore = 0
        for i in range(num_trials):
            finalScore, log = self.play(verbose=True) 
            if finalScore > bestFinalScore:
                bestFinalScore = finalScore
                bestLog = log
        with imageio.get_writer(gif_file, mode='I') as writer:
            for i in range(numpy.shape(bestLog)[0]):
                img=self.makeImage(bestLog[i][0], bestLog[i][1], board_size,
                                   graphic_size,top_margin, seperator_width)
                writer.append_data(img)
                if i == numpy.shape(bestLog)[0]-1:
                    for i in range(end_pause):
                        writer.append_data(img)

    def makeImage(self, score, state, board_size=4, graphic_size=750,
                  top_margin=40, seperator_width=12):
        img = numpy.full((graphic_size + top_margin, graphic_size, 3), 255,
                         numpy.uint8)
        # Define colors
        background_color = (146, 135, 125)
        color = {0:(158, 148, 138), 1:(238, 228, 218), 2:(237, 224, 200),
                 3:(242, 177, 121), 4:(245, 149, 99), 5:(246, 124, 95), 
                 6:(246, 94, 59), 7:(237, 207, 114), 8:(237, 204, 97), 
                 9:(237, 200, 80), 10:(237, 197, 63), 11:(237, 197, 63), 
                 12:(62, 237, 193), 13:(62, 237, 193), 14:(62,64,237), 
                 15:(140,62,237)}
        #Set font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Define spacing of tiles
        spacing = int((graphic_size-seperator_width)/board_size)
        # Write score at top of screen
        text = 'The score is ' + str(score)
        textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
        cv2.putText(img,text,(int((graphic_size-textsize[0])/2),
                              int((3*top_margin/4+textsize[1])/2)),
                    font,0.5,(0,0,0),1,cv2.LINE_AA)
        # Draw squares
        for i in range(4):
            for k in range(4):
                cv2.rectangle(img,
                              (int(seperator_width/2)+k*spacing,
                               int(top_margin+seperator_width/2)+i*spacing),
                              (int(seperator_width/2)+(k+1)*spacing,
                               int(top_margin+seperator_width/2)+(i+1)*spacing),
                              color[state[i][k]], -1)
                if state[i][k] == 0:
                    text = ''
                else:
                    text = str(2**state[i][k])
                textsize = cv2.getTextSize(text, font, 0.5, 2)[0]
                cv2.putText(img,text,
                            (int(seperator_width/2+k*spacing+(spacing-textsize[0])/2),
                             int(top_margin+seperator_width/2+i*spacing+(spacing+textsize[1])/2)),
                            font,0.5,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(img,text,(int(seperator_width/2+k*spacing+(spacing-textsize[0])/2),
                                      int(top_margin+seperator_width/2+i*spacing+(spacing+textsize[1])/2)),
                            font,0.5,(255,255,255),1,cv2.LINE_AA)
        # Draw outline grid
        for i in range(5):
            cv2.line(img, 
                    (int(seperator_width/2)+i*spacing,int(top_margin+seperator_width/2)),
                    (int(seperator_width/2)+i*spacing,int(graphic_size+top_margin-seperator_width/2)), 
                    background_color, seperator_width)
        for i in range(5):
            cv2.line(img,
                    (int(seperator_width/2),int(top_margin+seperator_width/2)+i*spacing),
                    (int(graphic_size-seperator_width/2),int(top_margin+seperator_width/2)+i*spacing),
                    background_color,seperator_width)
        return img

    def makeGraph(self, logFile, graphFile=None, rollingWindow=30):
        scores = []
        with open(logFile, mode='r') as log_File:
            reader = csv.reader(log_File, delimiter='\n')
            for row in reader:
                scores.append(int(row[0]))
        rollingAverages = numpy.convolve(scores, numpy.ones((rollingWindow,))/rollingWindow, mode='valid')
        x = numpy.arange(len(rollingAverages))+rollingWindow/2
        if graphFile is None:
            plt.plot(x, rollingAverages, label=self.getTag())
        else:
            plt.plot(x, rollingAverages)
        plt.xlabel('Trial')
        plt.ylabel('Score')
        if graphFile is not None:
            plt.savefig(graphFile)
            plt.clf()
    
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
