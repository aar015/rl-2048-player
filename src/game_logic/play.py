""" Based on the code from georgwiese from https://github.com/georgwiese/2048-rl
    Algorithms and strategies to play 2048 and collect experience."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from game_logic.game import Game
import csv

def play(strategy, learning, logFile=None, threshold=0):
    """Plays a single game, using a provided strategy.
    Args:
      strategy:  A function that takes as argument a state and a list of available
                 actions and returns an action from the list.
      learning:  A function that takes as arguement previous state, reward, and 
                 current state and updates the strategy function
      logFile:   File to logs to. If None never write a log file.
      threshold: If final score exceeds scoreThreshold then write a log
    Returns:
      score where score is the game score or -1 if try to do an illigial move"""
    game = Game()
    # record previous state to update learning algorithm
    prevState = game.state().copy()
    # metaState is state after move is made, but before adding random tile
    prevMetaState = game.state().copy()
    # whether or not game has reached a gameover state
    game_over = game.game_over()
    # If a logFile was provided record a log of game states
    if logFile is not None:
        log = []
        log.append(game.state().copy().flatten())
    while not game_over:
        # Choose next action
        next_action = strategy(game.state().copy(), game.available_actions())
        # Perform action and recieve a reward
        reward = game.do_action(next_action)
        # Record new metaState of game
        metaState = game.state().copy()
        # Add random tile
        game.add_random_tile()
        # If a logFile was provided and new state to log
        if logFile is not None:
            log =[]
            log.append(game.state().copy().flatten())
        # Update learning algorithm
        learning(prevMetaState, prevState, metaState, game.state().copy(), reward)
        # Update prevState and prevMetaState
        prevState = game.state().copy()
        prevMetaState = metaState
        # Check if game is over
        game_over = game.game_over()
    # If logFile was provided and game was a good game write the log
    if logFile is not None and game.score()>threshold:
        with open(logFile, mode='w') as log_File:
            writer = csv.writer(log_File, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONE)
            writer.writerow(['The final score is', game.score()])
            writer.writerows(log)
    # Return final score of game
    return game.score()
