""" Based on the code from georgwiese from https://github.com/georgwiese/2048-rl
    Algorithms and strategies to play 2048 and collect experience."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from game_logic.game import Game

def play(agent, verbose=False):
    """Plays a single game, using a provided strategy.
    Args:
      agent:   An agent to play and learn game
      verbose: If verbose is true also return intermediate game states and scores
    Returns:
      final score and log if verbose is set to true"""
    game = Game()
    # record previous state to update learning algorithm
    prevState = game.state().copy()
    # metaState is state after move is made, but before adding random tile
    prevMetaState = game.state().copy()
    # whether or not game has reached a gameover state
    game_over = game.game_over()
    # If verbose record a log of game states and scores
    if verbose:
        log = []
        log.append([game.score(), game.state().copy()])
    while not game_over:
        # Choose next action
        next_action = agent.chooseAction(game.state().copy(), game.available_actions())
        # Perform action and recieve a reward
        reward = game.do_action(next_action)
        # Record new metaState of game
        metaState = game.state().copy()
        # Add random tile
        game.add_random_tile()
        # If verbose add new state and score to log
        if verbose:
            log.append([game.score(), game.state().copy()])
        # Update learning algorithm
        agent.learn(prevMetaState, prevState, metaState, game.state().copy(), reward)
        # Update prevState and prevMetaState
        prevState = game.state().copy()
        prevMetaState = metaState
        # Check if game is over
        game_over = game.game_over()
    # If verbose return final score and log
    if verbose:
        return game.score(), log
    # Else return just final score of game
    else:
        return game.score()
