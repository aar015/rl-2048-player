'''Code relating to the learning agents'''
import numpy
import pickle
import random
from abc import ABC, abstractmethod
from .game import Game


def randArgMax(a):
    '''Returns the argmax of the array. Ties are broken radnomly.'''
    return numpy.argmax(numpy.random.random(numpy.shape(a))*(a==numpy.max(a)))


def load_agent(fileName):
    '''Load agent using pickle
    input:
        fileName: Save file'''
    with open(fileName, 'rb') as lf:
        return pickle.load(lf)


class Agent(ABC):
    '''Abstract class defining required functions for an agent'''

    def __init__(self, mask):
        '''Initialize the agent
        input:
            mask: Mask used to understand the game'''
        self.mask = mask

    @abstractmethod
    def learn(self, prevState, action, state, reward):
        '''Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        pass

    @abstractmethod
    def chooseAction(self, state, actions):
        '''Choose next action to take
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        pass

    @abstractmethod
    def getActionConfidence(self, state):
        '''Get confidence for each action
        input:
            state: Current state of game
        output: Array with confidence for each action'''
        pass

    def play(self):
        '''Agent plays a single game
        output: Final score of game'''
        game = Game(self.mask.getBoardSize())
        prevState = game.state().copy()
        game_over = game.game_over()
        while not game_over:
            next_action = self.chooseAction(game.state(), game.available_actions())
            reward = game.do_action(next_action)
            self.learn(prevState, next_action, game.state(), reward)
            prevState = game.state().copy()
            game_over = game.game_over()
        return game.score()

    def train(self, numIterations=1000):
        '''Train agent over many games
        input:
            numIterations: Number of games to play
        output: List of final score of games'''
        scores = numpy.zeros(numIterations, dtype=numpy.int32)
        for i in range(numIterations):
            scores[i] = self.play()
        return scores

    def save(self, fileName):
        '''Save agent with pickle
        input:
            fileName: Save file '''
        with open(fileName, 'wb') as sf:
            pickle.dump(self, sf)


class QAgent(Agent):
    '''Class to perform q learning'''

    def __init__(self, mask, a, g, e):
        '''Initialize the agent
        input:
            mask: Mask used to understand the game
            a: Learning rate
            g: Discount factor
            e: Exploration rate'''
        super().__init__(mask)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        self.q_table = numpy.zeros((self.mask.getNumTupleStates(), 4), dtype=float)

    def learn(self, prevState, action, state, reward):
        '''Q Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        tupleStates = self.mask.getTupleStates(prevState)
        next_action = self.chooseOptimalAction(state, Game(self.mask.getBoardSize(), state).available_actions())
        qError = self.alpha*(reward+self.gamma*self.lookUp(state,next_action)-self.lookUp(prevState,action))
        self.q_table[tupleStates, action] += qError

    def chooseAction(self, state, actions):
        '''Choose next action to take with q algorithm
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        if numpy.size(actions) == 0:
            return 0
        elif (random.random() < self.epsilon):
            return actions[random.randint(0, numpy.size(actions) - 1)]
        return self.chooseOptimalAction(state, actions)

    def chooseOptimalAction(self, state, actions):
        '''Choose next action ignoring exploration
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        values = numpy.sum(self.lookUp(state), axis=0)
        for action in [0, 1, 2, 3]:
            if not numpy.isin(action, actions):
                values[action] = -1
        return randArgMax(values)

    def getActionConfidence(self, state):
        '''Get confidence for each possible action
        input:
            state: Current state of game
        output: Array with confidence that should take actions'''
        actions = Game(self.mask.getBoardSize(), state).available_actions()
        values = numpy.sum(self.lookUp(state), axis=0)
        for action in [0, 1, 2, 3]:
            if not numpy.isin(action, actions):
                values[action] = 0
        total = numpy.sum(values)
        if total == 0:
            return values
        return values/total

    def lookUp(self, state, action=None):
        ''' Look up value of state/action pair in look up table
        input:
            state: State to look up
            action: Next action to take. If action is none look up the value for each action.
        output: Array of values for each tuple state/action pair in look up table'''
        tupleStates = self.mask.getTupleStates(state)
        if action is None:
            return numpy.array([self.q_table[tupleState] for tupleState in tupleStates])
        else:
            return numpy.array([self.q_table[tupleState, action] for tupleState in tupleStates])


class SARSAAgent(QAgent):
    '''Class to perform SARSA learning'''

    def learn(self, prevState, action, state, reward):
        '''SARSA Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        tupleStates = self.mask.getTupleStates(prevState)
        next_action = self.chooseAction(state, Game(self.mask.getBoardSize(), state).available_actions())
        sarsaError = self.alpha*(reward+self.gamma*self.lookUp(state,next_action)-self.lookUp(prevState,action))
        self.q_table[tupleStates, action] += sarsaError


class TD0Agent(Agent):
    '''Class to perform TD0 learning'''

    def __init__(self, mask, a, g, e):
        '''Initialize the agent
        input:
            mask: Mask used to understand the game
            a: Learning rate
            g: Discount factor
            e: Exploration rate'''
        super().__init__(mask)
        self.alpha = a
        self.gamma = g
        self.epsilon = e
        self.v_table = numpy.zeros(self.mask.getNumTupleStates(), dtype=float)

    def learn(self, prevState, action, state, reward):
        '''TD0 Learning Algorithm
        input:
            prevState: State before action is taken
            action: Action taken
            state: State after action is taken
            reward: Reward recieved from action'''
        tupleStates = self.mask.getTupleStates(prevState)
        tdError = self.alpha*(reward+self.gamma*self.lookUp(state)-self.lookUp(prevState))
        self.v_table[tupleStates] += tdError

    def chooseAction(self, state, actions):
        '''Choose next action to take with td0 algorithm
        input:
            state: Current state of game
            actions: Possible actions to take
        output: Next action to take'''
        if numpy.size(actions) == 0:
            return 0
        elif (random.random() < self.epsilon):
            return actions[random.randint(0, numpy.size(actions) - 1)]
        values = numpy.full(4, -1, dtype=float)
        for action in actions:
            tempGame = Game(self.mask.getBoardSize(), state.copy())
            reward = tempGame.do_action(action)
            values[action] = reward + numpy.sum(self.lookUp(tempGame.state()))
        return randArgMax(values)

    def getActionConfidence(self, state):
        '''Get confidence for each possible action
        input:
            state: Current state of game
        output: Array with confidence that should take actions'''
        actions = Game(self.mask.getBoardSize(), state).available_actions()
        values = numpy.zeros(4, dtype=float)
        for action in actions:
            tempGame = Game(self.mask.getBoardSize(), state.copy())
            reward = tempGame.do_action(action)
            values[action] = reward + numpy.sum(self.lookUp(tempGame.state()))
        total = numpy.sum(values)
        if total == 0:
            return values
        return values/total

    def lookUp(self, state):
        ''' Look up value of state in look up table
        input:
            state: State to look up
        output: Value of state in look up table'''
        return numpy.array([self.v_table[tupleState] for tupleState in self.mask.getTupleStates(state)])
