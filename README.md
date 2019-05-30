# Reinforcement Learning to Play 2048
This project trains agents to play the game 2048 using Q, SARSA, and TD0 learning agents. 

## Installation
This project is pip installable. On linux it can be installed by entering, ```pip3 install rl2048player``` into your terminal. For other operating systems, follow the pip instructions for that system. 

## Features
In order to use this package, you must first import it using
```
import rl2048player as rl
```
Now, I will review some of the main features of this package. For most of these features, there are addditional options not explained in this review. All options for any command can be found by reviewing the documentation in the code. 

### Masks
Pefore creating a learning agent, you must initialize a mask. Masks translate between the game board and the learning agent. This allows you to change the way the agent understands the game board without changing the agent itself. Currently, only one mask has been implemented. This mask breaks down the board into rows, columns, and 2x2 squares. This has the effect of  decoupling parts of the board that do not interact strongly with each other. The mask can be initialized using the code

```mask = rl.masks.Mask_rxcx4()```

### Agents
Once the mask has been initialized, you can initialize the agents. Currently, there are three agents that have been implemented: one using a Q learning algorithm, one using a SARSA learning algorithm, and one using a TD0 learning algorithm. These agents can be initialized by
```
agent = rl.agents.QAgent(mask)
agent = rl.agents.SARSAAgent(mask)
agent = rl.agents.TD0Agent(mask)
```
respectively. There are also options to change the agent's hyperparameters when initializing them. The default values are the hyperparameters I have found to work best. 

#### Training
To train the agents over x number of games and record the scores use the code
```
scores = agent.train(x)
```
You can then look at a graph of those scores by 
```
agent.makeGraph(scores)
plt.show()
```
There are also options to automatically save the scores and graph to a file. 

#### Making Gif's
Once the agent has been trained, you can create a gif of the agent playing a game and save it to a file by using the command
```
agent.makeGif(gif_file)
```

#### Saving and Loading
You can also save trained agent by using
```
agent.save(agent_file)
```
This save function only saves the look-up table for this agent, not the whole object. In order to load the agent, you need to initialize an agent with the same mask and hyperparameters and then load the look-up table using
```
agent = rl.agents.AppropriateAgent(AppropriateMask)
agent.load(agent_file)
```

### Examples
If you don't want to have to do all of this to use this package, there are some examples precoded in the package. Currently there are two examples that can be accessed using
```
rl.examples.example1()
rl.examples.example2()
```

## Acknowledgments
The game logic for this project is based on code by georgwiese which can be found at https://github.com/georgwiese/2048-rl. The implementation of the learning agents is based on method described in the paper Temporal difference learning of n-tuple networks for the game 2048, by M.Szubertand W. Jaskowski. All code is my original work except when noted. 

## Future Work
In the future, I plan to work on implementing additional masks and learning agents. 
