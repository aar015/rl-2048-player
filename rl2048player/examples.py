import matplotlib.pyplot as plt
from .agents import QAgent, TD0Agent, SARSAAgent
from .masks import Mask_rxcx4


def example1(q_gif_file='example_output/example1/QGame.gif',
             td0_gif_file='example_output/example1/TD0Game.gif',
             sarsa_gif_file='example_output/example1/SARSAGAme.gif',
             graph_file='example_output/example1/graph.png'):
    '''Train a q agent, td0 agent, and sarsa agent. Make gif of each agent
    playing a game and graph showing their performance as they train.
    input:
        q_gif_file: File to save gif of q agent game
        td0_gif_file: File to save gif of td0 agent game
        sarsa_gif_file: File to save gif of sarsa game
        graph_file: File to save graph'''
    # Initialize mask
    mask = Mask_rxcx4()
    # Initialize Agents
    qagent = QAgent(mask)
    td0agent = TD0Agent(mask)
    sarsaagent = SARSAAgent(mask)
    # Train agents
    qscores = qagent.train(6000)
    td0scores = td0agent.train(6000)
    sarsacores = sarsaagent.train(6000)
    # Make Gifs
    qagent.makeGif(q_gif_file, graphic_size=200, top_margin=20,
                   seperator_width=6, num_trials=50)
    td0agent.makeGif(td0_gif_file, graphic_size=200, top_margin=20,
                     seperator_width=6, num_trials=50)
    sarsaagent.makeGif(sarsa_gif_file, graphic_size=200, top_margin=20,
                       seperator_width=6, num_trials=50)
    # Make Graph
    qagent.makeGraph(scores=qscores, rollingWindow=200)
    td0agent.makeGraph(scores=td0scores, rollingWindow=200)
    sarsaagent.makeGraph(scores=sarsacores, rollingWindow=200)
    plt.legend()
    # Save graph
    plt.savefig(graph_file)


def example2(save_file='example_output/example2/agent.pickle',
             log_file='example_output/example2/log.csv',
             graph1_file='example_output/example2/graph1.png',
             graph2_file='example_output/example2/graph2.png'):
    '''Train TD0 agent, save it, load it, and train some more
    input:
        save_file: File to save agent
        log_file: File to record logs
        graph1_file: File for graph 1
        graph2_file: File for graph 2'''
    # Initialize mask
    mask = Mask_rxcx4()
    # Initialize agent
    agent = TD0Agent(mask)
    # Train agent
    scores = agent.train(2500, log_file)
    # Make graph 1
    agent.makeGraph(scores=scores, graphFile=graph1_file, rollingWindow=100)
    # Save agent
    agent.save(save_file)
    # Load agent
    agent.load(save_file)
    # Train some more
    agent.train(2500, log_file, 'a')
    # Make graph 2
    agent.makeGraph(logFile=log_file, graphFile=graph2_file, rollingWindow=100)
    # Save agent
    agent.save(save_file)
