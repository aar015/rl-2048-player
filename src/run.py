from learning.td_learning import TD0Agent
from learning.q_learning import QAgent
from learning.sarsa_learning import SARSAAgent
from game_logic.mask_rxcx4 import Mask_rxcx4
import os


def main():
    mask = Mask_rxcx4()
    agent = TD0Agent(mask, 0.025, .999, 0.005)
    doTests(agent)
    agent = QAgent(mask, 0.025, .99, 0.005)
    doTests(agent)
    agent = SARSAAgent(mask, 0.025, .9, 0.005)
    doTests(agent)


def doTests(agent):
    log_file = os.path.join(os.getcwd(),'logs', agent.getTag()+'.csv')
    save_file = os.path.join(os.getcwd(),'agents',agent.getTag()+'.pickle')
    gif_file = os.path.join(os.getcwd(), 'games', agent.getTag()+'.gif')
    graph_file = os.path.join(os.getcwd(), 'graphs', agent.getTag()+'.png')
    # agent.load(save_file)
    agent.train(2500, log_file, 'w')
    agent.save(save_file)
    agent.makeGif(gif_file, graphic_size=200, top_margin=20, seperator_width=6,
                 num_trials=10)
    agent.makeGraph(log_file, graph_file, 100)


if __name__ == "__main__":
    main()
