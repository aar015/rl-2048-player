from controllers.train import train
from learning.algorithms.td_meta_agent import TDMetaAgent
from learning.masks.mask_rxcx4 import Mask_rxcx4
import os


def main():
    mask = Mask_rxcx4()
    agent = TDMetaAgent(mask, 0.0025, 0.95, .001)
    log_File = os.path.join(os.getcwd(),'logs', agent.getTag() + '.csv')
    train(agent, 10, log_File)


if __name__ == "__main__":
    main()
