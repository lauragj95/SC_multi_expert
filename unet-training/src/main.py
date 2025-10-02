import os
import argparse

import torch

from data import get_data_supervised
from utils.globals import init_global_config
import utils.globals
from model_handler import ModelHandler
from utils.logging import start_logging
from tensorboardX import SummaryWriter

def main():
    print(os.curdir)

    start_logging()

    # load data
    trainloader, validateloader, testloader, annotators = get_data_supervised()
    writer = SummaryWriter(config['logging']['experiment_folder'])
    # load and train the model
    model_handler = ModelHandler(annotators)
    model_handler.train(trainloader, validateloader,testloader,writer)



if __name__ == "__main__":
    print('Load configuration')

    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_folder", "-ef", type=str, default="None",
                        help="Config path to experiment folder. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    init_global_config(args)
    config = utils.globals.config
    torch.manual_seed(config['model']['seed'])
    main()