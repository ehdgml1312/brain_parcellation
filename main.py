
import numpy as np
from utils import *
from train import train
from arg_helper import *

args = parse_arguments()
config = get_config(args.config_file)
dice = []

for j in range(config.num_exp):
    dice.append(train(config,j))

dice = np.array(dice)
mean = np.mean(dice,axis=0)
std = np.std(dice,axis=0)
torch.save(mean,os.path.join(config.save_dir,'mean={0:.4f}'.format(mean)))
torch.save(std,os.path.join(config.save_dir,'std={0:.4f}'.format(std)))