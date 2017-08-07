# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from osim.env import *
import rl_build

import argparse
import math
def getEnvironment(visualize=False):
    env = RunEnv(visualize)
    env.reset()


# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
args = parser.parse_args()

# Load walking environment
env = RunEnv(args.visualize)
env.reset()



# Total number of steps in training
nallsteps = args.steps


agent = rl_build.buildAgent(env)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, 
            nb_max_episode_steps=env.timestep_limit, log_interval=10000)
    
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

if not args.train:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
