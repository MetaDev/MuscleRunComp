import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse
import rl_build



parser = argparse.ArgumentParser(description='submit locomotion model')
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
args = parser.parse_args()
# Settings
remote_base = 'http://grader.crowdai.org:1729'

# Command line parameters

env = RunEnv(visualize=False)
client = Client(remote_base)
token="638463359c8d645f576da99a62e8c327"
# Create environment
observation = client.env_create(token)

agent =rl_build.buildAgent(env)
agent.load_weights(args.model)
function = lambda x : agent.actor.predict(np.reshape([x], (1, 1, -1)), batch_size=1)

# Run a single step
#
# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    v = np.array(observation).reshape((-1,1,env.observation_space.shape[0]))
    [observation, reward, done, info] = client.env_step(function(observation)[0].tolist())
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
