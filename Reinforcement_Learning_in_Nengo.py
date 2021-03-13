import td_grid
import nengo
import numpy as np

from utils import weight_init

# high-level configuration
stepsize = 10 # milleseconds between each action
n_neurons = 2500
n_actions = 4

fast_tau = 0
slow_tau = 0.01

# build the world and add the agent
env_map = """
#######
#     #
#     #
#  G  #
#     #
#     #
#######
"""

agent = td_grid.ContinuousAgent()

environment = td_grid.World(td_grid.GridCell, map=env_map, directions=4)
environment.add(agent, x=2, y=3, dir=2)

env_update = td_grid.EnvironmentInterface(agent, n_actions=4, epsilon=0.1)


def sensor(t):
    '''Return current x,y coordinates of agent as one hot representation'''
    data = np.zeros(25)
    idx = 5 * (agent.x - 1) + (agent.y - 1)
    data[idx] = 1

    return data


def reward(t):
    '''Call to get current reward signal provided to agent'''
    return agent.reward


with nengo.Network(seed=2) as model:
    env = td_grid.GridNode(environment, dt=0.001)

    # define nodes for plotting data, managing agent's interface with environment
    reward_node = nengo.Node(reward, size_out=1, label='reward')
    sensor_node = nengo.Node(sensor, size_out=25, label='sensor')
    update_node = nengo.Node(env_update.step, size_in=4, size_out=12, label='env')
    qvalue_node = nengo.Node(size_in=4)

    # define neurons to encode state representations
    state = nengo.Ensemble(n_neurons=n_neurons, dimensions=25,
                           intercepts=nengo.dists.Choice([0.15]), radius=2)

    # define neurons that compute the learning signal
    learn_signal = nengo.Ensemble(n_neurons=1000, dimensions=4)

    # connect the sensor to state ensemble
    nengo.Connection(sensor_node, state, synapse=None)
    reward_probe = nengo.Probe(reward_node, synapse=fast_tau)

    # connect state representation to environment interface
    q_conn = nengo.Connection(state.neurons, update_node,
                              transform=weight_init(shape=(n_actions, n_neurons)),
                              learning_rule_type=nengo.PES(1e-3, pre_tau=slow_tau),
                              synapse=fast_tau)

    # connect update node to error signal ensemble w/ fast, slow conns to compute prediction error
    nengo.Connection(update_node[0:n_actions], learn_signal, transform=-1, synapse=slow_tau)
    nengo.Connection(update_node[n_actions:2 * n_actions], learn_signal, transform=1, synapse=fast_tau)

    # connect the learning signal to the learning rule
    nengo.Connection(learn_signal, q_conn.learning_rule, transform=-1, synapse=fast_tau)

    # for plotting and visualization purposes
    nengo.Connection(update_node[2 * n_actions:], qvalue_node, synapse=fast_tau)

with nengo.Simulator(model) as sim:
    sim.run(10)