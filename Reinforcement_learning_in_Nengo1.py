import grid
import nengo
import numpy as np

mymap="""
#######
#     #
# # # #
# # # #
#G   R#
#######
"""


world = grid.World(grid.GridCell, map=mymap, directions=4)
body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

tau=0.1


def move(t, x):
    '''Defines a continuous action policy for the agent'''
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)

    if int(body.x) == 1:
        world.grid[4][4].wall = True
        world.grid[4][2].wall = False
    if int(body.x) == 4:
        world.grid[4][2].wall = True
        world.grid[4][4].wall = False


def sensor(t):
    '''Obtain environment state using sensors'''
    angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
    return [body.detect(d, max_distance=4)[0] for d in angles]


def braiten(x):
    '''Compute input to movement function based on sensor'''
    turn = x[2] - x[0]
    spd = x[1] - 0.5
    return spd, turn


def position_func(t):
    '''Create unit normalized state representation of grid world'''
    return body.x / world.width * 2 - 1, 1 - body.y / world.height * 2, body.dir / world.directions


with nengo.Network(seed=2) as model:
    env = grid.GridNode(world, dt=0.005)

    # define nodes and ensembles for managing action policy
    movement = nengo.Node(move, size_in=2)
    stim_radar = nengo.Node(sensor)

    radar = nengo.Ensemble(n_neurons=50, dimensions=3, radius=4, seed=2,
                           noise=nengo.processes.WhiteSignal(10, 0.1, rms=1))

    nengo.Connection(stim_radar, radar)
    nengo.Connection(radar, movement, function=braiten)

    # encode state information in ensemble of neurons
    position = nengo.Node(position_func)
    state = nengo.Ensemble(100, 3)

    nengo.Connection(position, state, synapse=None)

    reward = nengo.Node(lambda t: body.cell.reward)

    value = nengo.Ensemble(n_neurons=50, dimensions=1)

    # this sets up learning on our connection between state and value encodings
    learn_conn = nengo.Connection(state, value, function=lambda x: 0,
                                  learning_rule_type=nengo.PES(learning_rate=1e-4, pre_tau=tau))

    # this connection adds the reward to the learning signal
    nengo.Connection(reward, learn_conn.learning_rule, transform=-1, synapse=tau)

    # this connection adds the observed observed value
    nengo.Connection(value, learn_conn.learning_rule, transform=-0.9, synapse=0.01)

    # this connection substracts the predicted value
    nengo.Connection(value, learn_conn.learning_rule, transform=1, synapse=tau)