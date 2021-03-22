import nengo
import numpy as np

import ccm.lib.grid
import ccm.lib.continuous
import ccm.ui.nengo

mymap = """
#########
#       #
#       #
#   ##  #
#   ##  #
#       #
#########
"""
action_threshold = 0.1
learn_rate = 1e-4
learn_synapse = 0.030
radar_dim = 5
turn_bias = 0.25


class Cell(ccm.lib.grid.Cell):
    def color(self):
        return 'black' if self.wall else None

    def load(self, char):
        if char == '#':
            self.wall = True


world = ccm.lib.cellular.World(Cell, map=mymap, directions=4)

body = ccm.lib.continuous.Body()
world.add(body, x=1, y=3, dir=2)

model = nengo.Network(seed=8)
with model:
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 10.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        success = body.go_forward(speed * dt * max_speed)
        if not success:
            return 0
        else:
            return turn_bias + speed


    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)

    movement_node = nengo.Node(move, size_in=2, label='reward')
    nengo.Connection(movement, movement_node)

    env = ccm.ui.nengo.GridNode(world, dt=0.005)

    def detect(t):
        angles = (np.linspace(-0.5, 0.5, radar_dim) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]

    stim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=100*radar_dim, dimensions=radar_dim, radius=4)
    nengo.Connection(stim_radar, radar, synapse=learn_synapse)

    bg = nengo.networks.actionselection.BasalGanglia(3)
    thal = nengo.networks.actionselection.Thalamus(3)
    nengo.Connection(bg.output, thal.input)

    def u_fwd(x):
        return 0.8

    def u_left(x):
        return 0.6

    def u_right(x):
        return 0.7

    conn_fwd = nengo.Connection(radar, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES(learn_rate))
    conn_left = nengo.Connection(radar, bg.input[1], function=u_left, learning_rule_type=nengo.PES(learn_rate))
    conn_right = nengo.Connection(radar, bg.input[2], function=u_right, learning_rule_type=nengo.PES(learn_rate))

    nengo.Connection(thal.output[0], movement, transform=[[1], [0]])
    nengo.Connection(thal.output[1], movement, transform=[[0], [1]])
    nengo.Connection(thal.output[2], movement, transform=[[0], [-1]])

    # Generate the training (error) signal
    def error_func(t, x):
        actions = np.array(x[:3])
        utils = np.array(x[3:6])
        r = x[6]
        activate = x[7]

        max_action = max(actions)
        actions[actions < action_threshold] = 0
        actions[actions != max_action] = 0
        actions[actions == max_action] = 1

        return activate * (np.multiply(actions, (utils - r) ** 5) +
                           np.multiply((1 - actions), (utils - 1) * (1 - r) ** 5))

    errors = nengo.Node(error_func, size_in=8, size_out=3)
    nengo.Connection(thal.output, errors[:3])
    nengo.Connection(bg.input, errors[3:6])
    nengo.Connection(movement_node, errors[6])

    nengo.Connection(errors[0], conn_fwd.learning_rule)
    nengo.Connection(errors[1], conn_left.learning_rule)
    nengo.Connection(errors[2], conn_right.learning_rule)

    learn_on = nengo.Node(1)
    nengo.Connection(learn_on, errors[7])