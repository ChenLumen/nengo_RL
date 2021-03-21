import logging
import time
import nengo
import numpy as np
from grid import Cell as GridCell
from grid import ContinuousAgent, GridNode
from grid import World as GridWorld


class Cell(GridCell):
    def color(self):
        return 'black' if self.wall else None

    def load(self, char):
        if char == '#':
            self.wall = True
        else:
            self.wall = False


class WorldConfig:
    curr_ind = -1
    world_maps = [
        """
        #########
        #       #
        #       #
        #   ##  #
        #   ##  #
        #       #
        #########
        """,
        """
        #########
        #       #
        #       #
        #   ##  #
        #   ##  #
        #   ##  #
        #########
        """,
        """
        #########
        #       #
        #  ###  #
        #       #
        #  ###  #
        #       #
        #########
        """,
        """
        #########
        #       #
        #   #####
        #       #
        #####   #
        #       #
        #########
        """
    ]
    init_pos = [(1, 3, 2), (1, 3, 2), (1, 1, 1), (1, 1, 1)]

    world = None
    agent = None

    def get_init_pos(self):
        return self.init_pos[self.curr_ind]

    def get_map(self):
        return self.world_maps[self.curr_ind]

    def set_ind(self, new_ind):
        if 0 <= new_ind < len(self.world_maps):
            self.curr_ind = new_ind
            lines = self.get_map().splitlines()

            if len(lines) == 0:
                del lines[0]
            lines = [x.rstrip() for x in lines]
            for j, _ in enumerate(lines):
                for i, _ in enumerate(lines[0]):
                    self.world.get_cell(i, j).load(lines[j][i])

    def reset_pos(self):
        self.agent.x = self.get_init_pos()[0]
        self.agent.y = self.get_init_pos()[1]
        self.agent.dir = self.get_init_pos()[2]
        self.agent.cell = self.world.get_cell(self.agent.x, self.agent.y)


world_cfg = WorldConfig()
world = GridWorld(Cell, map=world_cfg.get_map(), directions=4)
agent = ContinuousAgent()
world_cfg.world = world
world_cfg.agent = agent
world_cfg.set_ind(0)
world_cfg.world.add(
    agent,
    x=world_cfg.get_init_pos()[0],
    y=world_cfg.get_init_pos()[1],
    dir=world_cfg.get_init_pos()[2],
)

# model parameters
learn_rate = 1e-4
learn_synapse = 0.030
learn_timeout = 60.0
radar_dim = 5
turn_bias = 0.25
action_threshold = 0.1
init_transform = [0.8, 0.6, 0.7]

seed = int(time.time())
print('Using seed: {0}'.format(seed))

if '__page__' in locals():
    print("Press 'q' to enable exploration and reset agent position.")
    print("Press 'e' to disable exploration and reset agent position.")
    print("Press 'w' to reset agent position.")
    print("Press 1-{0} to change maps.".format(len(world_cfg.world_maps)))

model = nengo.Network(seed=seed)
with model:
    env = GridNode(world_cfg.world, dt=0.005)

    def move(t, x, my_world=world_cfg):
        speed, rotation = x
        dt = 0.001
        max_speed = 10.0
        max_rotate = 10.0
        my_world.agent.turn(rotation * dt * max_rotate)
        success = my_world.agent.go_forward(speed * dt * max_speed)
        if not success:
            my_world.agent.color = "red"
            return 0
        else:
            my_world.agent.color = "blue"
            return turn_bias + speed

    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)
    movement_node = nengo.Node(move, size_in=2, label="reward")
    nengo.Connection(movement, movement_node)

    # Generate the context (radar distance to walls front, left, right)
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, radar_dim) + agent.dir) % world.directions
        return [agent.detect(d, max_distance=4)[0] for d in angles]

    sim_radar = nengo.Node(detect)

    radar = nengo.Ensemble(n_neurons=100*radar_dim, dimensions=radar_dim, radius=4)
    nengo.Connection(sim_radar, radar, synapse=learn_synapse)

    # Create the action selection networks
    bg = nengo.networks.BasalGanglia(3)
    thal = nengo.networks.Thalamus(3)
    nengo.Connection(bg.output, thal.input)

    # Convert the selection actions to movement transforms
    nengo.Connection(thal.output[0], movement, transform=[[1], [0]])
    nengo.Connection(thal.output[1], movement, transform=[[0], [1]])
    nengo.Connection(thal.output[2], movement, transform=[[0], [-1]])

    def u_fwd(x):
        return 0.8

    def u_left(x):
        return 0.6

    def u_right(x):
        return 0.7

    conn_fwd = nengo.Connection(radar, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES(learn_rate))
    conn_left = nengo.Connection(radar, bg.input[1], function=u_left, learning_rule_type=nengo.PES(learn_rate))
    conn_right = nengo.Connection(radar, bg.input[2], function=u_right, learning_rule_type=nengo.PES(learn_rate))

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

        return activate * (np.multiply(actions, (utils - r) * (1 - r) ** 5 +
                                       np.multiply((1 - actions), (utils - 1) * (1 - r) ** 5)))

    errors = nengo.Node(error_func, size_in=8, size_out=3)
    nengo.Connection(thal.output, errors[:3])
    nengo.Connection(bg.input, errors[3:6])
    nengo.Connection(movement_node, errors[6])

    nengo.Connection(errors[0], conn_fwd.learning_rule)
    nengo.Connection(errors[1], conn_left.learning_rule)
    nengo.Connection(errors[2], conn_right.learning_rule)

    # class LearnActive:
    #     """Class to store persistent learning state"""
    #
    #     def __init__(self, my_world, page_data=None):
    #         self.my_world = my_world
    #         self.page = page_data
    #
    #         self._is_learning = 1
    #         # _is_learning values:
    #         # <0: no learning
    #         # 1: learning, will stop at learning_timeout
    #         # 2: continuous learning
    #
    #     def __call__(self, t):
    #         if self.page is not None:
    #
    #             init_agent_pos = False
    #             # Create a dictionary isntead of if/else
    #             # "<key press>": (<learning>, <init_agent_pos>)
    #             keyboard_dict = {
    #                 "q": (2, True),
    #                 "e": (-1, True),
    #                 "w": (self._is_learning, True)
    #             }
    #
    #             for k in self.page.keys_pressed:
    #                 if k.isdigit():
    #                     new_map_ind = int(k) - 1
    #                     if new_map_ind != self.my_world.curr_ind:
    #                         self.my_world.set_ind(new_map_ind)
    #                         init_agent_pos = True
    #                     elif k in list(keyboard_dict.keys()):
    #                         self._is_learning, init_agent_pos = keyboard_dict[k]
    #
    #             learning = (
    #                 (t <= learn_timeout) or (self._is_learning == 2)
    #             ) and self._is_learning > 0
    #
    #             self._nengo_html_ = """
    #             <svg width="100%" height="100%" viewbox="0 0 200 75">
    #              <text x="50%" y="50%" fill="{0}" text-anchor="middle"
    #              alignment-baseline="middle" font-size="50">{1}</text>
    #             </svg>
    #             """.format(
    #                 "red" if learning else "grey",
    #                 "Explore: ON" if learning else "Explore: Off",
    #             )
    #
    #             if not learning and self._is_learning == 1:
    #                 init_agent_pos = True
    #                 self._is_learning = -1
    #
    #             if init_agent_pos:
    #                 self.my_world.reset_pos()
    #
    #             return int(learning)
    #         else:
    #             # Keyboard state branch not detected. Default to continuous learning
    #             self._nengo_html_ = """
    #             <svg width="100%" height="100%" viewbox="0 0 200 75">
    #              <text x="50%" y="50%" fill="red" text-anchor="middle"
    #              alignment-baseline="middle" font-size="50">
    #              Explore: ON</text>
    #             </svg>
    #             """
    #
    #             return 1

    learn_on = nengo.Node(1)
    nengo.Connection(learn_on, errors[7])