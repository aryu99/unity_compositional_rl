import mlagents
import imitation

from mlagents.trainers.demo_loader import demo_to_buffer
from imitation.data.types import Transitions
import os
import numpy as np

def load_and_parse_unity_demo(demo_file:str, sequence_length:int) \
                                    -> imitation.data.types.Transitions:
    """
    Load a unity .demo protobuffer file and parse it into a form usable by 
    the algorithms within the "imitation" python package.

    Arguments
    ---------
    demo_file:
        The path to the .demo file in question.
    sequence_length:
        The length of the trajectories to fill the buffer.

    Returns
    -------
    transitions:
        A imitation.data.types.Transitions object storing the demonstration
        data in a format that can be used by the "imitation" library.
    """
    file_path = os.path.abspath(demo_file)

    _, agent_buffer = demo_to_buffer(file_path=file_path, 
                                        sequence_length=sequence_length)
    return parse_agent_buffer(agent_buffer=agent_buffer)

def parse_agent_buffer(agent_buffer:mlagents.trainers.buffer.AgentBuffer) \
                                        -> imitation.data.types.Transitions:
    """
    Parse a Unity mlagents AgentBuffer object into a Transitions object
    from the imitation learning package.

    Arguments
    ---------
    agent_buffer:
        The agent buffer to be parsed.
    
    Returns
    -------
    transitions:
        The output Transitions object.
    """
    items = list(agent_buffer._fields.items())

    for key, val in items:

        if type(key) is not tuple:

            # Get "dones"
            if key.name == 'DONE':
                dones = np.array(val, dtype=bool)

            # Get actions
            if key.name == 'CONTINUOUS_ACTION':
                acts = np.array(val)

        # Get observations
        elif (key[0].name == 'OBSERVATION'):
            obs = np.array(val)

    next_obs = obs[1::]
    obs = obs[0:-1]
    dones = dones[0:-1]
    acts = acts[0:-1]
    infos = np.array([{}] * obs.shape[0])

    transitions_dict = {
        'obs' : obs,
        'dones' : dones,
        'acts' : acts,
        'next_obs' : next_obs,
        'infos' : infos,
    }

    return Transitions(**transitions_dict)

def main():
    file_path = os.path.abspath('../demos/labyrinth/IRLnonlava.demo')

    output = load_and_parse_unity_demo(demo_file=file_path, sequence_length=3382)

    # behavior_spec, agent_buffer = demo_to_buffer(file_path=file_path, sequence_length=3382)
    # items = list(agent_buffer._fields.items())
    print(output)

if __name__ == '__main__':
    main()