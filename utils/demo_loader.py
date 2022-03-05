from mlagents.trainers.demo_loader import demo_to_buffer
import os

def load_demo_obs_and_actions(demo_file:str, sequence_length:int):
    file_path = os.path.abspath(demo_file)

    behavior_spec, agent_buffer = demo_to_buffer(file_path=file_path, sequence_length=sequence_length)
    items = list(agent_buffer._fields.items())

    # Get observations

    # Get actions


def main():
    file_path = os.path.abspath('../demos/crawler/ExpertCrawler.demo')

    behavior_spec, agent_buffer = demo_to_buffer(file_path=file_path, sequence_length=10011)
    items = list(agent_buffer._fields.items())
    print(agent_buffer)

if __name__ == '__main__':
    main()