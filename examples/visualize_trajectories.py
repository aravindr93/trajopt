"""
Helper script for visualizing optimized trajectories
Expects trajectories in trajopt format
See: https://github.com/aravindr93/trajopt.git
"""

import pickle
import click 

DESC = '''
Helper script to visualize optimized trajectories (list of trajectories in trajopt format).\n
USAGE:\n
    $ python viz_trajectories.py --file path_to_file.pickle --repeat 100\n
'''
@click.command(help=DESC)
@click.option('--file', type=str, help='pickle file with trajectories', required= True)
@click.option('--repeat', type=int, help='number of times to play trajectories', default=10)
def main(file, repeat):
	trajectories = pickle.load(open(file, 'rb'))
	for _ in range(repeat):
		for traj in trajectories:
			traj.animate_result()

if __name__ == '__main__':
	main()