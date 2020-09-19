"""
Helper script for visualizing optimized trajectories
Expects trajectories in trajopt format
See: https://github.com/aravindr93/trajopt.git
"""

import pickle
import click 

from vtils.plotting.simple_plot import plot, show_plot, save_plot

def plot_path(path, fig_name, info_keys=None):
  
	# obs / user
	if info_keys:
		for key in sorted(info_keys):
			print(key)
			plot(path['env_infos'][key], subplot_id=(2,3,1), legend=key,  plot_name='user', fig_name=fig_name)
	else:
		plot(path['observations'], subplot_id=(2,3,1),  plot_name='observations', fig_name=fig_name)
	# actions
	plot(path['actions'], subplot_id=(2,3,4),  plot_name='actions', fig_name=fig_name)

	# score
	score_keys = [key for key in path['env_infos'].keys() if 'score' in key]
	for key in sorted(score_keys):
		plot(path['env_infos'][key], subplot_id=(1,3,3), legend=key,  plot_name='score', fig_name=fig_name)

	# Rewards
	for key in sorted(path['env_infos']['rwd_dict'].keys()):
		plot(path['env_infos']['rwd_dict'][key], subplot_id=(1,3,2), legend=key, plot_name='rewards', fig_name=fig_name)



DESC = '''
Helper script to visualize optimized trajectories (list of trajectories in trajopt format).\n
USAGE:\n
	$ python viz_trajectories.py --file path_to_file.pickle --repeat 100\n
'''
@click.command(help=DESC)
@click.option('-f', '--file', type=str, help='pickle file with trajectories', required= True)
@click.option('-r', '--repeat', type=int, help='number of times to play trajectories', default=10)
def main(file, repeat):
	trajectories = pickle.load(open(file, 'rb'))
	for _ in range(repeat):
		for traj in trajectories:
			path = traj.animate_and_gather_path()
			plot_path(path, 'test')
			show_plot()

if __name__ == '__main__':
	main()