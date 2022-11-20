"""
Helper script for visualizing optimized trajectories
Expects trajectories in trajopt format
See: https://github.com/aravindr93/trajopt.git
"""

import pickle, click, numpy as np

DESC = """
Helper script to visualize optimized trajectories (list of trajectories in trajopt format).\n
USAGE:\n
    $ python viz_trajectories.py --file path_to_file.pickle --repeat 100\n
"""


@click.command(help=DESC)
@click.option("--file", type=str, help="pickle file with trajectories", required=True)
@click.option(
    "--repeat", type=int, help="number of times to play trajectories", default=10
)
@click.option(
    "--outfile",
    type=str,
    help="outfile if offscreen visualization is desired",
    default=None,
)
def main(file, repeat, outfile):
    trajectories = pickle.load(open(file, "rb"))
    if outfile != None and outfile != "None":
        import skvideo.io

        trajectories = trajectories if type(trajectories) == list else [trajectories]
        for idx, traj in enumerate(trajectories):
            frames = traj.animate_result_offscreen()
            skvideo.io.vwrite(outfile, np.asarray(frames))
    else:
        for _ in range(repeat):
            trajectories = (
                trajectories if type(trajectories) == list else [trajectories]
            )
            for traj in trajectories:
                traj.animate_result()


if __name__ == "__main__":
    main()
