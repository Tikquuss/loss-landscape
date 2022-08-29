"""
    Plot the optimization path in the space spanned by principle directions.
"""

from plot_2D import plot_trajectory as plot_traj
from projection import setup_PCA_directions, project_trajectory
from model_loader import load
from net_plotter import get_weights

def plot_trajectory(args, model_files, lightning_module_class) :
    last_model_file = model_files[-1] 
    net = load(lightning_module_class, model_file = last_model_file)
    w = get_weights(net) # initial parameters
    s = net.state_dict()

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        dir_file = setup_PCA_directions(args, model_files, w, s, lightning_module_class)

    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file = project_trajectory(dir_file, w, s, model_files, args.dir_type, 'cos', lightning_module_class)

    plot_traj(proj_file, dir_file)

    return proj_file, dir_file

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--model_folder', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')

    args = parser.parse_args()

    # TODO
    model_files = []
    lightning_module_class = None
    
    proj_file, dir_file = plot_trajectory(args, model_files, lightning_module_class)