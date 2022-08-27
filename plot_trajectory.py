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