import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import gflags

from common_flags import FLAGS



def make_and_save_histograms(pred_steerings, real_steerings,
                             img_name = "histograms.png"):
    """
    Plot and save histograms from predicted steerings and real steerings.
    
    # Arguments
        pred_steerings: List of predicted steerings.
        real_steerings: List of real steerings.
        img_name: Name of the png file to save the figure.
    """
    pred_steerings = np.array(pred_steerings)
    real_steerings = np.array(real_steerings)
    max_h = np.maximum(np.max(pred_steerings), np.max(real_steerings))
    min_h = np.minimum(np.min(pred_steerings), np.min(real_steerings))
    bins = np.linspace(min_h, max_h, num=50)
    plt.hist(pred_steerings, bins=bins, alpha=0.5, label='Predicted', color='b')
    plt.hist(real_steerings, bins=bins, alpha=0.5, label='Real', color='r')
    plt.title('Predicted vs. real steering angles')
    plt.legend(fontsize=10)
    plt.savefig(img_name, bbox_inches='tight')
    
    
def _main():
    
    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(FLAGS.experiment_rootdir, 'predicted_and_real_steerings.json')
    with open(fname_steer,'r') as f:
        results_dict = json.load(f)
    make_and_save_histograms(results_dict['pred_steerings'], results_dict['real_steerings'],
                             os.path.join(FLAGS.experiment_rootdir, "histograms.png"))


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)