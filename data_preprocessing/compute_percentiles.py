'''

Compute some percentiles, in particular the median, an inferior percentile, and a superior percentile, from DVS data
in order to remove outliers and normalize event frames. These values are saved in a txt file.

Plot the histogram of positive and negative events.

'''


import h5py
import numpy as np
import os
import argparse
import glob
import collections
import math
import matplotlib.pyplot as plt
from keras.utils.generic_utils import Progbar


def compute_percentiles(all_events, inf, sup):
    """
    Compute the median, and an inferior and a superior percentiles defined by inf and sup, respectively.
    """
    # Accumulated sum
    prior_sum = np.cumsum(all_events[:, 1])
    
    # Total number of events    
    n_values = prior_sum[-1]
    
    # Position of the percentiles in the accumulated sum
    median_pos = math.ceil(0.5*n_values)
    inf_pos = math.ceil(inf*n_values)
    sup_pos = math.ceil(sup*n_values)
    
    # Index of the percentiles in the counter
    idx_median = np.array(np.where((prior_sum >= median_pos) != False))[0][0]
    idx_inf = np.array(np.where((prior_sum >= inf_pos) != False))[0][0]
    idx_sup = np.array(np.where((prior_sum >= sup_pos) != False))[0][0]
    
    # Get the values from the counter
    median = all_events[idx_median, 0]
    p_inf = all_events[idx_inf, 0]
    p_sup = all_events[idx_sup, 0]

    # Return median, inferior and superior percentiles
    return median, p_inf, p_sup


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', help='Path to frame-based hdf5 files.')
    parser.add_argument('--inf_pos_percentile', help='Inferior percentile for positive events.')
    parser.add_argument('--sup_pos_percentile', help='Superior percentile for positive events.')
    parser.add_argument('--inf_neg_percentile', help='Inferior percentile for negative events.')
    parser.add_argument('--sup_neg_percentile', help='Superior percentile for negative events.')
    args = parser.parse_args()

    # Initialize counters for positive and negative events
    all_pos_events = collections.Counter([])
    all_neg_events = collections.Counter([])

    # For every recording/hdf5 file
    recordings = glob.glob(args.source_folder + '/*.hdf5')
    prog_bar = Progbar(target=len(recordings))
    j = 0
    for rec in recordings:
               
        # Get the data
        f_in = h5py.File(rec, 'r')
        for key in f_in.keys():
            key = str(key)
            
            # Read DVS frames
            if key == 'dvs_frame':
                dvs_frames = f_in[key].value
                
                # For every dvs frame in the recording
                for i in range(dvs_frames.shape[0]):
                    pos_img = dvs_frames[i, :, :, 0]
                    pos_events = pos_img.flatten()
                    neg_img = dvs_frames[i, :, :, 1]
                    neg_events = neg_img.flatten()
                    
                    # Count positive and negative events
                    counter_pos = collections.Counter(pos_events[pos_events > 0])
                    counter_neg = collections.Counter(neg_events[neg_events > 0])
                    
                    # Update counters with events from new images
                    if i == 0:
                        all_pos_events = counter_pos
                        all_neg_events = counter_neg
                    else:
                        all_pos_events = all_pos_events + counter_pos
                        all_neg_events = all_neg_events + counter_neg

        f_in.close()
        prog_bar.update(j)
        j += 1
               
    # Sort the counters according to the number of events (not frequency)
    all_pos_events = np.array(sorted(all_pos_events.items()))
    all_neg_events = np.array(sorted(all_neg_events.items()))

    # Plot histogram of positive and negative events
    plt.hist(all_pos_events[:, 0], weights=all_pos_events[:, 1], bins=all_pos_events.shape[0],
             alpha=0.5, label='Positive', color='b')
    plt.hist(-1*all_neg_events[:, 0], weights=all_neg_events[:, 1], bins=all_neg_events.shape[0],
             alpha=0.5, label='Negative', color='r')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(args.source_folder, 'events.png'), bbox_inches='tight')
    
    # Compute and save percentiles for positive and negative events
    pos_median, pos_inf, pos_sup = compute_percentiles(all_pos_events, args.inf_pos_percentile, args.sup_pos_percentile)
    neg_median, neg_inf, neg_sup = compute_percentiles(all_neg_events, args.inf_neg_percentile, args.sup_neg_percentile)
    print("pos_median = {}, pos_inf = {}, pos_sup = {},"
          "neg_median = {}, neg_inf = {}, neg_sup = {}".format(pos_median, pos_inf, pos_sup, neg_median, neg_inf, neg_sup))
    np.savetxt(os.path.join(args.source_folder, 'percentiles.txt'),
               [pos_median, pos_inf, pos_sup, neg_median, neg_inf, neg_sup],
               delimiter=',', header='pos_median, pos_inf, pos_sup, neg_median, neg_inf, neg_sup')

