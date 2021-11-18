'''

Original file: prepare_cnn_data.py

------------------------------------------------------------------------------------------

Modified in order to:
    a) compute several hdf5 files from a source directory.
    b) avoid frame pre-processing.

------------------------------------------------------------------------------------------

Splits the recordings into short sequences of a few seconds each. Subsets of these
sequences are used for training and testing, respectively.

'''

from __future__ import print_function
import h5py
import os, sys, argparse
import glob

from hdf5_deeplearn_utils import build_train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', help='Path to frame-based hdf5 files.')
    parser.add_argument('--rewrite', default=0, type=int, help='Whether to overwrite training and testing information.')
    parser.add_argument('--train_length', default=5*60, type=float, help='Length of training sequences in seconds.')
    parser.add_argument('--test_length', default=2*60, type=float, help='Length of testing sequences in seconds')
    args = parser.parse_args()

    # For every recording/hdf5 file
    recordings = glob.glob(args.source_folder + '/*.hdf5')
    for rec in recordings:            
        # Get the data
        dataset = h5py.File(rec, 'a')
    
        print('Calculating train/test split...')
        sys.stdout.flush()
        build_train_test_split(dataset, train_div=args.train_length, test_div=args.test_length, force=args.rewrite)
    
        print('Done.  Preprocessing complete.')
        filesize = os.path.getsize(rec)
        print('Final size: {:.1f}MiB to {}.'.format(filesize/1024**2, rec))
