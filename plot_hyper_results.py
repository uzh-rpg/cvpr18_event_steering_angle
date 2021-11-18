import os
import sys
import gflags
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

FLAGS = gflags.FLAGS

gflags.DEFINE_string('exp_dir', "./model", 'Folder '
                     ' containing all the learning-rate experiments')


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
        
    evas = []
    rmses = []
    lrs = []
    
    
    experiments = glob.glob(FLAGS.exp_dir + '/expr*')
    for exp_name in experiments:
        file_name = os.path.join(exp_name, 'test_results.json')
        
        lr = file_name.split('/')[-2]
        lr = float(lr.split('_')[-1])
        lrs.append(lr)
        
        with open(file_name, 'r') as f:
            results_dict = json.load(f)
        
        predicted_dict = results_dict['predicted']
        
        evas.append(predicted_dict[0]['evas'][0])
        rmses.append(predicted_dict[0]['rmse'])
        
    evas = np.array(evas)
    rmses = np.array(rmses)
    lrs = np.array(lrs)
    
    plt.subplot(2, 1, 1)
    plt.stem(lrs, evas)
    plt.title('EVA')
    plt.subplot(2, 1, 2)
    plt.stem(lrs, rmses)
    plt.title('RMSE')
    plt.savefig(os.path.join(FLAGS.exp_dir, 'evas_rmses.png'))


if __name__ == "__main__":
    main(sys.argv)