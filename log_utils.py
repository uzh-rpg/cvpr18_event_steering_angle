import logz
import numpy as np

import keras
from keras import backend as K

MIN_LR=0.00001



class MyCallback(keras.callbacks.Callback):
    """
    Customized callback class.

    # Arguments
       filepath: Path to save model.
       period: Frequency in epochs with which model is saved.
       batch_size: Number of images per batch.
    """

    def __init__(self, filepath, period, batch_size, factor = 1.0):
        self.filepath = filepath
        self.period = period
        self.batch_size = batch_size
        self.factor = factor
        self.min_lr = MIN_LR

    def on_epoch_end(self, epoch, logs={}):

        # Save training and validation losses
        logz.log_tabular('steering_loss', logs.get('steering_loss'))
        logz.log_tabular('val_steering_loss', logs.get('val_steering_loss'))
        logz.dump_tabular()

        # Save model every 'period' epochs
        if (epoch+1) % self.period == 0:
            filename = self.filepath + '/model_weights_' + str(epoch) + '.h5'
            print("Saved model at {}".format(filename))
            self.model.save_weights(filename, overwrite=True)

        # Reduce lr in critical conditions
        std_pred = logs.get('pred_std')
        if std_pred < 0.05:
            current_lr = K.get_value(self.model.optimizer.lr)
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')

            new_lr = np.maximum(current_lr * self.factor, self.min_lr)
            if not isinstance(new_lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            K.set_value(self.model.optimizer.lr, new_lr)
            print("Reduced learing rate!\n")

        # Hard mining
        sess = K.get_session()
        mse_function = self.batch_size-(self.batch_size-10)*(
            np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
        self.model.k_mse.load(int(np.round(mse_function)), sess)
