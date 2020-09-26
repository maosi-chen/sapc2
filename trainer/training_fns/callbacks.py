import os, sys
import glob
import posixpath
import tensorflow as tf
from tensorflow.python.lib.io import file_io


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(posixpath.join(job_dir, file_path), mode='wb') as output_f:  # 'wb+'
            output_f.write(input_f.read())

def copy_file_from_gcs(job_dir, file_path):
    with file_io.FileIO(posixpath.join(job_dir, file_path), mode='rb') as input_f:
        with file_io.FileIO(file_path, mode='wb') as output_f:  # 'wb+'
            output_f.write(input_f.read())


# a workaround callback to move locally saved checkpoints to the gs bucket
class ExportCheckpointGS(tf.keras.callbacks.Callback):
    """
    """
    def __init__(self,
                 job_dir
                 ):
        self.job_dir = job_dir
        # self.ckpt_local_file_name = ckpt_local_file_name

    def get_checkpoint_FSNs(self):
        model_path_glob = 'weights.*'
        if not self.job_dir.startswith('gs://'):
            model_path_glob = os.path.join(self.job_dir, model_path_glob)
        checkpoints = glob.glob(model_path_glob)
        return checkpoints

    def on_epoch_begin(self, epoch, logs={}):
        """Compile and save model."""
        if epoch > 0:
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            # model_path_glob = 'weights.*'
            # if not self.job_dir.startswith('gs://'):
            #  model_path_glob = os.path.join(self.job_dir, model_path_glob)
            # checkpoints = glob.glob(model_path_glob)
            checkpoints = self.get_checkpoint_FSNs()
            if len(checkpoints) > 0:
                checkpoints.sort()
                if self.job_dir.startswith('gs://'):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])

    def on_train_end(self, logs={}):
        checkpoints = self.get_checkpoint_FSNs()
        if len(checkpoints) > 0:
            checkpoints.sort()
            if self.job_dir.startswith('gs://'):
                copy_file_to_gcs(self.job_dir, checkpoints[-1])

                ## copy the checkpoint file (list of available checkpoints in the current directory)
                # copy_file_to_gcs(self.job_dir, 'checkpoint')


class TriCyclicalLearningRateDecayCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            initial_learning_rate,
            maximal_learning_rate,
            step_size,
            steps_per_epoch,
            scale_mode="cycle",
            name="TriangularCyclicalLearningRate",
    ):
        super().__init__()
        self.initial_learning_rate = tf.convert_to_tensor(
            initial_learning_rate, dtype=tf.float32, name="initial_learning_rate"
        )
        self.maximal_learning_rate = tf.convert_to_tensor(
            maximal_learning_rate, dtype=tf.float32, name="maximal_learning_rate"
        )
        #self.initial_range = self.maximal_learning_rate - self.initial_learning_rate
        self.step_size = tf.convert_to_tensor(
            step_size, dtype=tf.float32
        )
        self.steps_per_epoch = tf.convert_to_tensor(
            steps_per_epoch, dtype=tf.float32
        )

        # lr_decay_segments = lr_decay_base_eps - 1
        self.lr_decay_base_eps = 20.0 #10.0
        self.lr_decay_rate_per_base_eps = 3.0
        self.lr_decay_segments = self.lr_decay_base_eps - 1.0
        ## assuming equal decay interval in log_p (any p) space,
        ## then in the original space, the ratio between any 2 nearby elements is constant.
        ## Proof:
        ##   let log_p(V(i+1)) = log_p(V(i)) + log_decay_interv,
        ##   then V(i+1)/V(i) = p^log_decay_interval. (which is a const.)
        ## So in order to decay the start lr to end lr in lr_decay_segments,
        ##   Pi(k=1,lr_decay_segments)(lr_decay_multiplier_per_ep)=lr_decay_rate_per_base_eps^-1
        ##   -> lr_decay_multiplier_per_ep^lr_decay_segments=lr_decay_rate_per_base_eps^-1
        ##   -> log_p(lr_decay_multiplier_per_ep)=log_p(lr_decay_rate_per_base_eps^(-1/lr_decay_segments))
        ##   -> lr_decay_multiplier_per_ep = lr_decay_rate_per_base_eps^(-1/lr_decay_segments)
        ##   -> lr_decay_multiplier_per_ep = lr_decay_rate_per_base_eps^(-1/(lr_decay_base_eps-1))
        ##   i.e., decay = r^(-1/(n-1)),
        ## Example 1: r=3.0, n=10.0 -> decay = 0.8850881520714603
        self.lr_decay_multiplier_per_ep = pow(self.lr_decay_rate_per_base_eps, -1.0/self.lr_decay_segments)
        self.lr_decay_multiplier_per_ep_ts = tf.convert_to_tensor(self.lr_decay_multiplier_per_ep,
                                                                  dtype=tf.float32)

        # self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.name = name
    
    def on_epoch_begin(self, epoch, logs=None):
        # self.ep_maximal_learning_rate = tf.case([
        #     (tf.less(epoch, 3), lambda: tf.add(tf.divide(self.initial_range, 1.0), self.initial_learning_rate)),
        #     (tf.less(epoch, 7), lambda: tf.add(tf.divide(self.initial_range, 2.0), self.initial_learning_rate)),
        #     (tf.less(epoch, 10), lambda: tf.add(tf.divide(self.initial_range, 4.0), self.initial_learning_rate)),
        #     (tf.greater_equal(epoch, 10), lambda: tf.add(tf.divide(self.initial_range, 8.0), self.initial_learning_rate))
        # ])
        # ep_lr_range = self.ep_maximal_learning_rate - self.initial_learning_rate
        epoch_ts = tf.convert_to_tensor(epoch, dtype=tf.float32)
        decay_multiplier_cur_ep = tf.math.pow(self.lr_decay_multiplier_per_ep_ts, epoch_ts)
        self.ep_maximal_learning_rate = self.maximal_learning_rate * decay_multiplier_cur_ep
        self.ep_minimal_learning_rate = self.initial_learning_rate * decay_multiplier_cur_ep
        ep_lr_range = self.ep_maximal_learning_rate - self.ep_minimal_learning_rate
        
        batches_OneEp = tf.range(self.steps_per_epoch, dtype=tf.float32)
        cycles_OneEp = tf.floor(1.0 + batches_OneEp / (2.0 * self.step_size))
        #x_OneEp = tf.abs(batches_OneEp / self.step_size)
        x_OneEp = tf.abs(batches_OneEp / self.step_size - 2.0 * cycles_OneEp + 1)
        One_Minus_x_OneEp = tf.maximum(tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32) - x_OneEp)
        
        #self.lr_OneEp = self.initial_learning_rate + ep_lr_range * One_Minus_x_OneEp
        self.lr_OneEp = self.ep_minimal_learning_rate + ep_lr_range * One_Minus_x_OneEp
    
    def on_train_batch_begin(self, batch, logs=None):
        # cycle = tf.floor(1 + batch / (2 * self.step_size))
        # x = tf.abs(batch / self.step_size - 2 * cycle + 1)

        # mode_step = cycle if self.scale_mode == "cycle" else batch
        # mode_step = cycle
        
        # batch_lr = self.initial_learning_rate + \
        #   self.ep_lr_range * tf.maximum(tf.cast(0, tf.float32), (1 - x))
        
        # []
        batch_lr = tf.gather(self.lr_OneEp, indices=batch, axis=0)
        
        tf.keras.backend.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(batch_lr))
        
        return
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        logs['lr'] = tf.keras.backend.get_value(self.ep_maximal_learning_rate)


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print ('\nLearning rate for epoch {} is {}'.format(
            epoch + 1, tf.keras.backend.get_value(self.model.optimizer.lr)))
        # epoch + 1, tf.keras.backend.get_value(fn_MPR_Model.optimizer.lr)))

# callback to print model's regularization loss and Unmasked BP-PSConv MSE loss at the begin/end of an epoch
#class PrintRegularizationLoss(tf.keras.callbacks.Callback):
class PrintModelLoss(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.reg_loss = []

    def get_current_reg_loss(self):
        reg_loss_ts_list = []
        add_loss_ts_list = []
        for iL in self.model.losses:
            try:
                tmp = iL.numpy()
                reg_loss_ts_list.append(iL)
            except:
                add_loss_ts_list.append(iL)
        reg_loss_sum_ts = tf.math.add_n(reg_loss_ts_list)
        #add_loss_sum_ts = tf.math.add_n(add_loss_ts_list)

        reg_loss_sum_val = tf.keras.backend.get_value(reg_loss_sum_ts)

        return reg_loss_sum_val #, add_loss_sum_ts

        #reg_loss_ts = tf.math.add_n(self.model.losses)
        #print("reg_loss_ts", reg_loss_ts)
        #tf.print("[tf.print] reg_loss_ts", reg_loss_ts)
        #return reg_loss_ts
        #reg_loss_val = tf.keras.backend.get_value(reg_loss_ts)
        #return reg_loss_val

    def on_train_begin(self, logs={}):
        #reg_loss_sum_val, add_loss_sum_ts = self.get_current_reg_loss()
        reg_loss_sum_val = self.get_current_reg_loss()
        self.reg_loss.append(reg_loss_sum_val)
        #self.reg_loss.append(self.get_current_reg_loss())
        print("regularization loss (on_train_begin): {}".format(self.reg_loss[-1]))
        #print("logs", logs)
        #tf.print("Model internal loss (on_train_begin): {}".format(self.reg_loss[-1]))

    def on_epoch_end(self, epoch, logs={}):
        #self.reg_loss.append(self.get_current_reg_loss())
        reg_loss_sum_val = self.get_current_reg_loss()
        self.reg_loss.append(reg_loss_sum_val)
        print("regularization loss (epoch {}): {}".format(epoch, self.reg_loss[-1]))
        #print("add_loss_sum_ts", add_loss_sum_ts)
        #print("logs", logs)



