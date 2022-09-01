# %%
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime

from model import Model
from dataset import DataLoader
from visual import visual, visual_for_real
    
# %%
def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    # See Bivariate case in multivariate normal distribution:
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution.
    normx = tf.math.subtract(x, mux)
    normy = tf.math.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.math.multiply(sx, sy)
    # Calculate the exponential factor
    z = tf.math.square(tf.math.divide(normx, sx)) + tf.math.square(tf.math.divide(normy, sy)) - \
        2*tf.math.divide(tf.math.multiply(rho, tf.math.multiply(normx, normy)), sxsy)
    
    negRho = 1 - tf.math.square(rho)
    # Numerator
    result = tf.math.exp(tf.math.divide(-z, 2*negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.math.multiply(sxsy, tf.math.sqrt(negRho))
    # Final PDF calculation
    result = tf.math.divide(result, denom) # Tensor(batch_size, seq_length, 1)

    return result

def get_coef(output):
    """generate mu, sigma, rho from the output of RNN model

    Args:
        output (Tensor): output of RNN model : Tensor(batch_size, seq_length, 1)

    Returns:
        tuple (mu_x, mu_y, sigma_x, sigma_y, rho) : Tensor(batch_size, seq_length, 1) 
    """
    z = output

    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, -1) # 5 is output_size
    # z_mux, z_muy, z_sx, z_sy, z_corr : (batch_size, seq_length, 1)

    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]

def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    """loss function of the RNN model, 
       Sum of negative log-likelihood estimates of all predicted trajectory points

    Args:
        z_mux (Tensor): RNN unit output mu_x, Tensor(batch_size, seq_length, 1)
        z_muy (Tensor): RNN unit output mu_y, Tensor(batch_size, seq_length, 1)
        z_sx (Tensor):  RNN unit output sigma_x, Tensor(batch_size, seq_length, 1)
        z_sy (Tensor):  RNN unit output sigma_y, Tensor(batch_size, seq_length, 1)
        z_corr (Tensor): RNN unit output rho, Tensor(batch_size, seq_length, 1)
        x_data (Tensor): groundtruth x, Tensor(batch_size, seq_length, 1)
        y_data (Tensor): groundtruth y, Tensor(batch_size, seq_length, 1)

    Returns:
        float: loss
    """
    # z_mux, z_muy, z_sx, z_sy, z_corr : output results
    result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

    epsilon = 1e-20

    result1 = -tf.math.log(tf.math.maximum(result0, epsilon))  # Numerical stability

    return tf.reduce_sum(result1)

def get_mean_error(pred_traj, true_traj, observed_length):
    """Compute ADE:
    The sum of distances between all predicted points and the 
    GroundTruth points / number of predicted trajectory points

    Args:
        pred_traj (_type_): _description_
        true_traj (_type_): _description_
        observed_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = pred_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)

def get_final_error(pred_traj, true_traj):
    """Compute FDE
    The distance between the last predicted point and the corresponding GroundTruth point

    Args:
        pred_traj (_type_): _description_
        true_traj (_type_): _description_

    Returns:
        _type_: _description_
    """

    error = np.linalg.norm(pred_traj[-1, :] - true_traj[-1, :])

    # Return the mean error
    return error

def sample_gaussian_2d(mux, muy, sx, sy, rho):
    """sample pred pos (x,y) based on mu, sigma, rho

    Returns:
        pred_x, pred_y: float, float
    """
    # Extract mean
    mean = [mux, muy]

    # Extract covariance matrix
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def test(args, IS_VISUALIZE):
    checkpoint_dir = './training_checkpoints'

    # Dataset to get data from
    dataset = [args.test_dataset]

    # Initialize the dataloader object to
    # Get sequences of length obs_length+pred_length
    data_loader = DataLoader(1, args.pred_length + args.obs_length, dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    tf.train.latest_checkpoint(checkpoint_dir)

    args.batch_size = 1

    test_model = build_model(args) # Model(args)

    test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    test_model.build(tf.TensorShape([1, None,  2]))

    # Maintain the total_error until now
    total_error = 0
    counter = 0
    final_error = 0.0

    truth_trajs = []
    pred_trajs = []
    gauss_params = []

    for batch_id in range(data_loader.num_batches):
        # Get the source, target data for the next batch
        batch, batch_next = data_loader.next_batch() # list[array(seq_length,2)] len(list)=batch_size

        base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch]) # array(batch_size, seq_length, 2)
        batch = batch - base_pos # array(batch_size, seq_length, 2)

        # The observed part of the trajectory
        obs_observed_traj = batch[0][:args.obs_length] # array(obs_length, 2)
        obs_observed_traj = tf.expand_dims(obs_observed_traj, 0) # Tensor(1, obs_length, 2)

        complete_traj = batch[0][:args.obs_length] # array(obs_length, 2)

        test_model.reset_states()

        # test_model.initial_state = None
        gauss_param = np.array([])

        for idx in range(args.pred_length):
            tensor_batch = tf.convert_to_tensor(obs_observed_traj) # Tensor(1, obs_length or 1, 2)

            logits = test_model(tensor_batch) # Tensor(1, obs_length or 1, output_size)

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits) # Tensor(1, obs_length or 1, 1)

            next_x, next_y = sample_gaussian_2d(o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]) # float

            obs_observed_traj = tf.expand_dims([[next_x, next_y]], 0) # Tensor(1, obs_length or 1, 2)

            if len(gauss_param) <=0:
                gauss_param = np.array([o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]])
            else:
                gauss_param = np.vstack((gauss_param, [o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]])) # finally (pred_length, output_size)


            complete_traj = np.vstack((complete_traj, [next_x, next_y])) # from (obs_length, 2) to (seq_length, 2)

        total_error += get_mean_error (complete_traj + base_pos[0], batch[0] + base_pos[0], args.obs_length)
        final_error += get_final_error(complete_traj + base_pos[0], batch[0] + base_pos[0])

        pred_trajs.append(complete_traj)
        truth_trajs.append(batch[0])
        gauss_params.append(gauss_param)

        print("Processed trajectory number: {} out of {} trajectories".format(batch_id, data_loader.num_batches))

    # Print the mean error across all the batches
    print("Total mean error of the model is {}".format(total_error/data_loader.num_batches))
    print("Total final error of the model is {}".format(final_error/data_loader.num_batches))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_file = "./results/pred_results_"+current_time+".pkl"
    f = open(data_file, "wb")
    pickle.dump([pred_trajs, truth_trajs, gauss_params], f)
    f.close()
    
    if IS_VISUALIZE:
        visual(data_file)
        

def test_real(args, test_data, IS_VISUALIZE):
    """_summary_

    Args:
        args (dict): _description_
        test_data (list[array]): list[array(obs_length,2)], len(list)=num_test_traj
        IS_VISUALIZE (bool): whether to visulize the prediction results
    """
    
    # load test model weights
    checkpoint_dir = './training_checkpoints'
    tf.train.latest_checkpoint(checkpoint_dir)

    args.batch_size = len(test_data)

    test_model = build_model(args) # Model(args)
    test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    test_model.build(tf.TensorShape([args.batch_size, None,  2])) # original: tf.TensorShape([1, None,  2])
    
    pred_trajs, truth_trajs = [], []

    # start prediction
    
    # Get the source, target data for the next batch
    batch = test_data # list[array(obs_length,2)] len(list)=batch_size

    base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch]) # array(batch_size, obs_length, 2)
    batch = batch - base_pos # array(batch_size, obs_length, 2)

    test_model.reset_states()
    # test_model.initial_state = None
    
    # The observed part of the trajectory
    obs_observed_traj = tf.convert_to_tensor(test_data, dtype=tf.float32) # Tensor(batch_size, obs_length, 2)

    complete_traj = np.empty([args.batch_size, args.seq_length, 2])     
    complete_traj[:,:args.obs_length,:] = batch[:,:args.obs_length,:] # array(batch_size, obs_length, 2)

    gauss_param = np.empty([args.batch_size, args.pred_length, 5])      
 
    for pred_id in range(args.pred_length):
        tensor_batch = obs_observed_traj # Tensor(batch_size, obs_length or 1, 2)

        logits = test_model(tensor_batch) # Tensor(batch_size, obs_length or 1, output_size)

        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits) # Tensor(batch_size, obs_length or 1, 1)
        
        next_pos = []
        for traj_id in range(args.batch_size): # here, arg.batch_size is also num_traj = len(test_data)
            next_x, next_y = sample_gaussian_2d(o_mux[traj_id][-1][0], o_muy[traj_id][-1][0], o_sx[traj_id][-1][0], o_sy[traj_id][-1][0], o_corr[traj_id][-1][0]) # float

            gauss_param[traj_id, pred_id, :] = np.array([o_mux[traj_id][-1][0], o_muy[traj_id][-1][0], o_sx[traj_id][-1][0], o_sy[traj_id][-1][0], o_corr[traj_id][-1][0]]) # (1, 5, 1)

            complete_traj[traj_id, args.obs_length+pred_id, :] = np.array([next_x, next_y])
            
        obs_observed_traj = tf.convert_to_tensor(complete_traj[:, args.obs_length:args.obs_length+pred_id+1, :]) # Tensor(batch_size, 1, 2)
    
    if IS_VISUALIZE:
        visual_for_real(complete_traj, gauss_param)

def build_model(args):
    """Embedding layer, LSTM/GRU layer, Output layer
    Embedding层将坐标(x,y)嵌入到64维的向量空间
    输出层输出每个预测点的二维高斯分布参数(包含5个参数:mux, muy, sx, sy, corr), 
    """
    output_size = 5 # mux, muy, sx, sy, corr
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(args.embedding_size, 
                              activation = tf.keras.activations.relu,
                              batch_input_shape = [args.batch_size, None, 2]),
        tf.keras.layers.GRU(args.rnn_size, 
                            return_sequences=True, # return the last output, or the full sequence
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(output_size)
    ])

    return model

def calc_prediction_error(mux, muy, sx, sy, corr, offset_positions, args):

    traj_nums = mux.shape[0] # batch_size

    pred_nums = mux.shape[1] # seq_length

    mean_error = 0.0
    final_error = 0.0
    for index in range(traj_nums):
        pred_traj = np.zeros((pred_nums, 2))
        for pt_index in range(pred_nums):
            next_x, next_y = sample_gaussian_2d(mux[index][pt_index][0],
                            muy[index][pt_index][0], sx[index][pt_index][0],
                            sy[index][pt_index][0], corr[index][pt_index][0])

            pred_traj[pt_index][0] = next_x
            pred_traj[pt_index][1] = next_y

        mean_error += get_mean_error(pred_traj, offset_positions[index], args.obs_length)
        final_error += get_final_error(pred_traj, offset_positions[index])

    mean_error = mean_error / traj_nums
    final_error = final_error / traj_nums

    return mean_error, final_error

def train(args):
    datasets = list(range(4))

    data_loader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)

    # Create a Vanilla LSTM model with the arguments
    model = build_model(args)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    optimizer = tf.keras.optimizers.RMSprop(args.learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # 检查点保存至的目录
    checkpoint_dir = './training_checkpoints'
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()
        model.reset_states() 
        # reset_states clears only the hidden states of your network. 
        # if set stateful=True:
        #   - should call reset_states every time for model calls independent
        # If not set:
        #   - all states are automatically reset after every batch computations 
        #     (so e.g. after calling fit, predict and evaluate also). 

        for batch_id in range(data_loader.num_batches):
            start = time.time()

            batch, next_batch = data_loader.next_batch()
            # len(x)=batch_size; len(y)=batch_size. x, y : list[ array(seq_length,2) ]

            base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch])
            # ele_batch: 2Darray(seq_length,2) ; ele_batch[0] 重复 seq_length, for all ele_batch.
            # base_pos = array(batch_size, seq_length, 2)

            batch_offset = batch - base_pos # array(batch_size, seq_length, 2)
            next_batch_offset = next_batch - base_pos # array(batch_size, seq_length, 2)

            with tf.GradientTape() as tape:
                tensor_batch = tf.convert_to_tensor(batch_offset, dtype=tf.float32)

                logits = model(tensor_batch) # logits: array(batch_size, seq_length, output_size)

                 # output -- mux, muy, sx, sy,corr : Tensor(batch_size, seq_length, 1)
                [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

                tensor_next_batch = tf.convert_to_tensor(next_batch_offset, dtype=tf.float32)
                
                # x_data, y_data in next_batch : Tensor(batch_size, seq_length, 1)
                [x_data, y_data] = tf.split(tensor_next_batch, 2, -1)

                # Compute the loss function
                loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data) # loss:float

                mean_error, final_error = calc_prediction_error(o_mux, o_muy, o_sx, o_sy, o_corr, tensor_next_batch, args)

                loss = tf.math.divide(loss, (args.batch_size * args.seq_length))

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.lr.assign(args.learning_rate * (args.decay_rate ** e))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss(loss)

            end = time.time()
            # Print epoch, batch, loss and time taken
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, mean error = {}, final_error = {}"
                    .format(e * data_loader.num_batches + batch_id,
                            args.num_epochs * data_loader.num_batches,
                            e, loss, end - start, mean_error, final_error))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        model.save_weights(checkpoint_prefix.format(epoch=e))    

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    parser.add_argument('--test_dataset', type=int, default=4,
                        help='Dataset to be tested on')
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')

    # args = parser.parse_args(args=[])
    args = parser.parse_args(args=['--batch_size', '70', '--num_epochs', '400', 
                                   '--seq_length', '20', '--obs_length', '14', '--pred_length', '6'])
    
    # train and set
    # train(args)
    # test(args, IS_VISUALIZE=True)

    test_data = [np.random.random_sample((args.seq_length,2)), 
                 np.random.random_sample((args.seq_length,2)), 
                 np.random.random_sample((args.seq_length,2))]
    test_real(args, test_data, IS_VISUALIZE=True)
    
# %%

