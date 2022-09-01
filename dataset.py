import os
import pickle
import numpy as np
import random

class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        # List of data directories where raw data resides
        self.data_dirs = ['./data/eth/univ', './data/eth/hotel',
                         './data/ucy/zara/zara01', './data/ucy/zara/zara02',
                         './data/ucy/univ']
        # self.data_dirs = ['./data/eth/univ', './data/eth/hotel']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist already or if forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.preprocess(self.used_data_dirs, data_file)

        # Load the data from the pickled file
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        """generate trajectories.cpkl for every pedestrain in all datasets.

        Args:
            data_dirs (list[str]): list of the used dataset directories
            data_file (str): trajectories.cpkl to store all trajectries for all pedestrains
        """
        all_ped_data = {}
        dataset_indices = []
        current_ped = 0
        # For each dataset
        for directory in data_dirs:
            # Define the path to its respective csv file
            file_path = os.path.join(directory, 'pixel_pos.csv')

            print("processing {}".format(file_path))

            # Load data from the csv file
            # Data is a 4 x numTrajPoints matrix
            # where each column is a (frameId, pedId, y, x) vector
            data = np.genfromtxt(file_path, delimiter=',') # 2D-array (4, num_points)

            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(data[1, :]))

            # For each pedestrian in the dataset
            for ped in range(1, numPeds+1):
                # Extract trajectory of the current ped
                traj = data[:, data[1, :] == ped] 
                # Format it as (x, y, frameId), 
                traj = traj[[3, 2, 0], :] # 2D array: (3, num_point_id)

                # Store this in the dictionary
                all_ped_data[current_ped + ped] = traj # dict( ped_id = array(3, num_traj) )

            # Current dataset done
            dataset_indices.append(current_ped+numPeds) # list (numPeds)
            current_ped += numPeds

            print("total ped nums: {}".format(numPeds))

        # The complete data is a tuple of all pedestrian data, and dataset ped indices
        complete_data = (all_ped_data, dataset_indices)
        # Store the complete data into the pickle file
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        """load trajectories.cpkl, choose trajectories which are longer than seq_length+2, and compute num_batches. 
           generate self.data:list[2D-array(num_traj, 2)] (only x,y, throw the frame_id info)

        Args:
            data_file (str): dictory of trajectories.cpkl
        """
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # Get the pedestrian data from the pickle file
        all_ped_data = self.raw_data[0] # dict: key is ped_id, value is pos
        # Not using dataset_indices for now
        # dataset_indices = self.raw_data[1]

        # Construct the data with sequences(or trajectories) longer than seq_length
        self.data = []
        counter = 0

        # For each pedestrian in the data
        for ped in all_ped_data:
            # Extract his trajectory
            traj = all_ped_data[ped] # (3, num_traj_ped_id) -< 3: x,y,frame_id
            # If the length of the trajectory is greater than seq_length (+2 as we need both source and target data)
            # 太短的traj不要了
            if traj.shape[1] > (self.seq_length+2):
                # TODO: (Improve) Store only the (x,y) coordinates for now
                self.data.append(traj[[0, 1], :].T) # (num_traj_ped_id, 2) -< 2: x,y
                # self.data: list[ array(num_traj_ped_id, 2) ]
                # Number of batches this datapoint is worth
                counter += int(traj.shape[1] / ((self.seq_length+2))) # 该ped的traj的长度 有几段 seq——len的长度

        print("all ped data len: {}, seq length: {}".format(len(all_ped_data), self.seq_length)) # 这里len(all_ped_data)应该是所有的num——ped，没有筛选过短的traj
        # Calculate the number of batches (each of batch_size) in the data
        self.num_batches = int(counter / self.batch_size) 
        # 有counter段长于seq-len的轨迹，每个batch的size是batch_size，一共有num_batches个batch

    def next_batch(self):
        """Function to get the next batch of points

        Returns:
            list[ array(seq_length,2) ], list[ array(seq_length,2) ]: corresponding one seg and next one seg. 
            y_batch中的traj是x_batch中的traj的下一步. len(x_batch)=batch_size, len(y_batch)=batch_size
        """ 
        # List of source and target data for the current batch
        x_batch = []
        y_batch = []
        # For each sequence in the batch
        for i in range(self.batch_size):
            # Extract the trajectory of the pedestrian pointed out by self.pointer
            traj = self.data[self.pointer]
            # Number of sequences corresponding to his trajectory
            n_batch = int(traj.shape[0] / (self.seq_length+2)) # one ped's traj contains n_batch segments which are longer than seq_len+2
            # Randomly sample a index from which his trajectory is to be considered
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            # Append the trajectory from idx until seq_length into source and target data
            x_batch.append(np.copy(traj[idx:idx+self.seq_length, :])) # list[ array(seq_length,2) ]
            y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :])) # next array traj

            if random.random() < (1.0/float(n_batch)):
                # Adjust sampling probability
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()

        return x_batch, y_batch

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset the data pointer
        '''
        self.pointer = 0