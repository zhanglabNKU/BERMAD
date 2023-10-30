#!/usr/bin/env python
import torch.utils.data
import numpy as np
import pandas as pd
import random
import pickle
import argparse

from BERMAD import training, testing
from pre_processing import pre_processing, read_cluster_similarity

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# nn parameter
code_dim = 20
batch_size = 50  # batch size for each cluster
base_lr = 1e-3
lr_step = 200  # step decay of learning rates
momentum = 0.9
l2_decay = 5e-5
gamma = 1  # regularization between reconstruction and transfer learning
log_interval = 1
# CUDA
device_id = 0  # ID of GPU to use
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# parameters from command line
parser = argparse.ArgumentParser(description='Training the BERMAD model')
parser.add_argument('-data_folder', type=str, default='./', help='folder for loading data and saving results')
parser.add_argument('-files', nargs='+', default=[], help='file names of different batches')
parser.add_argument('-similarity_thr', type=float, default=0.9, help='similarity threshold for distribution matching')
parser.add_argument('-num_epochs', type=float, default=2000, help='number of training epochs')
parser.add_argument('-alpha', type=float, default=0.3, help='weight for hidden1 layer')
parser.add_argument('-beta', type=float, default=0.3, help='weight for hidden2 layer')
parser.add_argument('-delta', type=float, default=0.3, help='weight for code layer')

plt.ioff()

if __name__ == '__main__':
    # load parameters from command line
    args = parser.parse_args()
    data_folder = args.data_folder
    dataset_file_list = args.files
    cluster_similarity_file = args.data_folder+'metaneighbor.csv'
    code_save_file = args.data_folder + 'code_list.pkl'
    similarity_thr = args.similarity_thr
    num_epochs = args.num_epochs
    dataset_file_list = [data_folder+f for f in dataset_file_list]
    alpha = args.alpha
    beta = args.beta
    delta = args.delta

    pre_process_paras = {'take_log': True, 'standardization': True, 'scaling': True}
    nn_paras = {'code_dim': code_dim, 'batch_size': batch_size, 'num_epochs': num_epochs,
                'base_lr': base_lr, 'lr_step': lr_step,
                'momentum': momentum, 'l2_decay': l2_decay, 'gamma': gamma,
                'cuda': cuda, 'log_interval': log_interval,
                'alpha': alpha, 'beta': beta, 'delta': delta}

    # read data
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    cluster_pairs = read_cluster_similarity(cluster_similarity_file, similarity_thr)
    nn_paras['num_inputs'] = len(dataset_list[0]['gene_sym'])

    # training
    model, loss_total_list, loss_reconstruct_list, loss_transfer_list = training(dataset_list, cluster_pairs, nn_paras)
    plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, data_folder + 'loss.png')

    # save codes
    code_list = testing(model, dataset_list, nn_paras)
    with open(code_save_file, 'wb') as f:
        pickle.dump(code_list, f)

    # back to ".csv"
    codes = np.hstack((code_list[0], code_list[1]))
    if len(dataset_file_list) > 2:
        for i in range(2, len(dataset_file_list)):
            codes = np.hstack((codes, code_list[i]))
    df = pd.DataFrame(codes).transpose()
    df.to_csv(os.path.join(data_folder, "combined.csv"))
