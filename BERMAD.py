import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import ae as models
from mmd import mix_rbf_mmd2
import math
import time
from imblearn.over_sampling import RandomOverSampler

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

imblearnSeed = 0


def training(dataset_list, cluster_pairs, nn_paras):
    """ Training an autoencoder to remove batch effects
    Args:
        dataset_list: list of datasets for batch correction
        cluster_pairs: pairs of similar clusters with weights
        nn_paras: parameters for neural network training

    Returns:
        model1: trained autoencoder1
        model2: trained autoencoder2
        loss_reconstruct_list: list of reconstruction loss
        loss_transfer_list: list of transfer loss
    """
    # load nn parameters
    batch_size = nn_paras['batch_size']
    num_epochs = nn_paras['num_epochs']
    num_inputs = nn_paras['num_inputs']
    cuda = nn_paras['cuda']

    # training data for autoencoder1 and autoencoder2, construct a Dataloader for each cluster
    cluster_loader_dict_list = []
    for i in range(len(dataset_list)):
        cluster_loader_dict = {}
        gene_exp = dataset_list[i]['gene_exp'].transpose()
        cluster_labels = dataset_list[i]['cluster_labels']  # cluster labels do not overlap between datasets
        unique_labels = np.unique(cluster_labels)
        # Random oversampling based on cell cluster sizes
        gene_exp, cluster_labels = RandomOverSampler(random_state=imblearnSeed).fit_resample(gene_exp, cluster_labels)

        # construct Dataloader List
        for j in range(len(unique_labels)):
            idx = cluster_labels == unique_labels[j]
            if cuda:
                torch_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(gene_exp[idx, :]).cuda(), torch.LongTensor(cluster_labels[idx]).cuda())
            else:
                torch_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(gene_exp[idx, :]), torch.LongTensor(cluster_labels[idx]))
            data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True,
                                                      drop_last=True)
            cluster_loader_dict[unique_labels[j]] = data_loader
        cluster_loader_dict_list.append(cluster_loader_dict)

    # create model
    model1 = models.autoencoder1(num_inputs=num_inputs)
    model2 = models.autoencoder2(num_inputs=num_inputs)
    if cuda:
        model1.cuda()
        model2.cuda()

    # training
    loss_total_list = []  # loss of total loss
    loss_reconstruct_list = []
    loss_transfer_list = []
    for epoch in range(1, num_epochs + 1):
        avg_loss, avg_reco_loss, avg_tran_loss = training_epoch(epoch, model1, model2, cluster_loader_dict_list,
                                                                cluster_pairs, nn_paras)
        # terminate early if loss is nan
        if math.isnan(avg_reco_loss) or math.isnan(avg_tran_loss):
            return [], model1, model2, [], [], []
        loss_total_list.append(avg_loss)
        loss_reconstruct_list.append(avg_reco_loss)
        loss_transfer_list.append(avg_tran_loss)

    return model1, model2, loss_total_list, loss_reconstruct_list, loss_transfer_list


def training_epoch(epoch, model1, model2, cluster_loader_dict_list, cluster_pairs, nn_paras):
    """ Training an epoch
    Args:
        epoch: number of the current epoch
        model1: autoencoder1
        model2: autoencoder2
        cluster_loader_dict_list: list of dict of Dataloaders indexed by clusters
        cluster_pairs: pairs of similar clusters with weights
        nn_paras: parameters for neural network training

    Returns:
        avg_total_loss: average total loss of mini-batches
        avg_reco_loss: average reconstruction loss of mini-batches
        avg_tran_loss: average transfer loss of mini-batches
    """
    log_interval = nn_paras['log_interval']
    # load nn parameters
    base_lr = nn_paras['base_lr']
    lr_step = nn_paras['lr_step']
    num_epochs = nn_paras['num_epochs']
    l2_decay = nn_paras['l2_decay']
    gamma = nn_paras['gamma']
    cuda = nn_paras['cuda']
    alpha = nn_params['alpha']
    beta = nn_params['beta']
    delta = nn_params['delta']

    # step decay of learning rate
    learning_rate = base_lr / math.pow(2, math.floor(epoch / lr_step))
    # regularization parameter between two losses
    gamma_rate = 2 / (1 + math.exp(-10 * epoch / num_epochs)) - 1
    gamma = gamma_rate * gamma

    if epoch % log_interval == 0:
        print('{:}, Epoch {}, learning rate {:.3E}, gamma {:.3E}'.format(
            time.asctime(time.localtime()), epoch, learning_rate, gamma))

    optimizer1 = torch.optim.Adam([
        {'params': model1.hidden1.parameters()},
        # {'params': model1.encoder.parameters()},
        # {'params': model1.decoder.parameters()},
        {'params': model1.hidden2.parameters()}
    ], lr=learning_rate, weight_decay=l2_decay)
    optimizer2 = torch.optim.Adam([
        {'params': model2.hidden1.parameters()},
        {'params': model2.encoder.parameters()},
        {'params': model2.decoder.parameters()},
        {'params': model2.hidden2.parameters()}
    ], lr=learning_rate, weight_decay=l2_decay)

    model1.train()
    model2.train()

    iter_data_dict_list = []
    num_iter = 0
    for cluster_loader_dict in cluster_loader_dict_list:
        iter_data_dict = {}
        for cls in cluster_loader_dict:
            iter_data = iter(cluster_loader_dict[cls])
            iter_data_dict[cls] = iter_data
        # use the largest dataset to define an epoch
        for cls in cluster_loader_dict:
            num_iter = max(num_iter, len(cluster_loader_dict[cls]))
        iter_data_dict_list.append(iter_data_dict)

    total_loss = 0
    total_reco_loss = 0
    total_tran_loss = 0
    num_batches = 0

    for it in range(num_iter):
        data_dict1, data_dict2 = {}, {}
        label_dict1, label_dict2 = {}, {}
        code_dict1, code_dict2 = {}, {}
        reconstruct_dict1, reconstruct_dict2 = {}, {}
        hidden1_dict1, hidden1_dict2 = {}, {}
        hidden2_dict1, hidden2_dict2 = {}, {}
        for iter_data_dict in iter_data_dict_list:
            if iter_data_dict_list.index(iter_data_dict) == 0:
                for cls in iter_data_dict:
                    # data, labels = iter_data_dict[cls].next() 版本问题
                    data, labels = next(iter(iter_data_dict[cls]))
                    data_dict1[cls] = data
                    label_dict1[cls] = labels
                    if it % len(cluster_loader_dict_list[0][cls]) == 0:
                        iter_data_dict[cls] = iter(cluster_loader_dict_list[0][cls])
                    data_dict1[cls] = Variable(data_dict1[cls])
                    label_dict1[cls] = Variable(label_dict1[cls])
            else:
                for cls in iter_data_dict:
                    # data, labels = iter_data_dict[cls].next() 版本问题
                    data, labels = next(iter(iter_data_dict[cls]))
                    data_dict2[cls] = data
                    label_dict2[cls] = labels
                    if it % len(cluster_loader_dict_list[1][cls]) == 0:
                        iter_data_dict[cls] = iter(cluster_loader_dict_list[1][cls])
                    data_dict2[cls] = Variable(data_dict2[cls])
                    label_dict2[cls] = Variable(label_dict2[cls])

        for cls in data_dict1:
            code, reconstruct, hidden1, hidden2 = model1(data_dict1[cls])
            code_dict1[cls] = code
            reconstruct_dict1[cls] = reconstruct
            hidden1_dict1[cls] = hidden1
            hidden2_dict1[cls] = hidden2
        for cls in data_dict2:
            code, reconstruct, hidden1, hidden2 = model2(data_dict2[cls])
            code_dict2[cls] = code
            reconstruct_dict2[cls] = reconstruct
            hidden1_dict2[cls] = hidden1
            hidden2_dict2[cls] = hidden2

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # transfer loss for cluster pairs in cluster_pairs matrix
        loss_transfer = torch.FloatTensor([0])
        if cuda:
            loss_transfer = loss_transfer.cuda()
        for i in range(cluster_pairs.shape[0]):
            cls_1 = int(cluster_pairs[i, 0])
            cls_2 = int(cluster_pairs[i, 1])
            if cls_1 not in code_dict1 or cls_2 not in code_dict2:
                continue
            mmd2_D = mix_rbf_mmd2(code_dict1[cls_1], code_dict2[cls_2], sigma_list) * delta + \
                     mix_rbf_mmd2(hidden1_dict1[cls_1], hidden1_dict2[cls_2], sigma_list) * alpha + \
                     mix_rbf_mmd2(hidden2_dict1[cls_1], hidden2_dict2[cls_2], sigma_list) * beta
            loss_transfer += mmd2_D * cluster_pairs[i, 2]

        # reconstruction loss for all clusters
        loss_reconstruct = torch.FloatTensor([0])
        if cuda:
            loss_reconstruct = loss_reconstruct.cuda()
        for cls in data_dict1:
            loss_reconstruct += F.mse_loss(reconstruct_dict1[cls], data_dict1[cls])
        for cls in data_dict2:
            loss_reconstruct += F.mse_loss(reconstruct_dict2[cls], data_dict2[cls])

        loss = loss_reconstruct + gamma * loss_transfer

        loss.backward()
        optimizer1.step()
        optimizer2.step()

        # update total loss
        num_batches += 1
        total_loss += loss.data.item()
        total_reco_loss += loss_reconstruct.data.item()
        total_tran_loss += loss_transfer.data.item()

    avg_total_loss = total_loss / num_batches
    avg_reco_loss = total_reco_loss / num_batches
    avg_tran_loss = total_tran_loss / num_batches

    if epoch % log_interval == 0:
        print('Avg_loss {:.3E}\t Avg_reconstruct_loss {:.3E}\t Avg_transfer_loss {:.3E}'.format(
            avg_total_loss, avg_reco_loss, avg_tran_loss))
    return avg_total_loss, avg_reco_loss, avg_tran_loss


def testing(model1, model2, dataset_list, nn_params):
    """ Training an epoch
    Args:
        model1: autoencoder1
        model2: autoencoder2
        dataset_list: list of datasets for batch correction
        nn_params: parameters for neural network training

    Returns:
        code_list: list of embedded codes
    """

    # load nn parameters
    cuda = nn_params['cuda']

    data_loader_list = []
    num_cells = []
    for dataset in dataset_list:
        torch_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset['gene_exp'].transpose()), torch.LongTensor(dataset['cell_labels']))
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=len(dataset['cell_labels']),
                                                  shuffle=False)
        data_loader_list.append(data_loader)
        num_cells.append(len(dataset['cell_labels']))

    model1.eval()
    model2.eval()

    code_list = []  # list of embedded codes
    for i in range(len(data_loader_list)):
        idx = 0
        with torch.no_grad():
            for data, labels in data_loader_list[i]:
                if cuda:
                    data, labels = data.cuda(), labels.cuda()
                code_tmp, _, _, _ = model1(data) if i == 0 else model2(data)
                code_tmp = code_tmp.cpu().numpy()
                if idx == 0:
                    code = np.zeros((code_tmp.shape[1], num_cells[i]))
                code[:, idx:idx + code_tmp.shape[0]] = code_tmp.T
                idx += code_tmp.shape[0]
        code_list.append(code)

    return code_list
