from __future__ import print_function

import numpy as np
import random
import math
import os

import torch
import torchvision
import torch.utils.data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from process_esol_data import pad_arrays_and_create_masks as p_esol

def get_shapes(dataset):
    """
    Get the input and output shapes of the data examples for each dataset used.
    dataset: name of the dataset used
    """
    if dataset == 'HAR':
        num_inputs = 561
        num_outputs = 6
        num_labels = 6
    elif dataset == 'ESOL':
        num_inputs = 9
        num_outputs = 1
        num_labels = None
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def load_data(dataset, seed):
    """
    Load the dataset from the drive.
    The har datasets need to be downloaded first with the provided scripts in /data.
    dataset: name of the dataset
    seed: seed for randomness
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset == 'ESOL':
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html#torch_geometric.datasets.MoleculeNet
        esol_data = MoleculeNet(root=".", name="ESOL")
        train_loader = DataLoader(
            esol_data[:int(len(esol_data) * 0.8)],
            batch_size=64, # for assign_data() to work, it should be 1
            shuffle=True
        )
        test_loader = DataLoader(
            esol_data[int(len(esol_data) * 0.8):],
            batch_size=64, # for assign_data() to work, it should be 1
            shuffle=False
        )
        return train_loader, test_loader

    elif dataset == 'HAR':
        train_dir = os.path.join("data", "HAR", "train", "")
        test_dir = os.path.join("data", "HAR", "test", "")

        file = open(train_dir + "X_train.txt", 'r')
        X_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(train_dir + "y_train.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "X_test.txt", 'r')
        X_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(test_dir + "y_test.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        # Loading which datapoint belongs to which client
        file = open(train_dir + "subject_train.txt", 'r')
        train_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "subject_test.txt", 'r')
        test_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        y_train, y_test, X_train, X_test = [], [], [], []

        clients = np.concatenate((train_clients, test_clients))
        for client in range(1, 31):
            mask = tuple([clients == client])
            x_client = X[mask]
            y_client = y[mask]

            split = np.concatenate((np.ones(int(np.ceil(0.75*len(y_client))), dtype=bool), np.zeros(int(np.floor(0.25*len(y_client))), dtype=bool)))
            np.random.shuffle(split)  # Generate mask for train test split with ~0.75 1
            x_train_client = x_client[split]
            y_train_client = y_client[split]
            x_test_client = x_client[np.invert(split)]
            y_test_client = y_client[np.invert(split)]

            # Attach vector of client id to training data for data assignment in assign_data()
            x_train_client = np.insert(x_train_client, 0, client, axis=1)
            if len(X_train) == 0:
                X_train = x_train_client
                X_test = x_test_client
                y_test = y_test_client
                y_train = y_train_client
            else:
                X_train = np.append(X_train, x_train_client, axis=0)
                X_test = np.append(X_test, x_test_client, axis=0)
                y_test = np.append(y_test, y_test_client)
                y_train = np.append(y_train, y_train_client)

        tensor_train_X = torch.tensor(X_train, dtype=torch.float32)
        tensor_test_X = torch.tensor(X_test, dtype=torch.float32)
        tensor_train_y = torch.tensor(y_train, dtype=torch.int64) - 1
        tensor_test_y = torch.tensor(y_test, dtype=torch.int64) - 1
        train_dataset = torch.utils.data.TensorDataset(tensor_train_X, tensor_train_y)
        test_dataset = torch.utils.data.TensorDataset(tensor_test_X, tensor_test_y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
        return train_loader, test_loader
    else:
        raise NotImplementedError



def assign_shared_data(train_data, n_clients):
    """
    Assign the same data to the clients and server because the dataset is too small that we cant split the dataset.
    """
    server_size = len(train_data)

    total_data_x = []
    total_data_edge_index = []
    total_batches = []
    total_labels = []

    for index, data_batch in enumerate(train_data):
        data_x, data_edge_index, label, batch = data_batch.x, data_batch.edge_index, data_batch.y, data_batch.batch
        total_data_x.append(data_x)
        total_data_edge_index.append(data_edge_index)
        total_batches.append(batch+index)
        total_labels.append(label)

    server_data_x = total_data_x[:server_size]
    server_data_edge_index = total_data_edge_index[:server_size]
    server_label = total_labels[:server_size]
    server_batch = total_batches[:server_size]

    server_data_x = torch.cat(server_data_x, dim=0)
    server_data_edge_index = torch.cat(server_data_edge_index, dim=1)
    server_label = torch.cat(server_label, dim=0)
    server_batch = torch.cat(server_batch, dim=0)

    # Initialize client data structures
    client_data_x = []
    client_data_edge_index = []
    client_labels = []
    client_batches = []

    for i in range(n_clients):
        client_data_x.append(server_data_x)
        client_data_edge_index.append(server_data_edge_index)
        client_labels.append(server_label)
        client_batches.append(server_batch)

    return (server_data_x, server_data_edge_index, server_batch), server_label, (client_data_x, client_data_edge_index, client_batches), client_labels


def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1, task_type="classification"):
    """
    Assign the data to the clients.
    train_data: dataloader of the training dataset
    bias: degree of non-iid between the classes loaded by each client
    device: device used in training and inference
    num_labels: number of classes
    num_workers: number of benign and malicious clients used during training
    server_pc: number of data examples in the server dataset
    p: bias probability in server dataset
    dataset: name of the dataset
    seed: seed for randomness
    """
    if task_type == "classification":
        other_group_size = (1 - bias) / (num_labels - 1)
        if dataset == "HAR":
            worker_per_group = 30 / num_labels
        else:
            raise NotImplementedError

        # assign training data to each worker
        if dataset == "HAR":
            each_worker_data = [[] for _ in range(30)]
            each_worker_label = [[] for _ in range(30)]
        else:
            raise NotImplementedError
        server_data = []
        server_label = []

        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])
        server_counter = [0 for _ in range(num_labels)]

        # compute the labels needed for each class
        if dataset == "HAR":
            esol_data = MoleculeNet(root=".", name="ESOL")
            train_data = DataLoader(
                esol_data[:int(len(esol_data) * 0.8)],
                batch_size=1,  # for assign_data() to work, it should be 1
                shuffle=True)

            for _, (data, label) in enumerate(train_data):
                data = data.to(device)
                label = label.to(device)

                for (x, y) in zip(data, label):
                    clientId = int(x[0].item())-1
                    x = x[1:len(x)]
                    x = x.reshape(1, 561)
                    # Assign x and y to appropriate client or server based on method by original code
                    if server_counter[int(y.cpu().numpy())] < samp_dis[int(y.cpu().numpy())]:
                        server_data.append(x)
                        server_label.append(y)
                        server_counter[int(y.cpu().numpy())] += 1
                    else:
                        each_worker_data[clientId].append(x)
                        each_worker_label[clientId].append(y)
        else:
           raise NotImplementedError

        if server_pc != 0:
            server_data = torch.cat(server_data, dim=0)
            server_label = torch.stack(server_label, dim=0)
        else:
            if dataset == "HAR":
                server_data = torch.empty(size=(0, 561)).to(device)
            else:
                raise NotImplementedError

            server_label = torch.empty(size=(0, )).to(device)

        each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data]
        each_worker_label = [torch.stack(each_worker, dim=0) for each_worker in each_worker_label]

        # randomly permute the workers
        if dataset == "HAR":
            random_order = np.random.RandomState(seed=seed).permutation(30)
        else:
            raise NotImplementedError
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        return server_data, server_label, each_worker_data, each_worker_label

    elif task_type == "regression":
        bias = 0
        seed = None
        server_size = 102
        n_clients = num_workers = num_workers
        num_samples = len(train_data)
        all_clients_size = num_samples - server_size
        each_client_data_size = all_clients_size // num_workers

        total_data_x = []
        total_data_edge_index = []
        total_batches = []
        total_labels = []

        for index, data_batch in enumerate(train_data):
            data_x, data_edge_index, label, batch = data_batch.x, data_batch.edge_index, data_batch.y, data_batch.batch
            total_data_x.append(data_x)
            total_data_edge_index.append(data_edge_index)
            total_batches.append(batch+index)
            total_labels.append(label)

        server_data_x = total_data_x[:server_size]
        server_data_edge_index = total_data_edge_index[:server_size]
        server_label = total_labels[:server_size]
        server_batch = total_batches[:server_size]

        server_data_x = torch.cat(server_data_x, dim=0)
        server_data_edge_index = torch.cat(server_data_edge_index, dim=1)
        server_label = torch.cat(server_label, dim=0)
        server_batch = torch.cat(server_batch, dim=0)

        # Initialize client data structures
        client_data_x = []
        client_data_edge_index = []
        client_labels = []
        client_batches = []

        for i in range(n_clients):
            start_idx = server_size + i * each_client_data_size
            end_idx = start_idx + each_client_data_size

            client_data_x.append(torch.cat(total_data_x[start_idx:end_idx], dim=0))
            client_data_edge_index.append(torch.cat(total_data_edge_index[start_idx:end_idx], dim=1))
            client_labels.append(torch.cat(total_labels[start_idx:end_idx], dim=0))
            temp = total_batches[start_idx:end_idx]
            temp = [temp - start_idx for temp in temp]
            client_batches.append(torch.cat(temp, dim=0))

        if bias > 0:
            for i in range(n_clients):
                num_bias_samples = int(each_client_data_size * bias)
                if num_bias_samples > 0:
                    bias_indices = np.random.choice(each_client_data_size, num_bias_samples, replace=False)
                    swap_client = (i + 1) % n_clients
                    client_data_x[i][bias_indices] = client_data_x[swap_client][bias_indices]
                    client_data_edge_index[i][:, bias_indices] = client_data_edge_index[swap_client][:, bias_indices]
                    client_labels[i][bias_indices] = client_labels[swap_client][bias_indices]
                    client_batches[i][bias_indices] = client_batches[swap_client][bias_indices]

        if seed is not None:
            np.random.seed(seed)
            for i in range(n_clients):
                perm = np.random.permutation(each_client_data_size)
                client_data_x[i] = client_data_x[i][perm]
                client_data_edge_index[i] = client_data_edge_index[i][:, perm]
                client_labels[i] = client_labels[i][perm]
                client_batches[i] = client_batches[i][perm]

        return (server_data_x, server_data_edge_index, server_batch), server_label, (client_data_x, client_data_edge_index, client_batches), client_labels

    else:
        raise NotImplementedError("Unsupported task type. Use 'classification' or 'regression'.")