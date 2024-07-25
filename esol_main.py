from __future__ import print_function

import aggregation_rules
import numpy as np
import random
import argparse
import attacks
import data_loaders

import os
import math
import subprocess

import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from models.gcn import GCN
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import wandb
from torch_geometric.nn import GCNConv
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

def parse_args():
    """
    Parses all commandline arguments.
    """
    parser = argparse.ArgumentParser(
        description="SAFEFL: MPC-friendly framework for Private and Robust Federated Learning")

    ### Model and Dataset
    parser.add_argument("--net", help="net: GCN | lr", type=str, default="GCN")
    parser.add_argument("--task_type", help="classification | regression", type=str, default="regression")
    parser.add_argument("--dataset", help="dataset: HAR | ESOL", type=str, default="ESOL")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0025)
    parser.add_argument("--adam_lr", help="adam learning rate for the local trainings", type=float, default=0.0007)
    parser.add_argument("--nparty", help="# parties", type=int, default=2)
    parser.add_argument('--use_agreggation', default=False, help='use Adam optimizer or FL aggregations')

    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--p", help="bias probability of class 1 in server dataset", type=float, default=0.1)

    ### Training
    parser.add_argument("--niter", help="# iterations", type=int, default=1)
    parser.add_argument("--local_epoch", help="# local optimization or train", type=int, default=2000)
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--gpu", help="no gpu = -1, gpu training otherwise", type=int, default=-1)
    parser.add_argument("--seed", help="seed", type=int, default=1)
    parser.add_argument("--nruns", help="number of runs for averaging accuracy", type=int, default=1)
    parser.add_argument("--test_every", help="testing interval", type=int, default=1)

    ### Aggregations
    parser.add_argument("--aggregation", help="fedavg or fltrust or", type=str, default="fltrust")

    # FLOD
    parser.add_argument("--flod_threshold", help="hamming distance threshold as fraction of total model parameters",
                        type=float, default=0.5)

    # FLAME
    parser.add_argument("--flame_epsilon", help="epsilon for differential privacy in FLAME", type=int, default=3000)
    parser.add_argument("--flame_delta", help="delta for differential privacy in FLAME", type=float, default=0.001)

    # DNC
    parser.add_argument("--dnc_niters", help="number of iterations to compute good sets in DnC", type=int, default=5)
    parser.add_argument("--dnc_c", help="filtering fraction, percentage of number of malicious clients filtered",
                        type=float, default=1)
    parser.add_argument("--dnc_b", help="dimension of subsamples must be smaller, then the dimension of the gradients",
                        type=int, default=2000)

    ### Attacks
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=6)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no",
                        choices=["no", "trim_attack", "krum_attack",
                                 "scaling_attack", "fltrust_attack", "label_flipping_attack", "min_max_attack",
                                 "min_sum_attack"])

    ### MP-SPDZ
    parser.add_argument('--mpspdz', default=False, action='store_true', help='Run example in multiprocess mode')
    parser.add_argument("--port", help="port for the mpc servers", type=int, default=14000)
    parser.add_argument("--chunk_size", help="data amount send between client and server at once", type=int,
                        default=200)
    parser.add_argument("--protocol", help="protocol used in MP-SPDZ", type=str, default="mascot",
                        choices=["semi2k", "mascot", "spdz2k", "replicated2k", "psReplicated2k"])
    parser.add_argument("--players", help="number of computation parties", type=int, default=2)
    parser.add_argument("--threads", help="number of threads per computation party in MP-SPDZ", type=int, default=1)
    parser.add_argument("--parallels", help="number of parallel computation for each thread", type=int, default=1)
    parser.add_argument('--always_compile', default=False, action='store_true',
                        help='compiles program even if it was already compiled')

    return parser.parse_args()



def predict(model, test_loader, device):
    all_preds = []
    all_targets = []

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            inputs_x, inputs_edge_index, inputs_batch, targets = batch.x, batch.edge_index, batch.batch, batch.y
            inputs_x = inputs_x.to(device)
            inputs_edge_index = inputs_edge_index.to(device)
            inputs_batch = inputs_batch.to(device)
            targets = targets.to(device)

            preds, logits = model(inputs_x.float(), inputs_edge_index.int(), inputs_batch)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    df = pd.DataFrame()
    df["y_real"] = all_targets
    df["y_pred"] = all_preds

    df["y_real"] = df["y_real"].apply(lambda row: row[0])
    df["y_pred"] = df["y_pred"].apply(lambda row: row[0])

    # Determine the limits for the plot based on real values
    min_val = df["y_real"].min()
    max_val = df["y_real"].max()

    axes = sns.scatterplot(data=df, x="y_real", y="y_pred")
    axes.set_xlabel("Real Solubility")
    axes.set_ylabel("Predicted Solubility")

    # Set the same scale for both axes based on real values
    axes.set_xlim(min_val, max_val)
    axes.set_ylim(min_val, max_val)

    plt.show()


def plot_train_loss(values, name):
    # Convert values to float and handle inf values
    losses_float = [float(loss) if np.isfinite(loss) else np.nan for loss in values]
    loss_indices = range(len(losses_float))
    ax = sns.lineplot(x=loss_indices, y=losses_float)
    ax.set(xlabel='Epoch', ylabel=f"{name}")
    plt.show()


def get_device(device):
    """
    Selects the device to run the training process on.
    device: -1 to only use cpu, otherwise cuda if available
    """
    if device == -1:
        ctx = torch.device('cpu')
    else:
        ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(ctx)
    return ctx


def get_net(net_type, num_inputs, num_outputs=10):
    """
    Selects the model architecture.
    net_type: name of the model architecture
    num_inputs: number of inputs of model
    num_outputs: number of outputs/classes
    """

    if net_type == "GCN":
        import models.gcn as gcn
        net = GCN(input_dim=num_inputs, output_dim=num_outputs)
        print(net)
    else:
        raise NotImplementedError
    return net


def get_byz(byz_type):
    """
    Gets the attack type.
    byz_type: name of the attack
    """
    if byz_type == "no":
        return attacks.no_byz
    elif byz_type == 'trim_attack':
        return attacks.trim_attack
    elif byz_type == "krum_attack":
        return attacks.krum_attack
    elif byz_type == "scaling_attack":
        return attacks.scaling_attack_scale
    elif byz_type == "fltrust_attack":
        return attacks.fltrust_attack
    elif byz_type == "label_flipping_attack":
        return attacks.no_byz
    elif byz_type == "min_max_attack":
        return attacks.min_max_attack
    elif byz_type == "min_sum_attack":
        return attacks.min_sum_attack
    else:
        raise NotImplementedError


def get_protocol(protocol, players):
    """
    Returns the shell script name and number of players for the protocol.
    protocol: name of the protocol
    players: number of parties
    """
    if players < 2:
        raise Exception("Number of players must at least be 2")

    if protocol == "semi2k":
        return "semi2k.sh", players

    elif protocol == "mascot":
        return "mascot.sh", players

    elif protocol == 'spdz2k':
        return "spdz2k.sh", players

    elif protocol == "replicated2k":
        if players != 3:
            raise Exception("Number of players must be 3 for replicated2k")
        return "ring.sh", 3

    elif protocol == "psReplicated2k":
        if players != 3:
            raise Exception("Number of players must be 3 for psReplicated2k")
        return "ps-rep-ring.sh", 3

    else:
        raise NotImplementedError


def evaluate_accuracy(data_iterator, net, device, trigger, dataset):
    """
    Evaluate the accuracy and backdoor success rate of the model. Fails if model output is NaN.
    data_iterator: test data iterator
    net: model
    device: device used in training and inference
    trigger: boolean if backdoor success rate should be evaluated
    dataset: name of the dataset used in the backdoor attack
    """
    net.eval()
    if wandb.config.dataset == "ESOL":
        total_mse_error = 0
        total_mae_error = 0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(data_iterator):
                inputs_x, inputs_edge_index, inputs_batch, targets = batch.x, batch.edge_index, batch.batch, batch.y
                inputs_x = inputs_x.to(device)
                inputs_edge_index = inputs_edge_index.to(device)
                inputs_batch = inputs_batch.to(device)
                targets = targets.to(device)

                preds, logits = net(inputs_x.float(), inputs_edge_index.int(), inputs_batch)

                if not torch.isnan(preds).any():
                    mse_error = F.mse_loss(preds, targets, reduction='sum').item()
                    mae_error = F.l1_loss(preds, targets).item()
                    total_mse_error += mse_error
                    total_mae_error += mae_error
                    total_samples += 1
                else:
                    print("NaN in output of net")
                    raise ArithmeticError

        avg_mse_error = total_mse_error / total_samples
        avg_mae_error = total_mae_error / total_samples
        return avg_mse_error, avg_mae_error


def plot_results(runs_test_accuracy, runs_backdoor_success, test_iterations, niter):
    """
    Plots the evaluation results.
    runs_test_accuracy: accuracy of the model in each iteration specified in test_iterations of every run
    runs_backdoor_success: backdoor success of the model in each iteration specified in test_iterations of every run
    test_iterations: list of iterations the model was evaluated in
    niter: number of iteration the model was trained for
    """
    test_acc_std = []
    test_acc_list = []
    backdoor_success_std = []
    backdoor_success_list = []

    # insert (0,0) as starting point for plot and calculate mean and standard deviation if multiple runs were performed
    if wandb.config.nruns == 1:
        if wandb.config.byz_type == "scaling_attack":
            runs_backdoor_success = np.insert(runs_backdoor_success, 0, 0, axis=0)
            backdoor_success_list = runs_backdoor_success
            backdoor_success_std = [0 for i in range(0, len(runs_backdoor_success))]
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=0)
        test_acc_list = runs_test_accuracy
        test_acc_std = [0 for i in range(0, len(runs_test_accuracy))]
    else:
        if wandb.config.byz_type == "scaling_attack":
            runs_backdoor_success = np.insert(runs_backdoor_success, 0, 0, axis=1)
            backdoor_success_list = np.mean(runs_backdoor_success, axis=0)
            backdoor_success_std = np.std(runs_backdoor_success, axis=0)
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=1)
        test_acc_std = np.std(runs_test_accuracy, axis=0)
        test_acc_list = np.mean(runs_test_accuracy, axis=0)

    test_iterations.insert(0, 0)
    # Print accuracy and backdoor success rate in array form to console
    print("Test accuracy of runs:")
    print(repr(runs_test_accuracy))
    if wandb.config.byz_type == "scaling_attack":
        print("Backdoor attack success rate of runs:")
        print(repr(runs_backdoor_success))

    # Determine in which iteration in what run the highest accuracy was achieved.
    # Also print overall mean accuracy and backdoor success rate
    max_index = np.unravel_index(runs_test_accuracy.argmax(), runs_test_accuracy.shape)
    if wandb.config.nruns == 1:
        print(
            "Run 1 in iteration %02d had the highest accuracy of %0.4f" % (max_index[0] * 50, runs_test_accuracy.max()))
    else:
        print("Run %02d in iteration %02d had the highest accuracy of %0.4f" % (
        max_index[0] + 1, max_index[1] * 50, runs_test_accuracy.max()))
        print("The average final accuracy was: %0.4f with an overall average:" % (test_acc_list[-1]))
        print(repr(test_acc_list))
        if wandb.config.byz_type == "scaling_attack":
            print("The average final backdoor success rate was: %0.4f with an overall average:" % backdoor_success_list[
                -1])
            print(repr(backdoor_success_list))
    # Generate plot with two axis displaying accuracy and backdoor success rate over the iterations
    if wandb.config.byz_type == "scaling_attack":
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        accuracy_plot = ax1.plot(test_iterations, test_acc_list, color='C0', label='accuracy')
        ax1.fill_between(test_iterations, test_acc_list - test_acc_std, test_acc_list + test_acc_std, color='C0')
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Backdoor success rate')
        backdoor_plot = ax2.plot(test_iterations, backdoor_success_list, color='C1', label='Backdoor success rate')
        ax2.fill_between(test_iterations, backdoor_success_list - backdoor_success_std,
                         backdoor_success_list + backdoor_success_std, color='C1')
        ax2.set_ylim(0, 1)

        lns = accuracy_plot + backdoor_plot
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=0)
        plt.xlim(0, niter)
        plt.title(
            "Test Accuracy + Backdoor success: " + wandb.config.net + ", " + wandb.config.dataset + ", " + wandb.config.aggregation + ", " + wandb.config.byz_type + ", nruns " + str(
                wandb.config.nruns))
        plt.grid()
        plt.show()
    # Generate plot with only the accuracy as one axis over the iterations
    else:
        plt.plot(test_iterations, test_acc_list, color='C0')
        plt.fill_between(test_iterations, test_acc_list - test_acc_std, test_acc_list + test_acc_std, color='C0')
        plt.title(
            "Test Accuracy: " + wandb.config.net + ", " + wandb.config.dataset + ", " + wandb.config.aggregation + ", " + wandb.config.byz_type + ", nruns " + str(
                wandb.config.nruns))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.xlim(0, niter)
        plt.ylim(0, 1)
        plt.grid()
        plt.show()


def weight_init(m):
    """
    Initializes the weights of the layer with random values.
    m: the layer which gets initialized
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=2.24)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        # GCNConv doesn't have a single weight attribute, it has parameters
        for param in m.parameters():
            if param.dim() > 1:  # This checks if the parameter is a weight matrix
                nn.init.xavier_uniform_(param, gain=2.24)
            else:  # This initializes bias
                nn.init.zeros_(param)


def main():
    """
    The main function that runs the entire training process of the model.
    args: arguments defining hyperparameters
    """
    wandb.init()
    args = parse_args()  # parse arguments
    wandb.config.update(args)
    # setup
    device = get_device(wandb.config.gpu)
    num_inputs, num_outputs, num_labels = data_loaders.get_shapes(wandb.config.dataset)
    byz = get_byz(wandb.config.byz_type)

    # Print all arguments
    paraString = ('dataset: p' + str(wandb.config.p) + '_' + str(wandb.config.dataset) + ", server_pc: " + str(
        wandb.config.server_pc) + ", bias: " + str(wandb.config.bias)
                  + ", nparty: " + str(wandb.config.nparty) + ", net: " + str(wandb.config.net) + ", niter: " + str(
                wandb.config.niter) + ", lr: " + str(wandb.config.lr)
                  + ", batch_size: " + str(wandb.config.batch_size) + ", nbyz: " + str(
                wandb.config.nbyz) + ", attack: " + str(wandb.config.byz_type)
                  + ", aggregation: " + str(wandb.config.aggregation) + ", FLOD_threshold: " + str(
                wandb.config.flod_threshold)
                  + ", Flame_epsilon: " + str(wandb.config.flame_epsilon) + ", Flame_delta: " + str(
                wandb.config.flame_delta) + ", Number_runs: " + str(wandb.config.nruns)
                  + ", DnC_niters: " + str(wandb.config.dnc_niters) + ", DnC_c: " + str(
                wandb.config.dnc_c) + ", DnC_b: " + str(wandb.config.dnc_b)
                  + ", MP-SPDZ: " + str(wandb.config.mpspdz) + ", Port: " + str(
                wandb.config.port) + ", Chunk_size: " + str(wandb.config.chunk_size)
                  + ", Protocol: " + wandb.config.protocol + ", Threads: " + str(
                wandb.config.threads) + ", Parallels: " + str(wandb.config.parallels)
                  + ", Seed: " + str(wandb.config.seed) + ", Test Every: " + str(wandb.config.test_every))
    print(paraString)

    # saving iterations for averaging
    runs_test_accuracy = []
    runs_backdoor_success = []
    test_iterations = []
    backdoor_success_list = []

    # model
    global_model = get_net(wandb.config.net, num_outputs=num_outputs, num_inputs=num_inputs)
    global_model.to(device)
    num_params = torch.cat([xx.reshape((-1, 1)) for xx in global_model.parameters()], dim=0).size()[0]  # used for FLOD to determine threshold
    # loss
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=wandb.config.adam_lr)

    # perform multiple runs
    for run in range(1, wandb.config.nruns + 1):
        grad_list = []
        test_acc_list = []
        test_iterations = []
        backdoor_success_list = []
        server_process = None

        # fix the seeds for deterministic results
        if wandb.config.seed > 0:
            wandb.config.seed = wandb.config.seed + run - 1
            torch.cuda.manual_seed_all(wandb.config.seed)
            torch.manual_seed(wandb.config.seed)
            random.seed(wandb.config.seed)
            np.random.seed(wandb.config.seed)

        # set aggregation specific variables
        if wandb.config.aggregation == "shieldfl":
            previous_global_gradient = 0  # important for ShieldFL, all other aggregation rules don't need it
            previous_gradients = []
        elif wandb.config.aggregation == "foolsgold":
            gradient_history = [torch.zeros(size=(num_params, 1)).to(device) for i in
                                range(wandb.config.nparty)]  # client gradient history for FoolsGold
        elif wandb.config.aggregation == "contra":
            gradient_history = [torch.zeros(size=(num_params, 1)).to(device) for i in
                                range(wandb.config.nparty)]  # client gradient history for CONTRA
            reputation = torch.ones(size=(wandb.config.nparty,)).to(device)  # reputation scores for CONTRA
            cos_dist = torch.zeros((wandb.config.nparty, wandb.config.nparty), dtype=torch.double).to(
                device)  # pairwise cosine similarity for CONTRA
        elif wandb.config.aggregation == "romoa":
            # don't know why they initialize it like this
            previous_global_gradient = torch.cat(
                [param.clone().detach().flatten() for param in global_model.parameters()]).reshape(-1, 1) + torch.normal(mean=0,
                                                                                                                std=1e-7,
                                                                                                                size=(
                                                                                                                num_params,
                                                                                                                1)).to(
                device)
            sanitization_factor = torch.full(size=(wandb.config.nparty, num_params),
                                             fill_value=(1 / wandb.config.nparty)).to(device)  # sanitization factors for Romoa

        train_data, test_data = data_loaders.load_data(wandb.config.dataset, wandb.config.seed)  # load the data

        # todo: assign data to the server and clients

        # perform data poisoning attacks
        if wandb.config.byz_type == "label_flipping_attack":
            each_party_label = attacks.label_flipping_attack(each_party_label, wandb.config.nbyz, num_labels)
        elif wandb.config.byz_type == "scaling_attack":
            each_party_data, each_party_label = attacks.scaling_attack_insert_backdoor(each_party_data,
                                                                                       each_party_label,
                                                                                       wandb.config.dataset,
                                                                                       wandb.config.nbyz, device)

        print("Data done")

        # start FLTrust computation parties
        if wandb.config.mpspdz:
            os.chdir("mpspdz")

            print("Starting Computation Parties")
            # start computation servers using a child process to run in parallel
            server_process = subprocess.Popen(
                ["./run_aggregation.sh", wandb.config.script, wandb.config.full_filename, str(wandb.config.players)])

            os.chdir("..")

        # training
        client_losses = []
        server_losses = []
        all_mse = []
        all_mae = []
        global_model.apply(weight_init)
        for global_epoch in range(wandb.config.niter):
            print(f"training epoch {global_epoch}/{wandb.config.niter}")
            global_model.train()

            # perform local training for each party
            for client_i in range(wandb.config.nparty):
                net = copy.deepcopy(global_model).to(device)
                print(f'client {client_i} is updated from global model')
                for local_e in range(wandb.config.local_epoch):
                    if wandb.config.net == "GCN":
                        for batch in train_data:
                            batch.to(device)
                            optimizer.zero_grad()
                            pred, embedding = net(batch.x.float(), batch.edge_index, batch.batch)
                            loss = loss_fn(pred, batch.y)
                            loss.backward()
                            optimizer.step()
                            client_losses.append(loss)
                    if local_e % 100 == 0:
                        wandb.log({"local_epoch": local_e, "loss_clients": loss})
                        print(f"local_epoch: {local_e}, loss_clients_{client_i}: {loss}")

                grad_list.append([param.grad.clone().detach() for param in net.parameters()])
                print("the gradients of party {} are appended: ".format(client_i))


            # compute server update and append it to the end of the list
            if wandb.config.aggregation in ["fltrust", "flod"] or wandb.config.byz_type == "fltrust_attack":
                net = copy.deepcopy(global_model).to(device)
                print(f'server model is updated from global model')
                for local_e in range(wandb.config.local_epoch):
                    if wandb.config.net == "GCN":
                        for batch in train_data:
                            batch.to(device)
                            optimizer.zero_grad()
                            pred, embedding = net(batch.x.float(), batch.edge_index, batch.batch)
                            loss = loss_fn(pred, batch.y)
                            loss.backward()
                            optimizer.step()
                            server_losses.append(loss)
                    if local_e % 100 == 0:
                        wandb.log({"epoch": global_epoch, "loss_server": loss})
                        print(f"local_epoch: {local_e}, loss_server: {loss}")
                grad_list.append([torch.clone(param.grad) for param in net.parameters()])
                print("server's gradients are appended: ")

            # perform the aggregation
            # print("train epoch: {} perform the aggregation using {}".format(e, wandb.config.aggregation))
            with torch.no_grad():
                if wandb.config.use_agreggation:
                    if wandb.config.mpspdz:
                        aggregation_rules.mpspdz_aggregation(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz,
                                                             device, param_num=num_params, port=wandb.config.port,
                                                             chunk_size=wandb.config.chunk_size,
                                                             parties=wandb.config.players)

                    elif wandb.config.aggregation == "fltrust":
                        aggregation_rules.fltrust(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device)

                    elif wandb.config.aggregation == "fedavg":
                        if wandb.config.dataset == "ESOL":
                            data_sizes = [x.size(dim=0) for x in each_party_label]
                        global_model = aggregation_rules.fedavg(grad_list, global_model, wandb.config.lr, wandb.config.nbyz, byz, device,
                                                 data_sizes)

                    elif wandb.config.aggregation == "krum":
                        aggregation_rules.krum(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device)

                    elif wandb.config.aggregation == "trim_mean":
                        aggregation_rules.trim_mean(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device)

                    elif wandb.config.aggregation == "median":
                        aggregation_rules.median(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device)

                    elif wandb.config.aggregation == "flame":
                        aggregation_rules.flame(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device,
                                                epsilon=wandb.config.flame_epsilon, delta=wandb.config.flame_delta)

                    elif wandb.config.aggregation == "shieldfl":
                        previous_global_gradient, previous_gradients = aggregation_rules.shieldfl(grad_list, net,
                                                                                                  wandb.config.lr,
                                                                                                  wandb.config.nbyz,
                                                                                                  byz, device,
                                                                                                  previous_global_gradient,
                                                                                                  global_epoch, previous_gradients)

                    elif wandb.config.aggregation == "flod":
                        aggregation_rules.flod(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device,
                                               threshold=math.floor(num_params * wandb.config.flod_threshold))

                    elif wandb.config.aggregation == "divide_and_conquer":
                        aggregation_rules.divide_and_conquer(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz,
                                                             device, niters=wandb.config.dnc_niters,
                                                             c=wandb.config.dnc_c, b=wandb.config.dnc_b)

                    elif wandb.config.aggregation == "foolsgold":
                        gradient_history = aggregation_rules.foolsgold(grad_list, net, wandb.config.lr,
                                                                       wandb.config.nbyz, byz, device,
                                                                       gradient_history=gradient_history)

                    elif wandb.config.aggregation == "contra":
                        gradient_history, reputation, cos_dist = aggregation_rules.contra(grad_list, net,
                                                                                          wandb.config.lr,
                                                                                          wandb.config.nbyz, byz,
                                                                                          device,
                                                                                          gradient_history=gradient_history,
                                                                                          reputation=reputation,
                                                                                          cos_dist=cos_dist, C=1)

                    elif wandb.config.aggregation == "signguard":
                        aggregation_rules.signguard(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device,
                                                    seed=wandb.config.seed)

                    elif wandb.config.aggregation == "flare":
                        aggregation_rules.flare(grad_list, net, wandb.config.lr, wandb.config.nbyz, byz, device,
                                                train_data)

                    elif wandb.config.aggregation == "romoa":
                        sanitization_factor, previous_global_gradient = aggregation_rules.romoa(grad_list, net,
                                                                                                wandb.config.lr,
                                                                                                wandb.config.nbyz, byz,
                                                                                                device,
                                                                                                F=sanitization_factor,
                                                                                                prev_global_update=previous_global_gradient,
                                                                                                seed=wandb.config.seed)

                    else:
                        raise NotImplementedError

                    del grad_list
                    grad_list = []
                # evaluate the model accuracy
                if (global_epoch + 1) % wandb.config.test_every == 0:
                    test_metric, test_success_rate = evaluate_accuracy(test_data, net, device,
                                                                       wandb.config.byz_type == "scaling_attack",
                                                                       wandb.config.dataset)
                    wandb.log({"test_metric": test_metric, "test_success_rate": test_success_rate})
                    # print("test_metric", test_metric)
                    # print("test_success_rate", test_success_rate)
                    test_acc_list.append(test_metric)
                    test_iterations.append(global_epoch)
                    if wandb.config.dataset == "ESOL":
                        all_mse.append(test_metric)
                        all_mae.append(test_success_rate)
                        wandb.log(
                            {"Iteration": global_epoch, "MeanSquareError": test_metric, "MeanAbsolutError": test_success_rate})
                        print("Iteration %02d.    MeanSquareError %0.4f    MeanAbsolutError %0.4f" % (
                        global_epoch, test_metric, test_success_rate))

                plot_train_loss(client_losses, "clients loss")
                #plot_train_loss(server_losses, "server loss")
                #plot_train_loss(all_mae, "Mean Absolut Error")
                #plot_train_loss(all_mse, "Mean Square Error")

        # global epochs ended

        if wandb.config.mpspdz:
            server_process.wait()  # wait for process to exit

        # Append accuracy and backdoor success rate to overall runs list
        if len(runs_test_accuracy) > 0:
            runs_test_accuracy = np.vstack([runs_test_accuracy, test_acc_list])
            if wandb.config.byz_type == "scaling_attack":
                runs_backdoor_success = np.vstack([runs_backdoor_success, backdoor_success_list])
        else:
            runs_test_accuracy = test_acc_list
            if wandb.config.byz_type == "scaling_attack":
                runs_backdoor_success = backdoor_success_list
        if wandb.config.byz_type == "scaling_attack":
            print("Run %02d/%02d done with final accuracy: %0.4f and backdoor success rate: %0.4f" % (
            run, wandb.config.nruns, test_acc_list[-1], backdoor_success_list[-1]))
        else:
            print("Run %02d/%02d done with final accuracy: %0.4f" % (run, wandb.config.nruns, test_acc_list[-1]))

    del test_acc_list
    test_acc_list = []

    predict(global_model, test_data, device)


if __name__ == "__main__":
    from sweep_config_my_fl import sweep_config

    sweep_id = wandb.sweep(sweep_config, project='sweep_SAFEFL')
    #main()
    wandb.agent(sweep_id, function=main, count=1)

