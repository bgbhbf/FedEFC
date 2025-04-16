# python version 3.8.2
# -*- coding: utf-8 -*-

import copy
import time
import numpy as np
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import scoreatpercentile

from util.options import args_parser, LARGE_VALUE, PERCENTILE
from util.local_training import LocalUpdate, globaltest, LocalUpdateDitto
from util.dataset import get_dataset, SubsetWithTargets
from util.util import setting_torch, add_noise_to_data, save_document
from model.build_model import build_model, initialize_weights
from util.fedavg import FedAvg, FedDyn
from model.lenet import model_testing
from util.count import compute_count_matrix


def local_pred_prob(args, user_id, netglob, netglob_temp, dataset_train, dict_users, y_train_noisy, y_train):
    netglob_temp.load_state_dict(copy.deepcopy(netglob.state_dict()))
    test_acc, test_losses, misclassified = [], [], []
    local_train_targets = SubsetWithTargets(dataset_train, dict_users[user_id])
    pred_probs, accuracy = model_testing(netglob_temp, args.device, local_train_targets,
                                         test_acc, test_losses, misclassified,
                                         y_train_noisy, y_train)
    return pred_probs, accuracy


def generate_count_matrix(args, user_id, dict_users, netglob, netglob_temp, dataset_train, y_train_noisy, y_train):
    pred_probs, _ = local_pred_prob(args, user_id, netglob, netglob_temp, dataset_train, dict_users, y_train_noisy, y_train)
    local_train_targets = SubsetWithTargets(dataset_train, dict_users[user_id])
    train_targets_label = local_train_targets.targets

    confident_joint = compute_count_matrix(
        labels=train_targets_label,
        pred_probs=pred_probs
    )

    return confident_joint

def joint_distribution_matrix(args, user_id, dict_users, netglob, netglob_temp, dataset_train, y_train_noisy, y_train):
    pred_probs, _ = local_pred_prob(args, user_id, netglob, netglob_temp, dataset_train, dict_users, y_train_noisy, y_train)
    local_train_targets = SubsetWithTargets(dataset_train, dict_users[user_id])
    train_targets_label = local_train_targets.targets

    from cleanlab.count import compute_confident_joint #confident learning using official libarary

    confident_joint, _ = compute_confident_joint(
        labels=train_targets_label,
        pred_probs=pred_probs,
        multi_label=False,
        return_indices_of_off_diagonals=True,
    )
    #for stable
    except_flag = 1 if len(np.unique(confident_joint)) < 3 else 0

    return train_targets_label, pred_probs, confident_joint, except_flag

def estimate_transition_matrix(probabilities, alpha=97, f_acc=None):
    num_classes = probabilities.shape[1]
    transition_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        class_probs = probabilities[:, i]
        threshold = scoreatpercentile(class_probs, alpha)
        perfect_examples_indices = np.where(class_probs >= threshold)[0]

        if len(perfect_examples_indices) == 0:
            transition_matrix[i, :] = 0
        else:
            transition_matrix[i, :] = np.mean(probabilities[perfect_examples_indices], axis=0)
    return transition_matrix

def class_distribution_weight(dataset_train, dict_users, user_id, args, noise_matrix):
    local_train_targets = SubsetWithTargets(dataset_train, dict_users[user_id])
    train_targets_label = local_train_targets.targets
    label_counts = np.bincount(train_targets_label, minlength=args.num_classes)
    label_probabilities = label_counts / len(train_targets_label)
    noise_matrix_weighted = noise_matrix * label_probabilities[:, np.newaxis]
    col_sum = noise_matrix_weighted.sum(axis=0)
    nonzero = col_sum != 0
    noise_matrix_weighted[:, nonzero] /= col_sum[nonzero]
    noise_matrix = noise_matrix_weighted.T
    return noise_matrix

def compute_noise_matrix(args, user_id, dict_users, netglob, netglob_temp, dataset_train, y_train_noisy, y_train):
    if args.method == 'FedEFC':
        count_matrix = generate_count_matrix(args, user_id, dict_users, netglob, netglob_temp, dataset_train, y_train_noisy, y_train)
        column_sum = count_matrix.sum(axis=0, keepdims=True).T
        count_matrix_temp = count_matrix.T
        noise_matrix = np.divide(count_matrix_temp, column_sum,out=np.zeros_like(count_matrix_temp, dtype=np.float64),where=column_sum != 0)
        noise_matrix = noise_matrix.T
        if args.weigth_compensate is True:
            noise_matrix = class_distribution_weight(dataset_train, dict_users, user_id, args, noise_matrix)

    elif args.method == 'forward_correction':
        pred_probs, _ = local_pred_prob(args, user_id, netglob, netglob_temp, dataset_train, dict_users, y_train_noisy, y_train)
        noise_matrix = estimate_transition_matrix(pred_probs, alpha=PERCENTILE)
        noise_matrix = class_distribution_weight(dataset_train, dict_users, user_id, args, noise_matrix)
    else:
        noise_matrix = None
    return noise_matrix


def update_client(args, rnd, early_stop_rnd, user_id, netglob, netglob_temp, dataset_train,
                  dict_users, dict_users_buffer, y_train_noisy, y_train, forwardCorrected):
    accuracies = 0
    if early_stop_rnd <= rnd:
        noise_matrix = None
        if args.method in ['FedEFC', 'forward_correction']:
            noise_matrix = compute_noise_matrix(args, user_id, dict_users, netglob, netglob_temp,
                                                        dataset_train, y_train_noisy, y_train)
        elif args.method == 'confident_learning':
            dict_users = FL_confident(args, user_id, netglob, netglob_temp, dataset_train,
                                dict_users, dict_users_buffer, y_train_noisy, y_train)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        w_local, accuracies = local.update_weights(net=copy.deepcopy(netglob).to(args.device),
                                                seed=args.seed, w_g=netglob.to(args.device),
                                                epoch=args.local_ep, joint_matrix=noise_matrix,
                                                correctedType=forwardCorrected)
    else:
        if args.method == 'FedDitto':
            local = LocalUpdateDitto(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        else:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        w_local, accuracies = local.update_weights(net=copy.deepcopy(netglob).to(args.device),
                                                   seed=args.seed, w_g=netglob.to(args.device),
                                                   epoch=args.local_ep)
    return w_local, dict_users, accuracies



def generate_local_train(dataset_train, dict_users):
    #client data and target after adding noise
    user_data_dict = {}
    for user_id, indices in dict_users.items():
        user_data_dict[user_id] = {
            'data': np.array([dataset_train.data[i] for i in indices]),
            'targets': np.array([dataset_train.targets[i] for i in indices])
        }
    return user_data_dict


def filtering_user_dict(dict_users, label_issues_indices, user_id, dict_users_buffer):
    updated_dict = copy.deepcopy(dict_users_buffer)
    user_data = list(updated_dict[user_id])
    extracted_values = [user_data[i] for i in label_issues_indices if 0 <= i < len(user_data)]
    for value in extracted_values:
        updated_dict[user_id].discard(value)
    return updated_dict

def confident_learning(args, user_id, dict_users, dict_users_buffer,
                       netglob, netglob_temp, dataset_train, y_train_noisy, y_train):
    
    from cleanlab.filter import find_label_issues #confident learning using official libarary

    train_targets_label, pred_probs, _, except_flag = joint_distribution_matrix(
        args, user_id, dict_users, netglob, netglob_temp, dataset_train, y_train_noisy, y_train)
    if except_flag == 1:
        return dict_users

    label_issues_indices = find_label_issues(
        labels=train_targets_label,
        pred_probs=pred_probs,
        filter_by="both",
        return_indices_ranked_by="self_confidence",
        rank_by_kwargs={"adjust_pred_probs": True},
    )

    label_issues_indices = set(label_issues_indices)
    return filtering_user_dict(dict_users, label_issues_indices, user_id, dict_users_buffer)

def FL_confident(args, user_id, netglob, netglob_temp, dataset_train, dict_users,
           dict_users_buffer, y_train_noisy, y_train):

    netglob_temp.load_state_dict(copy.deepcopy(netglob.state_dict()))
    return confident_learning(args, user_id, dict_users, dict_users_buffer,
                              netglob, netglob_temp, dataset_train, y_train_noisy, y_train)


# -------------------------- Main -------------------------- #
if __name__ == '__main__':
    args = args_parser()
    rootpath = setting_torch(args)
    Early_StopRND = LARGE_VALUE  # prestop point
    forwardCorrected = False  # Init

    # Method
    if args.method in ['FedAvg', 'FedDitto']:
        args.prestopping = False
    elif args.method in ['FedEFC', 'forward_correction']:
        forwardCorrected = True
    elif args.method == 'FedProx':
        args.prestopping = False
        args.beta = 1e-5 #Prox parameter
    elif args.method == 'FedDyn':
        controls = {}
        args.prestopping = False
    else:
        print("No matched method")

    dataset_train, dataset_test, dict_users = get_dataset(args)
    log_dir = f'./runs/{args.dataset}/alpha{args.alpha_dirichlet}/p{args.non_iid_prob_class}/noise{args.noise_amount}/sparsity{args.sparsity}/{args.method}'

    writer = SummaryWriter(log_dir)
    f_acc = save_document(rootpath, args)
    print(args)

    #add noise
    y_train = np.array(dataset_train.targets)
    y_test = np.array(dataset_test.targets)
    if args.noise_amount != 0:
        y_train_noisy = add_noise_to_data(args, y_train, y_test, args.noise_amount, args.sparsity)
        dataset_train.targets = y_train_noisy
    else:
        y_train_noisy = y_train

    dict_users_buffer = copy.deepcopy(dict_users) #for confident learning

    local_train = generate_local_train(dataset_train, dict_users)

    #init model
    netglob = build_model(args)
    netglob.apply(initialize_weights)
    netglob_temp = build_model(args)
    netglob_temp.apply(initialize_weights)
    if args.method == 'FedDyn':
        controls = {}
    m = max(int(args.frac * args.num_users), 1)
    prob = [1 / args.num_users] * args.num_users
    stored_high_acc = 0
    criterion_cnt = 0

    print("Using device:", args.device)

    for rnd in trange(0, args.rounds):
        start_time = time.time()
        w_locals = []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        estimate_accruacy = 0 #init estimate

        for user_id in idxs_users:
            w_local, dict_users, train_accuracies = update_client(args, rnd, Early_StopRND, user_id,netglob, netglob_temp, dataset_train, dict_users, dict_users_buffer, y_train_noisy, y_train, forwardCorrected)
            w_locals.append(copy.deepcopy(w_local))
            estimate_accruacy += train_accuracies

        estimate_accruacy = estimate_accruacy/m
        print(f"Round: {rnd}, current_accruacy: {estimate_accruacy}")
        dict_len = [len(dict_users[user_id]) for user_id in idxs_users]
        if args.method == 'FedDyn':
            w_glob_fl, controls = FedDyn(args, global_model=copy.deepcopy(netglob),w_locals=w_locals, idx_users=idxs_users, controls=controls)
        else:
            w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc_s = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        writer.add_scalar('testacc', acc_s, rnd)
        writer.add_scalar('validacc', estimate_accruacy, rnd)

        if args.prestopping:
            if criterion_cnt < args.gamma_thr and rnd > args.start_monitor:
                if estimate_accruacy > stored_high_acc:
                    stored_high_acc = estimate_accruacy
                    criterion_cnt = 0
                else:
                    criterion_cnt += 1
            if criterion_cnt == args.gamma_thr:
                Early_StopRND = rnd
                criterion_cnt = LARGE_VALUE

        print(f"Federate Learning: round {rnd}, test acc {acc_s:.4f}")
        f_acc.write(f"[Federate Learning]: round {rnd}, test acc {acc_s:.4f}\n"); f_acc.flush()
        end_time = time.time()
        print(f"Round execution time: {end_time - start_time:.5f} seconds")

    torch.cuda.empty_cache()
