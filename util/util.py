import numpy as np
import torch
import os
import datetime
import random

from torch.utils.data import Dataset
from util.noise_generation import generate_noise_matrix_from_trace, generate_noisy_labels

def add_noise_to_data(args, y_train, y_test, noise_amount, sparsity):
    np.random.seed(args.seed)
    internal_seed = 42
    py_train = np.bincount(y_train) / float(len(y_train))
    py_test = np.bincount(y_test) / float(len(y_test))


    trace = (1 - noise_amount) * args.num_classes

    print(f"Generate noise matrix with trace value: {trace}, noise amount: {noise_amount} and sparsity: {sparsity}")
    print(f"py sum: {np.sum(py_train)}")
    print(f"Number of classes: {args.num_classes}")

    noise_matrix_train = generate_noise_matrix_from_trace(
        args.num_classes,
        trace=trace,
        py=py_train,
        frac_zero_noise_rates=sparsity,
        valid_noise_matrix=True,
        seed=internal_seed,
    )

    noisy_labels_train = generate_noisy_labels(y_train, noise_matrix_train)
    return noisy_labels_train


class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def save_document(rootpath, args):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    txtsave_path = os.path.join(rootpath, 'txtsave')
    if not os.path.exists(txtsave_path):
        os.makedirs(txtsave_path)
    txtpath = os.path.join(txtsave_path, f"{time_str}_Noise_{args.noise_amount:.1f}_Sparsity_{args.sparsity:.1f}_{args.dataset}_{args.model}")

    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += f"_nonIID_p_{args.non_iid_prob_class:.1f}_dirich_{args.alpha_dirichlet:.1f}"

    f_acc = open(txtpath + '_acc.txt', 'a')
    return f_acc


def setting_torch(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "./record/"
    return rootpath
