
import numpy as np
import torch
from torchvision import datasets, transforms
from util.sampling import non_iid_dirichlet_sampling, iid_sampling
import torch.utils
from torch.utils.data import Subset



def get_dataset(args):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'cifar10':
        data_path = './cifar10'
        args.num_classes = 10
        args.model = 'resnet18'

        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset == 'cifar100':
        data_path = './cifar100'
        args.num_classes = 100
        args.model = 'resnet34'

        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset == 'mnist':
        data_path = './mnist'
        args.num_classes = 10
        args.model = 'cnn'

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        dict_users = iid_sampling(n_train, args.num_users, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
        

    return dataset_train, dataset_test, dict_users
    

class SubsetWithTargets(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if isinstance(indices, set):
            indices = list(indices)
        self.indices = indices 
        
        if hasattr(dataset, 'targets'):
            self.targets = [dataset.targets[i] for i in indices]
        else:
            raise AttributeError(f"Dataset has no attribute 'targets'")

    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]
        return data, target

    def __len__(self):
        return len(self.indices)
 