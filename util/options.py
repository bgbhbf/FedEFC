import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac', type=float, default=0.1, help="fration of selected clients in usual training stage")

    parser.add_argument('--gpu', type=int, default=0, help="gpu number")
    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")

    # noise arguments
    parser.add_argument('--sparsity', type=float, default=0.8, help = 'Sparsity of noise matrix')
    parser.add_argument('--noise_amount', type=float, default=0.2, help = 'Noise amount of noise matrix')

    # Method
    parser.add_argument('--method', type=str, default='FedEFC', help="method option:FedAvg/FedProx/FedDyn/FedDitto/confident_learning/forward_correction/FedEFC")

    # FedEFC Parameters
    parser.add_argument('--prestopping', action='store_false', help='Enable prestopping (True if included, False if omitted)')
    parser.add_argument('--gamma_thr', type=int, default=6, help="Persistant to threshold")
    parser.add_argument('--start_monitor', type=int, default=40, help="Maintain round evenif low accuracy")
    parser.add_argument('--weigth_compensate', action='store_true', help="wheter to use weigth compensate" )
    parser.add_argument('--calibrate', action='store_true', help="wheter to calis" )

    # Fed Parameters
    parser.add_argument('--mu', type=float, default=1e-5, help='hyper parameter for feddyn')
    parser.add_argument('--beta', type=float, default=0.0, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")

    # other arguments
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")


    #Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--noniid_type', type=str, default = "dirichlet", help="Non IID case")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.5, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10.0)

    # other arguments
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    return parser.parse_args()

## Constant values
FLOATING_POINT_COMPARISON = 1e-6  # floating point comparison for fuzzy equals
MINI_BATCH_LOCALMODEL = 128
LARGE_VALUE = 1e+10
PERCENTILE = 97