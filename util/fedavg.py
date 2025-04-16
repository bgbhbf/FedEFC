import copy
import torch


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def FedDyn(args, global_model, w_locals, idx_users, controls):

    for idx_user in idx_users:
        if idx_user not in controls:
            controls[idx_user] = {k: torch.zeros_like(param, device=args.device) for k, param in global_model.state_dict().items()}

    w_avg = copy.deepcopy(w_locals[0])

    with torch.no_grad():
        # Perform weight averaging
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
            for i, idx_user in enumerate(idx_users):
                controls[idx_user][k] += global_model.state_dict()[k] - w_locals[i][k]
                w_avg[k] -= args.mu * controls[idx_user][k]
                

    return w_avg, controls