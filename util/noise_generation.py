from typing import Optional

import numpy as np
from util.options import FLOATING_POINT_COMPARISON


def value_counts(x, *, num_classes: Optional[int] = None) -> np.ndarray:
    unique_classes, counts = np.unique(x, return_counts=True)

    if num_classes is None or num_classes == len(unique_classes):
        return counts

    labels_are_integers = np.issubdtype(np.array(x).dtype, np.integer)
    if labels_are_integers and num_classes <= np.max(unique_classes):
        raise ValueError(f"Required: num_classes > max(x), but {num_classes} <= {np.max(x)}.")


    total_counts = np.zeros(num_classes, dtype=int)
    count_ids = unique_classes if labels_are_integers else slice(len(unique_classes))
    total_counts[count_ids] = counts

    return total_counts

def noise_matrix_is_valid(noise_matrix, py, *, verbose=False) -> bool:
    K = len(py)

    N = float(10000)

    ps = np.dot(noise_matrix, py) 
    joint_noise = np.multiply(noise_matrix, py)  
    if not (abs(joint_noise.sum() - 1.0) < FLOATING_POINT_COMPARISON):
        return False

    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print(
                "E1E2/C",
                round(E1 * E2 / C),
                "E1",
                round(E1),
                "E2",
                round(E2),
                "C",
                round(C),
                "|",
                round(E1 * E2 / C + E1 + E2 + C),
                "|",
                round(E1 * E2 / C),
                "<",
                round(O),
            )
            print(
                round(ps[i] * py[i]),
                "<",
                round(joint_noise[i][i]),
                ":",
                ps[i] * py[i] < joint_noise[i][i],
            )

        if not (ps[i] * py[i] < joint_noise[i][i]):
            return False

    return True


def generate_noisy_labels(true_labels, noise_matrix) -> np.ndarray:

    true_labels = np.asarray(true_labels)
    K = len(noise_matrix)
    py = value_counts(true_labels) / float(len(true_labels))
    count_joint = (noise_matrix * py * len(true_labels)).astype(int)
    np.fill_diagonal(count_joint, 0)

    # Generate labels
    labels = np.array(true_labels)
    for k in range(K):  
        labels_per_class = np.where(count_joint[:, k] != 0)[0]
        label_counts = count_joint[labels_per_class, k]
        noise = [labels_per_class[i] for i, c in enumerate(label_counts) for z in range(c)]
        idx_flip = np.where((labels == k) & (true_labels == k))[0]
        if len(idx_flip) and len(noise) and len(idx_flip) >= len(noise):  # pragma: no cover
            labels[np.random.choice(idx_flip, len(noise), replace=False)] = noise


    return labels


def generate_noise_matrix_from_trace(
        K: object,
        trace: object,
    *,
        max_trace_prob: object = 1.0,
        min_trace_prob: object = 1e-5,
        max_noise_rate: object = 1 - 1e-5,
        min_noise_rate: object = 0.0,
        valid_noise_matrix: object = True,
        py: object = None,
        frac_zero_noise_rates: object = 0.0, #sparsity
        seed: object = 0,
        max_iter: object = 10000,
) -> Optional[np.ndarray]:
    
    # Utilize cleanlab lib.

    if valid_noise_matrix and trace <= 1:
        raise ValueError(
            "trace = {}. trace > 1 is necessary for a".format(trace)
            + " valid noise matrix to be returned (valid_noise_matrix == True)"
        )

    if valid_noise_matrix and py is None and K > 2:
        raise ValueError(
            "py must be provided (not None) if the input parameter" + " valid_noise_matrix == True"
        )

    if K <= 1:
        raise ValueError("K must be >= 2, but K = {}.".format(K))

    if max_iter < 1:
        return None

    np.random.seed(seed)

    if K == 2:
        if frac_zero_noise_rates >= 0.5: 
            noise_mat = np.array(
                [
                    [1.0, 1 - (trace - 1.0)],
                    [0.0, trace - 1.0],
                ]
            )
            return noise_mat if np.random.rand() > 0.5 else np.rot90(noise_mat, k=2)
        else:  
            diag = generate_n_rand_probabilities_that_sum_to_m(2, trace)
            noise_matrix = np.array(
                [
                    [diag[0], 1 - diag[1]],
                    [1 - diag[0], diag[1]],
                ]
            )
            return noise_matrix

    for z in range(max_iter):
        noise_matrix = np.zeros(shape=(K, K))

        nm_diagonal = generate_n_rand_probabilities_that_sum_to_m(
            n=K,
            m=trace,
            max_prob=max_trace_prob,
            min_prob=min_trace_prob,
        )
        np.fill_diagonal(noise_matrix, nm_diagonal)


        num_col_with_noise = K - np.count_nonzero(1 == nm_diagonal)
        num_zero_noise_rates = int(K * (K - 1) * frac_zero_noise_rates)
        num_zero_noise_rates -= (K - num_col_with_noise) * (K - 1)
        num_zero_noise_rates = np.maximum(num_zero_noise_rates, 0)  # Prevent negative
        num_zero_noise_rates_per_col = (
            randomly_distribute_N_balls_into_K_bins(
                N=num_zero_noise_rates,
                K=num_col_with_noise,
                max_balls_per_bin=K - 2,
                min_balls_per_bin=0,
            )
            if K > 2
            else np.array([0, 0])
        )  # Special case when K == 2
        stack_nonzero_noise_rates_per_col = list(K - 1 - num_zero_noise_rates_per_col)[::-1]
        # Randomly generate noise rates for columns with noise.
        for col in np.arange(K)[nm_diagonal != 1]:
            num_noise = stack_nonzero_noise_rates_per_col.pop()
            # Generate num_noise noise_rates for the given column.
            noise_rates_col = list(
                generate_n_rand_probabilities_that_sum_to_m(
                    n=num_noise,
                    m=1 - nm_diagonal[col],
                    max_prob=max_noise_rate,
                    min_prob=min_noise_rate,
                )
            )
            rows = np.random.choice(
                [row for row in range(K) if row != col], num_noise, replace=False
            )
            for row in rows:
                noise_matrix[row][col] = noise_rates_col.pop()
        if not valid_noise_matrix or noise_matrix_is_valid(noise_matrix, py):
            return noise_matrix

    return None


def generate_n_rand_probabilities_that_sum_to_m(
    n,
    m,
    *,
    max_prob=1.0,
    min_prob=0.0,
) -> np.ndarray:

    if n == 0:
        return np.array([])
    if (max_prob + FLOATING_POINT_COMPARISON) < m / float(n):
        raise ValueError(
            "max_prob must be greater or equal to m / n, but "
            + "max_prob = "
            + str(max_prob)
            + ", m = "
            + str(m)
            + ", n = "
            + str(n)
            + ", m / n = "
            + str(m / float(n))
        )
    if min_prob > (m + FLOATING_POINT_COMPARISON) / float(n):
        raise ValueError(
            "min_prob must be less or equal to m / n, but "
            + "max_prob = "
            + str(max_prob)
            + ", m = "
            + str(m)
            + ", n = "
            + str(n)
            + ", m / n = "
            + str(m / float(n))
        )

    result = np.random.dirichlet(np.ones(n)) * m

    min_val = min(result)
    max_val = max(result)
    while max_val > (max_prob + FLOATING_POINT_COMPARISON):
        new_min = min_val + (max_val - max_prob)
        adjustment = (max_prob - new_min) * np.random.rand()
        result[np.argmin(result)] = new_min + adjustment
        result[np.argmax(result)] = max_prob - adjustment
        min_val = min(result)
        max_val = max(result)

    min_val = min(result)
    max_val = max(result)
    while min_val < (min_prob - FLOATING_POINT_COMPARISON):
        min_val = min(result)
        max_val = max(result)
        new_max = max_val - (min_prob - min_val)
        adjustment = (new_max - min_prob) * np.random.rand()
        result[np.argmax(result)] = new_max - adjustment
        result[np.argmin(result)] = min_prob + adjustment
        min_val = min(result)
        max_val = max(result)

    return result


def randomly_distribute_N_balls_into_K_bins(
    N,  # int
    K,  # int
    *,
    max_balls_per_bin=None,
    min_balls_per_bin=None,
) -> np.ndarray:

    if N == 0:
        return np.zeros(K, dtype=int)
    if max_balls_per_bin is None:
        max_balls_per_bin = N
    else:
        max_balls_per_bin = min(max_balls_per_bin, N)
    if min_balls_per_bin is None:
        min_balls_per_bin = 0
    else:
        min_balls_per_bin = min(min_balls_per_bin, N / K)
    if N / float(K) > max_balls_per_bin:
        N = max_balls_per_bin * K

    arr = np.round(
        generate_n_rand_probabilities_that_sum_to_m(
            n=K,
            m=1,
            max_prob=max_balls_per_bin / float(N),
            min_prob=min_balls_per_bin / float(N),
        )
        * N
    )
    while sum(arr) != N:
        while sum(arr) > N:  # pragma: no cover
            arr[np.argmax(arr)] -= 1
        while sum(arr) < N:
            arr[np.argmin(arr)] += 1
    return arr.astype(int)




