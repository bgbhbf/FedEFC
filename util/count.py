from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]
FLOATING_POINT_COMPARISON = 1e-6  
CONFIDENT_THRESHOLDS_LOWER_BOUND = (
    2 * FLOATING_POINT_COMPARISON
)  
TINY_VALUE = 1e-100
CLIPPING_LOWER_BOUND = 1e-6  


def calibrate_confident_joint(
    confident_joint: np.ndarray, labels: LabelLike, *, multi_label: bool = False
) -> np.ndarray:

    num_classes = len(confident_joint)
    unique_classes, counts = np.unique(labels, return_counts=True)

    if num_classes is None or num_classes == len(unique_classes):
      label_counts = counts
    else:

      total_counts = np.zeros(num_classes, dtype=int)
      labels_are_integers = np.issubdtype(np.array(labels).dtype, np.integer)
      count_ids = unique_classes if labels_are_integers else slice(len(unique_classes))
      total_counts[count_ids] = counts
      label_counts = total_counts
    calibrated_cj = (
        confident_joint.T
        / np.clip(confident_joint.sum(axis=1), a_min=TINY_VALUE, a_max=None)
        * label_counts
    ).T
    calibrated_cj = (
        calibrated_cj
        / np.clip(np.sum(calibrated_cj), a_min=TINY_VALUE, a_max=None)
        * sum(label_counts)
    )


    count_matrix = np.apply_along_axis(
        func1d=round_preserving_sum,
        axis=1,
        arr=calibrated_cj,
    ).astype(int)

    return count_matrix

def round_preserving_sum(iterable) -> np.ndarray:

    floats = np.asarray(iterable, dtype=float)
    ints = floats.round()
    orig_sum = np.sum(floats).round()
    int_sum = np.sum(ints).round()
    while abs(int_sum - orig_sum) > FLOATING_POINT_COMPARISON:
        diff = np.round(orig_sum - int_sum)
        increment = -1 if int(diff < 0.0) else 1
        changes = min(int(abs(diff)), len(iterable))
        # Orders indices by difference. Increments # of changes.
        indices = np.argsort(floats - ints)[::-increment][:changes]
        for i in indices:
            ints[i] = ints[i] + increment
        int_sum = np.sum(ints).round()
    return ints.astype(int)

def compute_count_matrix(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    calibrate: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    
    labels = np.asarray(labels)
    thresholds = get_confident_thresholds(labels, pred_probs)
    thresholds = np.asarray(thresholds)
    pred_probs_bool = pred_probs >= thresholds - 1e-6
    num_confident_bins = pred_probs_bool.sum(axis=1)
    at_least_one_confident = num_confident_bins > 0 # 1 satisfying threshold
    more_than_one_confident = num_confident_bins > 1 # more 2 satisfying threshold
    pred_probs_argmax = pred_probs.argmax(axis=1)
    confident_argmax = pred_probs_bool.argmax(axis=1) 
    true_label_guess = np.where(
        more_than_one_confident,
        pred_probs_argmax,
        confident_argmax,
    )
    true_labels_confident = true_label_guess[at_least_one_confident]
    labels_confident = labels[at_least_one_confident]
    confident_joint = confusion_matrix(
        y_true=true_labels_confident,
        y_pred=labels_confident,
        labels=range(pred_probs.shape[1]),
    ).T

    np.fill_diagonal(confident_joint, confident_joint.diagonal().clip(min=1))
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)

    return confident_joint


def get_confident_thresholds(
    labels: LabelLike,
    pred_probs: np.ndarray,
) -> np.ndarray:

    labels = labels_to_array(labels)
    all_classes = range(pred_probs.shape[1])
    unique_classes = set(labels)
    BIG_VALUE = 2 * pred_probs.max()
    confident_thresholds = [
        np.mean(pred_probs[:, k][labels == k]) if k in unique_classes else BIG_VALUE
        for k in all_classes
    ]
    confident_thresholds = np.clip(
        confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
    )
    return confident_thresholds


def labels_to_array(y: Union[LabelLike, np.generic]) -> np.ndarray:
    if isinstance(y, pd.Series):
        y_series: np.ndarray = y.to_numpy()
        return y_series
    elif isinstance(y, pd.DataFrame):
        y_arr = y.values
        assert isinstance(y_arr, np.ndarray)
        if y_arr.shape[1] != 1:
            raise ValueError("labels must be one dimensional.")
        return y_arr.flatten()
    else:  
        try:
            return np.asarray(y)
        except:
            raise ValueError(
                "List of labels must be convertable to 1D numpy array via: np.ndarray(labels)."
            )