from itertools import combinations
import time
from multiprocessing import Pool
from functools import partial
import numpy as np


def get_partitions(n, k=None, max_depth=None):
    partitions = []

    def partition(target, max_value, suffix):
        if max_depth is not None and len(partitions) > max_depth:
            return

        if target == 0:
            if len(suffix) == k:
                partitions.append(suffix)
        else:
            if max_value > 1:
                partition(target, max_value - 1, suffix)
            if max_value <= target:
                partition(target - max_value, max_value, [max_value] + suffix)

    partition(n, n, [])
    return partitions


def combinations_from_partition(arr, lens, max_combs=None):
    all_combinations = []

    def combs(a, lens_index, res):
        if max_combs and len(all_combinations) == max_combs:
            return

        if lens_index < len(lens):
            l = lens[lens_index]
            for c in combinations(a, l):
                if max_combs and len(all_combinations) == max_combs:
                    return

                c = list(c)
                rest = list(set(a) - set(c))
                combs(rest, lens_index + 1, res + [c])  # copy, not inplace
        else:
            all_combinations.append(res)

    combs(arr, 0, [])
    return all_combinations


def line_to_unique(line):
    return tuple(sorted([tuple(sorted(x)) for x in line]))


def drop_duplicated_combinations(combs):
    combs_unique = [line_to_unique(x) for x in combs]
    return list(set(combs_unique))


def get_combinations(seq, k, max_partitions=None, max_combs=None):
    assert len(seq) == len(list(set(seq)))

    all_combs = []
    partitions = get_partitions(len(seq), k, max_depth=max_partitions)

    for partition in partitions:
        print(f'Partition: {partition}, combinations counting..')
        combs_partition = combinations_from_partition(seq, partition, max_combs)
        combs_partition = drop_duplicated_combinations(combs_partition)
        print(f'{len(combs_partition)} combination')
        all_combs.extend(combs_partition)

    return all_combs


def get_labels_count(labels, labels_uniq, features=None):
    counts = []
    for l in labels_uniq:
        label_mask = (labels == l).values
        if features is None:
            cnt = labels[label_mask].shape[0]
        else:
            cnt = features[label_mask].sum()
        counts.append(cnt)
    return counts


def folds_counts_diff(fc, penalty_mult=5):
    fc = np.array(fc)
    diff = fc.std(axis=0).mean()
    penalty = (fc == 0).sum()
    diff += penalty_mult * penalty
    return diff


def comb_diff(y, groups, features, comb):
    y_uniq = list(set(y))
    folds_counts = []
    for fold in comb:
        fold_mask = groups.isin(fold).values
        features_fold = None
        if features is not None:
            features_fold = features.iloc[fold_mask]

        fold_counts = get_labels_count(y.iloc[fold_mask], y_uniq, features_fold)
        folds_counts.append(fold_counts)
    diff = folds_counts_diff(folds_counts)
    return diff


def get_combs_diff(y, groups, combs, n_jobs=None, features=None):
    func = partial(comb_diff, y, groups, features)
    if n_jobs == 0:
        all_diffs = [func(c) for c in combs]
    else:
        pool = Pool(n_jobs)
        all_diffs = pool.map(func, combs)
    return all_diffs


class StratifiedGroupKFold:
    def __init__(self, n_splits=3, max_partitions=1, max_combs=1e6, n_jobs=None):
        self.n_splits = n_splits
        self.max_partitions = max_partitions
        self.max_combs = max_combs
        self.n_jobs = n_jobs

    def split(self, X, y=None, groups=None, features=None):
        unique_groups = list(set(groups))
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        start_time = time.time()
        combs = get_combinations(unique_groups, self.n_splits, self.max_partitions, self.max_combs)
        print(f'Time for calculating combinations: {(time.time()-start_time)/60:.2f} min')
        all_diffs = get_combs_diff(y, groups, combs, self.n_jobs, features)
        best_folds = combs[np.argmin(all_diffs)]

        for fold in best_folds:
            test_inds = np.where(groups.isin(fold))[0]
            train_inds = np.where(~groups.isin(fold))[0]
            yield (train_inds, test_inds)


__all__ = ['StratifiedGroupKFold', 'get_labels_count']