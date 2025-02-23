from dataclasses import dataclass
import numpy as np


@dataclass
class Block:
    start_d1: int
    start_d2: int
    end_d1: int
    end_d2: int
    deltas: np.array


def get_update_ranges(X_min, X_max, resolutions):
    nblocks = 5
    size_percent = 0.1
    size_d1 = size_percent * (X_max[0] - X_min[0]) // resolutions[0] * resolutions[0]
    size_d2 = size_percent * (X_max[1] - X_min[1]) // resolutions[1] * resolutions[1]
    update_ranges = []
    for _ in range(nblocks):
        X_start_d1 = np.random.choice(
            np.arange(X_min[0], X_max[0] - size_d1, resolutions[0])
        )
        X_start_d2 = np.random.choice(
            np.arange(X_min[1], X_max[1] - size_d2, resolutions[1])
        )
        X_end_d1 = X_start_d1 + size_d1
        X_end_d2 = X_start_d2 + size_d2
        update_ranges.append((X_start_d1, X_start_d2, X_end_d1, X_end_d2))
    return update_ranges


def generate_update_queries():
    pass


def convert_index(X_start_d1, X_start_d2, X_end_d1, X_end_d2, resolutions, X_min):
    X_s_d1 = (X_start_d1 - X_min[0]) // resolutions[0]
    X_s_d2 = (X_start_d2 - X_min[1]) // resolutions[1]
    X_e_d1 = (X_end_d1 - X_min[0]) // resolutions[0]
    X_e_d2 = (X_end_d2 - X_min[1]) // resolutions[1]
    return (X_s_d1, X_s_d2, X_e_d1, X_e_d2)


def create_update_blocks(regions):
    blocks = []
    for region in regions:

        # start_d1, start_d2, end_d1, end_d2 = region
        # range_d1 = (end_d1 - start_d1) / resolutions[0]
        # range_d2 = (end_d2 - start_d2) / resolutions[1]
        X_s_d1, X_s_d2, X_e_d1, X_e_d2 = region
        range_d1 = X_e_d1 - X_s_d1
        range_d2 = X_e_d2 - X_s_d2
        deltas = np.zeros((range_d1, range_d2), dtype=np.float32)
        blocks.append(Block(X_s_d1, X_s_d2, X_e_d1, X_e_d2, deltas))

    return blocks
