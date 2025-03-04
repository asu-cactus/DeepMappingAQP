from dataclasses import dataclass
import random
from typing import Sequence
import math
import pdb
import numpy as np

# Set random seed for reproducibility
random.seed(42)

NDECIMALS = 1


@dataclass
class RangeBlock:
    start: int
    end: int
    deltas: Sequence[float]


@dataclass
class Range:
    start: int
    end: int


@dataclass
class UpdateEntry:
    point: float
    value: float


class Update:
    def __init__(self, args, ranges):
        self.size = 0
        self.buffer_capacity = args.buffer_capacity
        self.resolution = args.resolutions[0]
        self.blocks = self._initialize_blocks(ranges)
        self.buffer = self._initialize_buffer()

    def _initialize_blocks(self, ranges):
        blocks = []
        for r in ranges:
            deltas = np.zeros(round((r.end - r.start) / self.resolution))
            blocks.append(RangeBlock(r.start, r.end, deltas))
        return blocks

    def _initialize_buffer(self):
        buffer = []
        for block in self.blocks:
            buffer.append(np.zeros_like(block.deltas))
        return buffer

    def update(self, entry: UpdateEntry):
        self.size += 1

        for block, temp_delta in zip(self.blocks, self.buffer):
            if block.start <= entry.point <= block.end:
                idx = round((entry.point - block.start) / self.resolution)
                temp_delta[idx] += entry.value
                break

        if self.size >= self.buffer_capacity:
            self._flush_buffer()
            self.size = 0

    def _flush_buffer(self):
        for block, temp_delta in zip(self.blocks, self.buffer):
            # Cumulative sum
            block.deltas += np.cumsum(temp_delta)
            temp_delta.fill(0)


def get_update_ranges(X_min, X_max, resolution):
    nblocks = 5
    range_percent = 0.05

    ranges = []
    while nblocks > 0:
        range_size = round(range_percent * (X_max - X_min))

        X_start = random.uniform(X_min, X_max - range_size) // resolution * resolution
        X_end = X_start + range_size

        # HARD CODED: round to 1 decimal place
        X_start, X_end = round(X_start, NDECIMALS), round(X_end, NDECIMALS)
        assert X_min <= X_start <= X_max - range_size

        # Check if the range is already in the ranges
        for r in ranges:
            if r.start <= X_start <= r.end or r.start <= X_end <= r.end:
                break
        else:
            ranges.append(Range(X_start, X_end))
            nblocks -= 1
    return ranges


def generate_update_queries(value_min, value_max, ranges, nqueries, resolution):
    for _ in range(nqueries):
        r = random.choice(ranges)
        update_point = random.uniform(r.start, r.end) // resolution * resolution
        update_point = round(update_point, NDECIMALS)
        update_value = random.uniform(value_min, value_max)
        yield UpdateEntry(update_point, update_value)


def query_updates(range_query, update: Update):
    range_start, range_end = range_query
    overall_change = 0
    for block in update.blocks:
        start_within = range_start <= block.start <= range_end
        end_within = range_start <= block.end <= range_end
        if range_start >= block.start and range_end <= block.end:
            # When the range is within the block
            idx1 = round((range_start - block.start) / update.resolution)
            idx2 = round((range_end - block.start) / update.resolution)
            return block.deltas[idx2] - block.deltas[idx1 - 1]

        elif start_within and end_within:
            print("Range covers the entire block")
            # When the range covers the entire block
            overall_change += block.deltas[-1]
        elif start_within:
            idx = round((range_end - block.start) / update.resolution)
            overall_change += block.deltas[idx]
        elif end_within:
            idx = round((range_start - block.start) / update.resolution)
            overall_change += block.deltas[-1] - block.deltas[idx - 1]
    return overall_change


def generate_range_queries(X_min, X_max, range_size, nqueries):

    for _ in range(nqueries):
        range_start = random.uniform(X_min, X_max - range_size)
        range_end = range_start + range_size
        range_start = round(range_start, NDECIMALS)
        range_end = round(range_end, NDECIMALS)
        yield (range_start, range_end)


@dataclass
class Arguments:
    buffer_capacity: int
    resolutions: Sequence[float]


class NaiveUpdate:
    def __init__(self):
        self.values = {}

    def update(self, entry: UpdateEntry):
        key = round(entry.point, NDECIMALS)
        self.values[key] = self.values.get(key, 0) + entry.value

    def query(self, range_query):
        range_start, range_end = range_query
        overall_change = 0
        for key, value in self.values.items():
            if range_start <= key <= range_end:
                print(f"point: {key}, value: {value} fall within the range")
                overall_change += value
        return overall_change


if __name__ == "__main__":
    X_min = 0
    X_max = 1000
    value_min = -100
    value_max = 100

    nqueries = 5000

    args = Arguments(buffer_capacity=nqueries, resolutions=[0.1])
    resolution = args.resolutions[0]

    ranges = get_update_ranges(X_min, X_max, resolution)
    print("Ranges:")
    print(ranges)

    update_queries = list(
        generate_update_queries(value_min, value_max, ranges, nqueries, resolution)
    )
    print("Update queries:")
    print(update_queries)

    update = Update(args, ranges)
    for query in update_queries:
        update.update(query)

    range_size = 0.1 * (X_max - X_min)
    nqueries = 20
    range_queries = list(generate_range_queries(X_min, X_max, range_size, nqueries))

    naive_update = NaiveUpdate()
    for query in update_queries:
        naive_update.update(query)

    for range_query in range_queries:
        print(f"Range query: {range_query}")
        update_res = query_updates(range_query, update)
        naive_update_res = naive_update.query(range_query)

        print(f"Fancy Update result: {update_res}")
        print(f"Naive update result: {naive_update_res}")
        if not math.isclose(update_res, naive_update_res, rel_tol=1e-5):
            pdb.set_trace()

# point: 423.0, value: -79.85975838632683 fall within the range
# range_query: (358.9, 458.9)
# block start: 399.5, block end: 449.4
