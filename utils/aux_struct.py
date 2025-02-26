from bitarray import bitarray
import numpy as np
from typing import Union
from update import Block
import pdb


class AuxStruct:
    def __init__(self, init_bit_list: list[int], init_list: np.array):
        self.bit_array = bitarray(init_bit_list)
        self.values = init_list
        assert self.bit_array.count(1) == len(init_list)

        size_in_bytes = self.bit_array.buffer_info()[1] + self.values.nbytes
        print(f"Size of AuxStruct in KB: {size_in_bytes / 1024:.2f} KB")

    def get(self, index: Union[int, np.array], empty_value) -> Union[np.array, float]:
        if not isinstance(index, int):
            # For each integer value in index, get the corresponding value in the array
            return np.array([self.get(i.item(), empty_value) for i in index])
        else:
            if self.bit_array[index] == 0:
                return empty_value
            index = self.bit_array.count(1, 0, index)
            return self.values[index]


class AuxStructWithUpdateBlocks:
    def __init__(self, ndarray: np.array, blocks: list[Block]):
        self.array = ndarray
        self.blocks = blocks

    def get(self, index: tuple[int, int]):
        d1, d2 = index
        try:
            return self.array[d1, d2]
        except:
            pdb.set_trace()
