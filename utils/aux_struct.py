from bitarray import bitarray
import numpy as np
from typing import Union


class AuxStruct:
    def __init__(self, init_bit_list: list[int], init_list: np.array):
        self.bit_array = bitarray(init_bit_list)
        self.values = init_list
        assert self.bit_array.count(1) == len(init_list)

    def get(self, index: int, empty_value) -> Union[float, None]:
        if self.bit_array[index] == 0:
            return empty_value
        index = self.bit_array.count(1, 0, index)
        return self.values[index]
