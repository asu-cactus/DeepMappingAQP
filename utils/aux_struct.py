from bitarray import bitarray
import numpy as np
from typing import Union


class AuxStruct:
    def __init__(self, init_bit_list: list[int], init_list: list[float]):
        self.bit_array = bitarray(init_bit_list)
        self.values = np.array(init_list, dtype=np.float32)
        assert self.bit_array.count(1) == len(init_list)

    def get(self, index: int) -> Union[float, None]:
        if self.bit_array[index] == 0:
            return None
        index = self.bit_array.count(1, 0, index + 1)
        return self.values[index]
