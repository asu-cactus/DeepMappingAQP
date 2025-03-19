from bitarray import bitarray
import numpy as np
from typing import Union
import pdb


class AuxStruct:
    def __init__(self, init_bit_list: list[int], init_list: np.array):
        self.bit_array = bitarray(init_bit_list)
        self.values = init_list
        assert self.bit_array.count(1) == len(init_list)

    def get(self, index: Union[int, np.array], empty_value) -> Union[np.array, float]:
        if not isinstance(index, int):
            # For each integer value in index, get the corresponding value in the array
            return np.array([self.get(i.item(), empty_value) for i in index])
        else:
            # The trainset is a sample, so the query may exceed the length of the bit array
            if index >= len(self.bit_array):
                index = len(self.bit_array) - 1
            # Get the value at the index if the bit is set, else return empty_value
            if self.bit_array[index] == 0:
                return empty_value
            index = self.bit_array.count(1, 0, index)
            return self.values[index]


def get_combined_size(model, aux_struct):
    model_size = sum([p.nelement() * p.element_size() for p in model.parameters()])
    aux_struct_size = aux_struct.bit_array.buffer_info()[1] + aux_struct.values.nbytes
    combined_size_in_bytes = model_size + aux_struct_size
    print(
        f"""
Model size: {model_size / 1024:.2f} KB, AuxStruct size: {aux_struct_size / 1024:.2f} KB
Total size: {combined_size_in_bytes / 1024:.2f} KB
        """
    )
    return combined_size_in_bytes
