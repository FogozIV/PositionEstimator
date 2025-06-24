import ctypes
import dataclasses
import struct
import utils.SimulatingRobot as sr
import numpy as np
import matplotlib.pyplot as plt

@dataclasses.dataclass
class CalibrationDataLegacy:
    left: int
    right: int
    @staticmethod
    def get_length():
        return 2* 4
    @staticmethod
    def from_bytes(data: bytes) -> 'CalibrationDataLegacy':
        if len(data) < CalibrationDataLegacy.get_length():
            raise ValueError(
                f'Not enough bytes to unpack CalibrationDataLegacy (need {CalibrationDataLegacy.get_length()} bytes)')
        values = struct.unpack('>2l', data)
        return CalibrationDataLegacy(*values)

@dataclasses.dataclass
class CalibrationData:
    left: int
    right: int
    dt: float
    @staticmethod
    def get_length():
        return 2* 4 + 1 * 8
    @staticmethod
    def from_bytes(data: bytes) -> 'CalibrationData':
        if len(data) < CalibrationData.get_length():
            raise ValueError(
                f'Not enough bytes to unpack CalibrationData (need {CalibrationData.get_length()} bytes)')
        values = struct.unpack('>2l', data)
        values = list(values)
        values.append(struct.unpack('>1Q', data)[0])
        return CalibrationData(*values)


def read_file(filename, class_type=CalibrationData):
    result = []
    indexes = []
    flag_1 = ctypes.c_int32(0xCAFEBEEF).value
    flag_2 = ctypes.c_int32(0xBEEFCAFE).value
    with open(filename, "rb") as f:
        while data := f.read(class_type.get_length()):
            result.append(class_type.from_bytes(data))
            if result[-1].left == flag_1 and result[-1].right == flag_2:
                indexes.append(len(result) - 1)
                result.pop()
    return indexes, result




