#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math

import numpy as np

from coremltools import _logger as logger

from .annotate import annotate, class_annotate, delay_type
from .type_bool import bool
from .type_spec import Type


def make_float(width):
    delay_type_float = getattr(delay_type, "fp" + str(width))

    @class_annotate()
    class double:
        _width = width

        def __init__(self, v=0.0):
            self._val = v

        @property
        def val(self):
            return self._val

        @val.setter
        def val(self, v):
            from .type_mapping import (builtin_to_string, nptype_from_builtin,
                                       numpy_type_to_builtin_type)

            if not isinstance(v, np.generic):

                if isinstance(v, np.ndarray) and v.ndim == 0:
                    # Rank zero tensor case. Use as a scalar.
                    self._val = v.item()
                else:
                    raise ValueError(
                        f"Types should have zero-rank ndarray input, got {v} instead."
                    )

            elif isinstance(v, np.floating):
                v_type = numpy_type_to_builtin_type(v.dtype)
                if v_type.get_bitwidth() <= self.get_bitwidth():
                    self._val = v
                else:
                    self._val = v.astype(nptype_from_builtin(self.__class__))
                    logger.warning(
                        "Saving value type of {} into a builtin type of {}, might lose precision!".format(
                            v.dtype, builtin_to_string(self.__class__)
                        )
                    )
            else:
                self._val = v.astype(nptype_from_builtin(self.__class__))
                logger.warning(
                    "Saving value type of {} into a builtin type of {}, might be incompatible or loses precision!".format(
                        v.dtype, builtin_to_string(self.__class__)
                    )
                )

        @classmethod
        def __type_info__(cls):
            return Type("fp" + str(cls._width), python_class=cls)

        @classmethod
        def get_bitwidth(cls):
            return cls._width

        @annotate(delay_type_float, other=delay_type_float)
        def __add__(self, other):
            assert isinstance(other, double)
            return double(self.val + other.val)

        @annotate(delay_type_float, other=delay_type_float)
        def __sub__(self, other):
            assert isinstance(other, double)
            return double(self.val - other.val)

        @annotate(delay_type_float, other=delay_type_float)
        def __mul__(self, other):
            assert isinstance(other, double)
            return double(self.val * other.val)

        @annotate(delay_type_float, other=delay_type_float)
        def __div__(self, other):
            assert isinstance(other, double)
            return double(self.val / other.val)

        @annotate(delay_type_float, other=delay_type_float)
        def __mod__(self, other):
            assert isinstance(other, double)
            return double(self.val % other.val)

        @annotate(delay_type.bool, other=delay_type_float)
        def __lt__(self, other):
            return bool(self.val < other.val)

        @annotate(delay_type.bool, other=delay_type_float)
        def __gt__(self, other):
            return bool(self.val > other.val)

        @annotate(delay_type.bool, other=delay_type_float)
        def __le__(self, other):
            return bool(self.val <= other.val)

        @annotate(delay_type.bool, other=delay_type_float)
        def __ge__(self, other):
            return bool(self.val >= other.val)

        @annotate(delay_type.bool, other=delay_type_float)
        def __eq__(self, other):
            return bool(self.val == other.val)

        @annotate(delay_type.bool, other=delay_type_float)
        def __ne__(self, other):
            return bool(self.val != other.val)

        @annotate(delay_type.bool)
        def __bool__(self):
            return self.val != 0

        @annotate(delay_type.int)
        def __int__(self):
            return int(self)

        @annotate(delay_type_float)
        def __double__(self):
            return float(self.val)

        @annotate(delay_type.str)
        def __str__(self):
            return str(self.val)

        @annotate(delay_type_float)
        def __log__(self):
            return math.log(self.val)

        @annotate(delay_type_float)
        def __exp__(self):
            return math.exp(self.val)

        @annotate(delay_type_float)
        def __neg__(self):
            return double(-self.val)

    double.__name__ = "fp%d" % double.get_bitwidth()
    return double


fp16 = make_float(16)
fp32 = make_float(32)
fp64 = make_float(64)
float = fp32
double = fp64

_FLOAT_TYPES = (fp16, fp32, fp64)

def is_float(t):
    return any(t is i or isinstance(t, i) for i in _FLOAT_TYPES)
