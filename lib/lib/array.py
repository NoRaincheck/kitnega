"""
Pure-Python ndarray class.

Influenced by
- https://github.com/wadetb/tinynumpy
- https://github.com/tinygrad/tinygrad
"""

import ctypes
from collections.abc import Iterable
from math import sqrt

_DTYPE_INFO = {
    "float64": ("f8", ctypes.c_double, 8),
    "float32": ("f4", ctypes.c_float, 4),
    "int64": ("q", ctypes.c_int64, 8),
    "int32": ("i", ctypes.c_int32, 4),
    "int16": ("h", ctypes.c_int16, 2),
    "int8": ("b", ctypes.c_int8, 1),
    "uint64": ("Q", ctypes.c_uint64, 8),
    "uint32": ("I", ctypes.c_uint32, 4),
    "uint16": ("H", ctypes.c_uint16, 2),
    "uint8": ("B", ctypes.c_uint8, 1),
    "bool": ("?", ctypes.c_bool, 1),
}

_KNOWN_DTYPES = set(_DTYPE_INFO)


def _convert_dtype(dtype, fmt="dtype"):
    if dtype not in _DTYPE_INFO:
        raise TypeError(f"data type {dtype!r} not understood")
    if fmt == "dtype":
        return dtype
    if fmt == "short":
        return _DTYPE_INFO[dtype][0]
    if fmt == "ctypes":
        return _DTYPE_INFO[dtype][1]
    raise ValueError(f"unknown format: {fmt!r}")


def _itemsize(dtype):
    return _DTYPE_INFO[dtype][2]


def _strides_for_shape(shape, itemsize):
    strides = []
    for s in reversed(shape):
        strides.append(itemsize)
        itemsize *= s
    return tuple(reversed(strides))


def _size_for_shape(shape):
    size = 1
    for s in shape:
        size *= s
    return size


def _ceildiv(a, b):
    return -(-a // b)


def _get_step(view):
    expected = view.itemsize
    for i in range(view.ndim - 1, -1, -1):
        if view.strides[i] != expected:
            return 0
        expected *= view.shape[i]
    return 1


def _key_for_index(index, shape):
    key = []
    for s in reversed(shape):
        key.append(index % s)
        index //= s
    return tuple(reversed(key))


def _flatten(obj):
    for item in obj:
        if isinstance(item, (tuple, list)):
            yield from _flatten(item)
        else:
            yield item


class ndarray:
    __slots__ = ("_dtype", "_shape", "_strides", "_itemsize", "_offset", "_base", "_data")

    def __init__(self, shape, dtype="float64", buffer=None, offset=0, strides=None):
        if not isinstance(shape, Iterable):
            raise TypeError("shape must be an iterable")
        shape = tuple(shape)
        if not all(isinstance(s, int) for s in shape):
            raise TypeError("shape must be a tuple of ints")
        self._shape = shape

        dtype = dtype or "float64"
        if dtype not in _KNOWN_DTYPES:
            raise TypeError(f"data type {dtype!r} not understood")
        self._dtype = dtype
        self._itemsize = _itemsize(dtype)

        if buffer is None:
            self._base = None
            self._offset = 0
            self._strides = _strides_for_shape(shape, self._itemsize)
        else:
            if isinstance(buffer, ndarray) and buffer.base is not None:
                buffer = buffer.base
            self._base = buffer
            if isinstance(buffer, ndarray):
                buffer = buffer.data
            if not isinstance(offset, int) or offset < 0:
                raise TypeError("offset must be a non-negative int")
            self._offset = offset
            if strides is None:
                strides = _strides_for_shape(shape, self._itemsize)
            if not isinstance(strides, tuple) or len(strides) != len(shape):
                raise TypeError("strides must be a tuple matching shape length")
            if not all(isinstance(s, int) for s in strides):
                raise TypeError("strides must be a tuple of ints")
            self._strides = strides

        buffersize = self._strides[0] * self._shape[0] // self._itemsize
        buffersize += self._offset
        BufferClass = _convert_dtype(dtype, "ctypes") * buffersize

        if buffer is None:
            self._data = BufferClass()
        elif isinstance(buffer, ctypes.Array):
            self._data = BufferClass.from_address(ctypes.addressof(buffer))
        else:
            self._data = BufferClass.from_buffer(buffer)

    @property
    def __array_interface__(self):
        typestr = "<" + _convert_dtype(self.dtype, "short")
        readonly = False
        if isinstance(self._data, ctypes.Array):
            ptr = ctypes.addressof(self._data)
        elif hasattr(self._data, "buffer_info"):
            ptr = self._data.buffer_info()[0]
        else:
            raise TypeError("cannot get address to underlying array data")
        ptr += self._offset * self.itemsize
        return dict(
            version=3,
            shape=self.shape,
            typestr=typestr,
            descr=[("", typestr)],
            data=(ptr, readonly),
            strides=self.strides,
        )

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        offset, shape, strides = self._index_helper(key)
        if not shape:
            return self._data[offset]
        return ndarray(shape, self.dtype, offset=offset, strides=strides, buffer=self)

    def __setitem__(self, key, value):
        offset, shape, strides = self._index_helper(key)
        if not shape:
            self._data[offset] = value
            return
        view = ndarray(shape, self.dtype, offset=offset, strides=strides, buffer=self)
        if isinstance(value, (int, float)):
            value_list = [value] * view.size
        elif isinstance(value, (tuple, list)):
            value_list = list(_flatten(value))
        else:
            if not isinstance(value, ndarray):
                value = array(value, copy=False)
            value_list = value._toflatlist()
        if view.size != len(value_list):
            raise ValueError("number of elements in source does not match target")
        if view.dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"):
            value_list = [int(v) for v in value_list]
        subviews = [view]
        value_index = 0
        while subviews:
            subview = subviews.pop(0)
            step = _get_step(subview)
            if step:
                block = value_list[value_index : value_index + subview.size]
                s = slice(subview._offset, subview._offset + subview.size * step, step)
                view._data[s] = block
                value_index += subview.size
            else:
                for i in range(subview.shape[0]):
                    subviews.append(subview[i])

    def __float__(self):
        if self.size == 1:
            return float(self._data[self._offset])
        raise TypeError("only length-1 arrays can be converted to scalar")

    def __int__(self):
        if self.size == 1:
            return int(self._data[self._offset])
        raise TypeError("only length-1 arrays can be converted to scalar")

    def __repr__(self):
        if self.size > 100:
            shapestr = "x".join(str(s) for s in self.shape)
            return f"<ndarray {shapestr} {self.dtype} at 0x{id(self):x}>"
        return self._repr_full()

    def _repr_full(self):
        def _repr_r(s, axis, offset):
            axisindent = min(2, max(0, self.ndim - axis - 1))
            if axis < len(self.shape):
                s += "["
                for k_index, k in enumerate(range(self.shape[axis])):
                    if k_index > 0:
                        s += ("\n       " + " " * axis) * axisindent
                    offset_ = offset + k * self._strides[axis] // self.itemsize
                    s = _repr_r(s, axis + 1, offset_)
                    if k_index < self.shape[axis] - 1:
                        s += ", "
                s += "]"
            else:
                r = repr(self._data[offset])
                if "." in r:
                    r = " " + r
                    if r.endswith(".0"):
                        r = r[:-1]
                s += r
            return s

        s = _repr_r("", 0, self._offset)
        if self.dtype not in ("float64", "int32"):
            return f"array({s}, dtype='{self.dtype}')"
        return f"array({s})"

    def __eq__(self, other):
        if not isinstance(other, ndarray):
            return NotImplemented
        if self.shape != other.shape:
            return False
        out = empty(self.shape, "bool")
        out[:] = [i1 == i2 for (i1, i2) in zip(self.flat, other.flat)]
        return out

    def __add__(self, other):
        if isinstance(other, (int, float)):
            out = empty(self.shape, self.dtype)
            out[:] = [x + other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i + j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            out = empty(self.shape, self.dtype)
            out[:] = [x - other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i - j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            out = empty(self.shape, self.dtype)
            out[:] = [other - x for x in self._data]
            return out
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = empty(self.shape, self.dtype)
            out[:] = [x * other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i * j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            out = empty(self.shape, self.dtype)
            out[:] = [x / other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i / j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            out = empty(self.shape, self.dtype)
            out[:] = [x // other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i // j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, (int, float)):
            out = empty(self.shape, self.dtype)
            out[:] = [x % other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i % j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            out = empty(self.shape, self.dtype)
            out[:] = [x ** other for x in self._data]
            return out
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            out = empty(self.shape, self.dtype)
            out[:] = [i ** j for (i, j) in zip(self.flat, other.flat)]
            return out
        return NotImplemented

    def __iadd__(self, other):
        self[:] = self + other
        return self

    def __isub__(self, other):
        self[:] = self - other
        return self

    def __imul__(self, other):
        self[:] = self * other
        return self

    def __itruediv__(self, other):
        self[:] = self / other
        return self

    def __ifloordiv__(self, other):
        self[:] = self // other
        return self

    def __imod__(self, other):
        self[:] = self % other
        return self

    def __ipow__(self, other):
        self[:] = self ** other
        return self

    def _index_helper(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        axis = 0
        shape = []
        strides = []
        offset = self._offset

        for k in key:
            axissize = self._shape[axis]
            if isinstance(k, int):
                if k >= axissize:
                    raise IndexError(f"index {k} is out of bounds for axis {axis} with size {axissize}")
                offset += k * self._strides[axis] // self.itemsize
                axis += 1
            elif isinstance(k, slice):
                start, stop, step = k.indices(self.shape[axis])
                shape.append(_ceildiv(stop - start, step))
                strides.append(step * self._strides[axis])
                offset += start * self._strides[axis] // self.itemsize
                axis += 1
            elif k is Ellipsis:
                raise TypeError("ellipsis are not supported")
            elif k is None:
                shape.append(1)
                stride = 1
                for s in self._strides[axis:]:
                    stride *= s
                strides.append(stride)
            else:
                raise TypeError("key elements must be instances of int or slice")

        shape.extend(self.shape[axis:])
        strides.extend(self._strides[axis:])
        return offset, tuple(shape), tuple(strides)

    def _toflatlist(self):
        value_list = []
        subviews = [self]
        while subviews:
            subview = subviews.pop(0)
            if not isinstance(subview, ndarray):
                value_list.append(subview)
                continue
            step = _get_step(subview)
            if step:
                s = slice(subview._offset, subview._offset + subview.size * step, step)
                value_list += self._data[s]
            else:
                for i in range(subview.shape[0]):
                    subviews.append(subview[i])
        return value_list

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return _size_for_shape(self._shape)

    @property
    def nbytes(self):
        return self.size * self.itemsize

    def _get_shape(self):
        return self._shape

    def _set_shape(self, newshape):
        if newshape == self.shape:
            return
        if self.size != _size_for_shape(newshape):
            raise ValueError("total size of new array must be unchanged")
        self._shape = tuple(newshape)
        self._strides = _strides_for_shape(self._shape, self.itemsize)

    shape = property(_get_shape, _set_shape)

    @property
    def strides(self):
        return self._strides

    @property
    def dtype(self):
        return self._dtype

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def base(self):
        return self._base

    @property
    def data(self):
        return self._data

    @property
    def flat(self):
        subviews = [self]
        while subviews:
            subview = subviews.pop(0)
            if not isinstance(subview, ndarray):
                yield subview
                continue
            step = _get_step(subview)
            if step:
                s = slice(subview._offset, subview._offset + subview.size * step, step)
                for i in self._data[s]:
                    yield i
            else:
                for i in range(subview.shape[0]):
                    subviews.append(subview[i])

    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose()

    def fill(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("value must be int or float")
        self[:] = value

    def copy(self):
        out = empty(self.shape, self.dtype)
        out[:] = self
        return out

    def flatten(self):
        out = empty((self.size,), self.dtype)
        out[:] = self
        return out

    def ravel(self):
        return self.reshape((self.size,))

    def reshape(self, newshape):
        out = self.view()
        try:
            out.shape = newshape
        except (AttributeError, ValueError):
            out = self.copy()
            out.shape = newshape
        return out

    def transpose(self):
        if self.ndim < 2:
            return self.view()
        shape = self.shape[::-1]
        out = empty(shape, self.dtype)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                out[j, i] = self[i, j]
        return out

    def astype(self, dtype):
        out = empty(self.shape, dtype)
        out[:] = self
        return out

    def view(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if dtype == self.dtype:
            return ndarray(self.shape, dtype, buffer=self, offset=self._offset, strides=self.strides)
        if self.ndim == 1:
            itemsize = _itemsize(dtype)
            size = self.nbytes // itemsize
            offsetinbytes = self._offset * self.itemsize
            offset = offsetinbytes // itemsize
            return ndarray((size,), dtype, buffer=self, offset=offset)
        raise ValueError("new type not compatible with array")

    def all(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        return all(self.flat)

    def any(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        return any(self.flat)

    def min(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        return min(self.flat)

    def max(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        return max(self.flat)

    def sum(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        return sum(self.flat)

    def prod(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        p = 1.0
        for i in self.flat:
            p *= float(i)
        return p

    def ptp(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        mn = self._data[self._offset]
        mx = mn
        for i in self.flat:
            if i > mx:
                mx = i
            if i < mn:
                mn = i
        return mx - mn

    def mean(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        return self.sum() / self.size

    def argmax(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        r = self._data[self._offset]
        r_index = 0
        for i_index, i in enumerate(self.flat):
            if i > r:
                r = i
                r_index = i_index
        return r_index

    def argmin(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        r = self._data[self._offset]
        r_index = 0
        for i_index, i in enumerate(self.flat):
            if i < r:
                r = i
                r_index = i_index
        return r_index

    def var(self, axis=None):
        if axis is not None:
            raise TypeError("axis argument is not supported")
        m = self.mean()
        acc = 0.0
        for x in self.flat:
            acc += abs(x - m) ** 2
        return acc / self.size

    def std(self, axis=None):
        return sqrt(self.var(axis))


def array(object, dtype=None, copy=True):
    if isinstance(object, ndarray):
        if dtype is None:
            dtype = object.dtype
        if copy:
            out = empty(object.shape, dtype)
            out[:] = object
            return out
        elif dtype == object.dtype:
            return object
        else:
            out = empty(object.shape, dtype)
            out[:] = object
            return out
    if dtype is None:
        try:
            float(object[0])
            dtype = "float64"
        except (TypeError, IndexError):
            dtype = "float64"
    shape = []
    obj = object
    while isinstance(obj, (tuple, list)):
        shape.append(len(obj))
        if len(obj) == 0:
            break
        obj = obj[0]
    out = empty(tuple(shape), dtype)
    out[:] = object
    return out


def zeros(shape, dtype="float64"):
    return ndarray(shape, dtype)


def empty(shape, dtype="float64"):
    return ndarray(shape, dtype)
