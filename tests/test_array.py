"""Tests for the pure-Python ndarray class."""

from math import isclose


class TestArrayCreation:
    def test_array_from_list(self):
        from lib.array import array

        a = array([1, 2, 3])
        assert a.shape == (3,)
        assert a.dtype == "float64"
        assert a.size == 3
        assert a.ndim == 1

    def test_array_from_nested_list(self):
        from lib.array import array

        a = array([[1, 2], [3, 4]])
        assert a.shape == (2, 2)
        assert a.ndim == 2

    def test_array_with_dtype(self):
        from lib.array import array

        a = array([1, 2, 3], dtype="int32")
        assert a.dtype == "int32"

    def test_array_from_ndarray_no_copy(self):
        from lib.array import array, zeros

        a = zeros((3,))
        b = array(a, copy=False)
        assert b is a

    def test_array_from_ndarray_copy(self):
        from lib.array import array, zeros

        a = zeros((3,))
        b = array(a, copy=True)
        assert b is not a
        assert (b == a).all()

    def test_zeros(self):
        from lib.array import zeros

        a = zeros((2, 3))
        assert a.shape == (2, 3)
        assert a.dtype == "float64"
        assert a.sum() == 0.0

    def test_zeros_with_dtype(self):
        from lib.array import zeros

        a = zeros((2,), dtype="int32")
        assert a.dtype == "int32"

    def test_empty(self):
        from lib.array import empty

        a = empty((3, 4))
        assert a.shape == (3, 4)
        assert a.size == 12


class TestArrayProperties:
    def test_shape(self):
        from lib.array import array

        a = array([[1, 2, 3], [4, 5, 6]])
        assert a.shape == (2, 3)
        assert a.ndim == 2
        assert a.size == 6

    def test_itemsize_and_nbytes(self):
        from lib.array import array

        a = array([1, 2, 3], dtype="float64")
        assert a.itemsize == 8
        assert a.nbytes == 24

        b = array([1, 2, 3], dtype="int32")
        assert b.itemsize == 4
        assert b.nbytes == 12

    def test_dtype(self):
        from lib.array import array

        assert array([1], dtype="float64").dtype == "float64"
        assert array([1], dtype="int32").dtype == "int32"
        assert array([1], dtype="bool").dtype == "bool"

    def test_strides(self):
        from lib.array import array

        a = array([[1, 2, 3], [4, 5, 6]])
        assert a.strides == (24, 8)

    def test_T(self):
        from lib.array import array

        a = array([[1, 2], [3, 4], [5, 6]])
        t = a.T
        assert t.shape == (2, 3)
        assert t[0, 0] == 1
        assert t[1, 0] == 2

    def test_T_1d(self):
        from lib.array import array

        a = array([1, 2, 3])
        assert a.T is a

    def test_len(self):
        from lib.array import array

        assert len(array([1, 2, 3])) == 3
        assert len(array([[1, 2], [3, 4]])) == 2

    def test_base(self):
        from lib.array import zeros

        a = zeros((3,))
        v = a[0:2]
        assert v.base is a

    def test_flat(self):
        from lib.array import array

        a = array([[1, 2], [3, 4]])
        assert list(a.flat) == [1, 2, 3, 4]


class TestIndexing:
    def test_getitem_int(self):
        from lib.array import array

        a = array([10, 20, 30])
        assert a[0] == 10.0
        assert a[2] == 30.0

    def test_getitem_slice(self):
        from lib.array import array

        a = array([10, 20, 30, 40])
        v = a[1:3]
        assert v.shape == (2,)
        assert list(v.flat) == [20.0, 30.0]

    def test_getitem_2d_int(self):
        from lib.array import array

        a = array([[1, 2], [3, 4], [5, 6]])
        assert a[1, 0] == 3.0
        assert a[2, 1] == 6.0

    def test_getitem_2d_slice(self):
        from lib.array import array

        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        v = a[0:2, 1:3]
        assert v.shape == (2, 2)
        assert list(v.flat) == [2, 3, 5, 6]

    def test_getitem_step(self):
        from lib.array import array

        a = array([0, 1, 2, 3, 4, 5])
        v = a[::2]
        assert v.shape == (3,)
        assert list(v.flat) == [0, 2, 4]

    def test_setitem_scalar(self):
        from lib.array import array

        a = array([1, 2, 3])
        a[0] = 99
        assert a[0] == 99.0

    def test_setitem_slice(self):
        from lib.array import array

        a = array([1, 2, 3, 4])
        a[1:3] = [99, 88]
        assert list(a.flat) == [1, 99, 88, 4]

    def test_setitem_fill_slice(self):
        from lib.array import array

        a = array([1, 2, 3])
        a[:] = 0
        assert list(a.flat) == [0, 0, 0]

    def test_setitem_2d(self):
        from lib.array import array

        a = array([[1, 2], [3, 4]])
        a[0, :] = [9, 8]
        assert list(a.flat) == [9, 8, 3, 4]

    def test_index_error(self):
        from lib.array import array

        a = array([1, 2])
        try:
            a[5]
            assert False, "expected IndexError"
        except IndexError:
            pass


class TestArithmetic:
    def test_add_scalar(self):
        from lib.array import array

        a = array([1, 2, 3])
        b = a + 2
        assert list(b.flat) == [3, 4, 5]

    def test_add_arrays(self):
        from lib.array import array

        a = array([1, 2, 3])
        b = array([4, 5, 6])
        c = a + b
        assert list(c.flat) == [5, 7, 9]

    def test_radd(self):
        from lib.array import array

        a = array([1, 2, 3])
        b = 10 + a
        assert list(b.flat) == [11, 12, 13]

    def test_sub_scalar(self):
        from lib.array import array

        a = array([5, 6])
        b = a - 3
        assert list(b.flat) == [2, 3]

    def test_sub_arrays(self):
        from lib.array import array

        a = array([5, 6])
        b = array([1, 2])
        c = a - b
        assert list(c.flat) == [4, 4]

    def test_rsub(self):
        from lib.array import array

        a = array([1, 2])
        b = 10 - a
        assert list(b.flat) == [9, 8]

    def test_mul_scalar(self):
        from lib.array import array

        a = array([2, 3])
        b = a * 4
        assert list(b.flat) == [8, 12]

    def test_mul_arrays(self):
        from lib.array import array

        a = array([2, 3])
        b = array([4, 5])
        c = a * b
        assert list(c.flat) == [8, 15]

    def test_truediv_scalar(self):
        from lib.array import array

        a = array([6, 9])
        b = a / 3
        assert list(b.flat) == [2.0, 3.0]

    def test_truediv_arrays(self):
        from lib.array import array

        a = array([6, 9])
        b = array([3, 2])
        c = a / b
        assert list(c.flat) == [2.0, 4.5]

    def test_div_by_zero(self):
        from lib.array import array

        a = array([1])
        try:
            a / 0
            assert False, "expected ZeroDivisionError"
        except ZeroDivisionError:
            pass

    def test_floordiv(self):
        from lib.array import array

        a = array([7, 9])
        b = a // 3
        assert list(b.flat) == [2, 3]

    def test_mod(self):
        from lib.array import array

        a = array([7, 9])
        b = a % 3
        assert list(b.flat) == [1, 0]

    def test_pow(self):
        from lib.array import array

        a = array([2, 3])
        b = a**3
        assert list(b.flat) == [8, 27]

    def test_shape_mismatch(self):
        from lib.array import array

        a = array([1, 2])
        b = array([1, 2, 3])
        try:
            a + b
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestInPlace:
    def test_iadd(self):
        from lib.array import array

        a = array([1, 2])
        a += 5
        assert list(a.flat) == [6, 7]

    def test_isub(self):
        from lib.array import array

        a = array([5, 6])
        a -= 3
        assert list(a.flat) == [2, 3]

    def test_imul(self):
        from lib.array import array

        a = array([2, 3])
        a *= 4
        assert list(a.flat) == [8, 12]

    def test_itruediv(self):
        from lib.array import array

        a = array([6, 9])
        a /= 3
        assert list(a.flat) == [2.0, 3.0]

    def test_ifloordiv(self):
        from lib.array import array

        a = array([7, 9])
        a //= 3
        assert list(a.flat) == [2, 3]

    def test_imod(self):
        from lib.array import array

        a = array([7, 9])
        a %= 3
        assert list(a.flat) == [1, 0]

    def test_ipow(self):
        from lib.array import array

        a = array([2, 3])
        a **= 3
        assert list(a.flat) == [8, 27]


class TestMethods:
    def test_fill(self):
        from lib.array import array

        a = array([1, 2, 3])
        a.fill(7)
        assert list(a.flat) == [7, 7, 7]

    def test_copy(self):
        from lib.array import array

        a = array([1, 2])
        b = a.copy()
        assert b is not a
        assert (b == a).all()

    def test_flatten(self):
        from lib.array import array

        a = array([[1, 2], [3, 4]])
        f = a.flatten()
        assert f.shape == (4,)
        assert list(f.flat) == [1, 2, 3, 4]

    def test_ravel(self):
        from lib.array import array

        a = array([[1, 2], [3, 4]])
        r = a.ravel()
        assert r.shape == (4,)
        assert list(r.flat) == [1, 2, 3, 4]

    def test_reshape(self):
        from lib.array import array

        a = array([1, 2, 3, 4, 5, 6])
        r = a.reshape((2, 3))
        assert r.shape == (2, 3)
        assert r[0, 0] == 1
        assert r[1, 2] == 6

    def test_reshape_incompatible(self):
        from lib.array import array

        a = array([1, 2, 3])
        try:
            a.reshape((2, 2))
            assert False, "expected ValueError"
        except ValueError, AttributeError:
            pass

    def test_transpose_2d(self):
        from lib.array import array

        a = array([[1, 2], [3, 4]])
        t = a.transpose()
        assert t.shape == (2, 2)
        assert t[0, 1] == 3
        assert t[1, 0] == 2

    def test_astype(self):
        from lib.array import array

        a = array([1, 2], dtype="float64")
        b = a.astype("int32")
        assert b.dtype == "int32"
        assert list(b.flat) == [1, 2]

    def test_view_same_dtype(self):
        from lib.array import array

        a = array([1, 2, 3])
        v = a.view()
        assert v.shape == (3,)
        assert list(v.flat) == [1, 2, 3]

    def test_view_different_dtype_1d(self):
        from lib.array import array

        a = array([1, 2], dtype="int32")
        v = a.view("int16")
        assert v.dtype == "int16"
        assert v.shape == (4,)


class TestStatistics:
    def test_all(self):
        from lib.array import array

        assert array([1, 2, 3]).all()
        assert not array([0, 1]).all()

    def test_any(self):
        from lib.array import array

        assert array([0, 1]).any()
        assert not array([0, 0]).any()

    def test_min(self):
        from lib.array import array

        assert array([3, 1, 2]).min() == 1.0

    def test_max(self):
        from lib.array import array

        assert array([3, 1, 2]).max() == 3.0

    def test_sum(self):
        from lib.array import array

        assert array([1, 2, 3]).sum() == 6.0

    def test_prod(self):
        from lib.array import array

        assert array([2, 3, 4]).prod() == 24.0

    def test_mean(self):
        from lib.array import array

        assert array([1, 2, 3, 4]).mean() == 2.5

    def test_ptp(self):
        from lib.array import array

        assert array([3, 1, 4, 2]).ptp() == 3.0

    def test_argmax(self):
        from lib.array import array

        assert array([3, 1, 4, 2]).argmax() == 2

    def test_argmin(self):
        from lib.array import array

        assert array([3, 1, 4, 2]).argmin() == 1

    def test_var(self):
        from lib.array import array

        a = array([1, 2, 3, 4])
        assert isclose(a.var(), 1.25)

    def test_std(self):
        from lib.array import array

        a = array([1, 2, 3, 4])
        assert isclose(a.std(), 1.118033988749895)


class TestMisc:
    def test_repr_small(self):
        from lib.array import array

        a = array([1, 2, 3])
        r = repr(a)
        assert "array(" in r
        assert "1" in r
        assert "2" in r
        assert "3" in r

    def test_repr_large(self):
        from lib.array import array

        a = array(list(range(101)))
        r = repr(a)
        assert "<ndarray" in r
        assert "101" in r

    def test_repr_with_dtype(self):
        from lib.array import array

        a = array([1, 2], dtype="int16")
        r = repr(a)
        assert "dtype='int16'" in r

    def test_eq_same(self):
        from lib.array import array

        a = array([1, 2])
        b = array([1, 2])
        assert (a == b).all()

    def test_eq_different(self):
        from lib.array import array

        a = array([1, 2])
        b = array([3, 4])
        assert not (a == b).any()

    def test_eq_shape_mismatch(self):
        from lib.array import array

        a = array([1, 2])
        b = array([1, 2, 3])
        assert (a == b) is False

    def test_float_conversion(self):
        from lib.array import array

        assert float(array([3.14])) == 3.14

    def test_int_conversion(self):
        from lib.array import array

        assert int(array([42])) == 42

    def test_float_error(self):
        from lib.array import array

        try:
            float(array([1, 2]))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_int_error(self):
        from lib.array import array

        try:
            int(array([1, 2]))
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_invalid_dtype(self):
        from lib.array import ndarray

        try:
            ndarray((3,), dtype="foobar")
            assert False, "expected TypeError"
        except TypeError:
            pass
