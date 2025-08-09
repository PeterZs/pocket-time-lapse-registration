import numpy as np
import pickle
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import numbers


class NDArray(object):
    # def __init__(self, *args, **kwargs):
    #     self._ndarray = np.ndarray(*args, **kwargs);
    def __init__(self, data=None, shape=None, copy_data=True, **kwargs):
        super(NDArray, self).__init__(**kwargs)
        if(data is None and shape is None):
            assert(False), "Must provide data or shape ti NDArray constructor";

        if(isinstance(data, np.ndarray)):
            ndata = data;
            # ndata = data.copy();
        elif(isinstance(data, (list, tuple))):
            ndata = np.array(data);
        elif(isinstance(data, NDArray)):
            ndata = data._ndarray;
        else:
            assert(data is None), "Did not recognize data={} for NDArray".format(data);
            ndata = None;

        if(shape is not None and ndata is not None):
            if(not isinstance(ndata, np.ndarray)):
                ndata = np.array(ndata);
            assert(shape == ndata.shape), "shape {} and data with shape {} are inconsistant".format(shape, ndata.shape);
        if(ndata is not None):
            if(not isinstance(ndata, np.ndarray)):
                ndata = np.array(ndata);
            if(copy_data):
                self._ndarray=ndata.copy();
            else:
                self._ndarray = ndata;
        else:
            self._ndarray = np.zeros(shape);


    # <editor-fold desc="Property: 'as_numpy_array'">
    @property
    def as_numpy_array(self):
        return self._get_as_numpy_array();
    def _get_as_numpy_array(self):
        return self._ndarray;
    # </editor-fold>

    @classmethod
    def from_numpy_array(cls, numpy_array):
        return cls(numpy_array);



    @classmethod
    def _make_new(cls, *args, **kwargs):
        return cls(*args, **kwargs);

    def _r_operator_return_class(self, *args, **kwargs):
        """
        Constructor for class to return from right side operators
        :param args:
        :param kwargs:
        :return:
        """
        return self._make_new(*args, **kwargs);


    # <editor-fold desc="Container methods for wrapping container ndarray">
    # def __len__(self):
    #     return self._ndarray.__len__();
    
    def __repr__(self):
        return self._ndarray.__repr__();

    def __str__(self):
        return self._ndarray.tolist().__str__();
    
    def __getitem__(self, key):
        return self._ndarray.__getitem__(key);
    
    def __setitem__(self, key, value):
        return self._ndarray.__setitem__(key, value);
    
    def __iter__(self):
        return self._ndarray.__iter__();
    
    def __contains__(self):
        return self._ndarray.__contains__();
    
    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #         return self._ndarray == other._ndarray;
    #     return self._ndarray == other


    def __iadd__(self, other):
        if (isinstance(other, NDArray)):
            self._ndarray.__iadd__(other._ndarray);
        else:
            self._ndarray.__iadd__(other);
        return self;

    def __isub__(self, other):
        if (isinstance(other, NDArray)):
            self._ndarray.__isub__(other._ndarray);
        else:
            self._ndarray.__isub__(other);
        return self;

    def __imul__(self, other):
        if (isinstance(other, NDArray)):
            self._ndarray.__imul__(other._ndarray);
        else:
            self._ndarray.__imul__(other);
        return self;

    def __idiv__(self, other):
        if (isinstance(other, NDArray)):
            self._ndarray.__idiv__(other._ndarray);
        else:
            self._ndarray.__idiv__(other);
        return self;


    def __add__(self, other):
        if(isinstance(other, NDArray)):
            return self._r_operator_return_class(np.add(self._ndarray, other._ndarray));
        else:
            return self._r_operator_return_class(np.add(self._ndarray, other));

    def __radd__(self, other):
        return self.__add__(other);

    def __sub__(self, other):
        if(isinstance(other, NDArray)):
            return self._r_operator_return_class(np.subtract(self._ndarray, other._ndarray));
        else:
            return self._r_operator_return_class(np.subtract(self._ndarray, other));

    def __rsub__(self, other):
        if (isinstance(other, NDArray)):
            return self._r_operator_return_class(np.subtract(other._ndarray, self._ndarray));
        else:
            return self._r_operator_return_class(np.subtract(other, self._ndarray));

    def __mul__(self, other):
        if(isinstance(other, NDArray)):
            return self._r_operator_return_class(np.multiply(self._ndarray, other._ndarray));
        else:
            return self._r_operator_return_class(np.multiply(self._ndarray, other));

    def __rmul__(self, other):
        return self.__mul__(other);

    def __div__(self, other):
        if(isinstance(other, NDArray)):
            return self._r_operator_return_class(np.divide(self._ndarray, other._ndarray.astype(float)));
        else:
            return self._r_operator_return_class(np.divide(self._ndarray, other.astype(float)));


    def __truediv__(self, other):
        return self.__div__(other);

    def __floordiv__(self, other):
        assert(False), "Don't use __floordiv__ on {} class... too much confusion with regard to Python version compatibility...".format(self.__class__);
        # if(isinstance(other, NDArray)):
        #     return self._RClass(np.divide(self._ndarray, other._ndarray.astype(float)));
        # else:
        #     return self._RClass(np.divide(self._ndarray, other.astype(float)));

    def __rdiv__(self, other):
        if (isinstance(other, NDArray)):
            return self._r_operator_return_class(np.divide(other._ndarray.astype(float), self._ndarray));
        else:
            return self._r_operator_return_class(np.divide(other.astype(float), self._ndarray));


    # def __eq__(self, other):
    #     if (isinstance(other, NDArray)):
    #         req = other._ndarray.__eq__(self._ndarray);
    #     else:
    #         req =  (other == (self._ndarray));
    #     # print("{} == {} :\n{} to {}\n".format(self, other, req, req.all()));
    #     return req;

    def __eq__(self, other):
        if (isinstance(other, NDArray)):
            req = other._ndarray.__eq__(self._ndarray);
        else:
            req =  (other == (self._ndarray));
        # print("{} == {} :\n{} to {}\n".format(self, other, req, req.all()));
        return req.all();

    def elementwise_equal(self, other):
        if (isinstance(other, NDArray)):
            req = other._ndarray.__eq__(self._ndarray);
        else:
            req =  (other == (self._ndarray));
        # print("{} == {} :\n{} to {}\n".format(self, other, req, req.all()));
        return req;


    # def __eq__(self, other):
    #     if (isinstance(other, NDArray)):
    #         req = other._ndarray.__eq__(self._ndarray);
    #     else:
    #         req =  (other == (self._ndarray));
    #     # print("{} == {} :\n{} to {}\n".format(self, other, req, req.all()));
    #     return req.all();



    # def __add__(self, other):
    #     if(isinstance(other, self.__class__)):
    #         return self._ndarray.__add__(other._ndarray);
    #     else:
    #         return self._ndarray.__add__(other);
    # def __sub__(self, other):
    #     if(isinstance(other, self.__class__)):
    #         return self._ndarray.__sub__(other._ndarray);
    #     else:
    #         return self._ndarray.__sub__(other);
    # </editor-fold>

    @property
    def T(self):
        return self._ndarray.T;

    @property
    def data(self):
        return self._ndarray.data;

    @property
    def flags(self):
        return self._ndarray.flags;

    @property
    def dtype(self):
        return self._ndarray.dtype;

    @property
    def flat(self):
        return self._ndarray.flat;
    
    @property
    def imag(self):
        return self._ndarray.imag;
    
    @property
    def real(self):
        return self._ndarray.real;
    
    @property
    def size(self):
        return self._ndarray.size;

    @property
    def itemsize(self):
        return self._ndarray.itemsize;

    @property
    def nbytes(self):
        return self._ndarray.nbytes;

    @property
    def ndim(self):
        return self._ndarray.ndim;

    @property
    def shape(self):
        return np.shape(self._ndarray);

    @property
    def strides(self):
        return self._ndarray.strides;

    @property
    def base(self):
        return self._ndarray.base;

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return self._r_operator_return_class(self._ndarray.astype(dtype, order=order, casting = casting, subok=subok, copy=copy));

    def tolist(self):
        return self._ndarray.tolist();


    def homogenize(self):
        if(np.not_equal(self._ndarray[-1], 0)):
            self._ndarray = self._ndarray/self._ndarray[-1];

    def norm(self, **kwargs):
        return np.linalg.norm(self._ndarray, **kwargs);

    def L2(self):
        return np.linalg.norm(self._ndarray);

    def normalize(self, **kwargs):
        norm = self.norm(**kwargs);
        self._ndarray = self._ndarray*np.true_divide(1.0, norm);

    def to_dictionary(self):
        """
        :return:
        """
        if (hasattr(super(NDArray, self), 'to_dictionary')):
            d = super(NDArray, self).to_dictionary();
        else:
            d = {};
        # d['ndarray'] = self._ndarray.dumps();
        # d['ndarray'] = jsonpickle.encode(self._ndarray);
        d['ndarray'] = self._ndarray;
        return d;

    def init_from_dictionary(self, d):
        if (hasattr(super(NDArray, self), 'init_from_dictionary')):
            super(NDArray, self).init_from_dictionary(d);
        # self._ndarray = pickle.loads(d['ndarray']);
        # self._ndarray = jsonpickle.decode(d['ndarray']);
        self._ndarray = d['ndarray'];

# class NDArrayObject():
#     def to_dictionary(self):
#         """
#         :return:
#         """
#         d = super(NDArray, self).to_dictionary();
#         d['ndarray'] = self._ndarray.dumps();
#         return d;
#
#     def init_from_dictionary(self, d):
#         super(NDArray, self).init_from_dictionary(d);
#         self._ndarray = pickle.loads(d['ndarray']);