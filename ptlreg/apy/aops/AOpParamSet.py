import ptlreg.apy.utils
from ptlreg.apy.core.aobject.AObject import AObject
from ptlreg.apy.core.dicts.IndexDict import IndexDict, IndexPropFunc

class AOpParamSet(AObject):
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(AOpParamSet, self).__init__();
        self.kwargs = kwargs;

    @property
    def _string_to_hash(self):
        return self.encode_param_key(self.kwargs);

    @staticmethod
    def encode_param_key(kwargs):
        return ptlreg.apy.utils.jsonpickle.encode(kwargs);

    @staticmethod
    def decode_param_key(kwargs):
        return ptlreg.apy.utils.jsonpickle.decode(kwargs);

    # <editor-fold desc="Property: 'hash_value'">
    @property
    def _hash_value(self):
        return self.get_info("hash_value");
    @_hash_value.setter
    def _hash_value(self, value):
        self.set_info('hash_value', value);
    # </editor-fold>


    def get_hash(self):
        if(self._hash_value is not None):
            return self._hash_value;
        return hash(self._string_to_hash);

    def __hash__(self):
        return self.get_hash();

    def __eq__(self,other):
        if (other is None):
            return False;
        if(len(self.kwargs.keys()) != len(other.kwargs.keys())):
            return False;
        for k in self.kwargs.keys():
            if(self.encode_param_key(self.kwargs[k])!=self.encode_param_key(other.kwargs[k])):
                return False;
        return True;

    def __ne__(self,other):
        return (not self.__eq__(other));

    # <editor-fold desc="Property: 'kwargs'">
    @property
    def kwargs(self):
        return self.get_info("kwargs");
    @kwargs.setter
    def kwargs(self, value):
        self.set_info('kwargs', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'result_info'">
    @property
    def result_info(self):
        return self.get_info("result_info");
    @result_info.setter
    def result_info(self, value):
        self.set_info('result_info', value);
    # </editor-fold>


class AOpParamSetMap(IndexDict):
    def __init__(self, **kwargs):
        super(AOpParamSetMap, self).__init__(**kwargs);

    def __getitem__(self, key):
        return self._aoparamset_hash_map.__getitem__(key);

    def __setitem__(self, key, value):
        return self._aoparamset_hash_map.__setitem__(key, value);

    def get(self, **kwargs):
        ps = AOpParamSet(**kwargs);
        return self._aoparamset_hash_map.get(ps);

    def create_param_set(self, **kwargs):
        ps = AOpParamSet(**kwargs);
        self._add_indexed_object(ps);
        return ps;

    @IndexPropFunc(_is_unique_id=True)
    def _aoparamset_hash(self, aoparamset):
        return aoparamset.get_hash();

    def save_op_param_set_map(self, file_path):
        f = open(file_path, 'wb');
        ptlreg.apy.utils.pickle.dump(self, f, protocol=2);
        f.close();
        return True;

    @staticmethod
    def load_op_param_set_map(file_path):
        f=open(file_path, 'rb');
        newd = ptlreg.apy.utils.pickle.load(f);
        f.close();
        return newd;
