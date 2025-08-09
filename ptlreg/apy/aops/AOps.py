from ptlreg.apy.core.DictDir import DictDir
from functools import wraps
import ptlreg.apy.utils
from ptlreg.apy.aops.AOpParamSet import *
import six

import inspect
try:
    getfullargspec = inspect.getfullargspec;
except AttributeError:
    getfullargspec = inspect.getargspec;





class AOps(object):
    REGISTERED_OP_TYPES_DICT = DictDir();
    REGISTERED_OP_TYPES_LIST = [];
    REGISTERED_OPS_LIST = [];
    @classmethod
    def AddOpType(cls, op_type):
        op_details = op_type.aop_type_details();
        assert(not cls.REGISTERED_OP_TYPES_DICT.get_item(op_details)), "AOpTypeDetails {} already used by {}\nold class: {}\nnew class: {}!".format(op_details, op_type.__name__, cls.REGISTERED_OP_TYPES_DICT.get_item(op_details), op_type);
        # op_type._RegisteredOps = DictDir(base_dictionary=)
        cls.REGISTERED_OP_TYPES_DICT.add_item(item_detail_list=op_details, item_value=op_type)
        cls.REGISTERED_OP_TYPES_LIST.append(op_type);

    @classmethod
    def GetOpType(cls, op_type_name):
        return cls.REGISTERED_OP_TYPES_DICT(op_type_name);

    @classmethod
    def GetOp(cls, op_type_name, op_name):
        op_type = cls.REGISTERED_OP_TYPES_DICT(op_type_name);
        return op_type.get_op_by_name(op_name);

    # @classmethod
    # def GetOpType(cls, op_class):
    #     op_details = op_class.AOpTypeDetails();
    #     return cls.REGISTERED_OP_TYPES_DICT(op_details);


# def register_op_type(op_type_to_register):
#     AOps.AddOpType(op_type_to_register);

def _getArgDefaultDictionary(func):
    aspec = getfullargspec(func);
    return dict(zip(reversed(aspec.args), reversed(aspec.defaults)))

class AOpTypeMeta(type):
    @classmethod
    def GetOpManager(cls):
        return AOps;

    def __new__(cls, *args, **kwargs):
        newoptype = super(AOpTypeMeta, cls).__new__(cls, *args, **kwargs);
        # register_graph_op_class(newoptype);
        cls.GetOpManager().AddOpType(newoptype);
        newoptype._RegisteredOps = DictDir();
        return newoptype;

    def __init__(cls, *args, **kwargs):
        super(AOpTypeMeta, cls).__init__(*args, **kwargs);
        # return super(AOpTypeMeta, cls).__init__(*args, **kwargs);


    def __call__(cls, *args, **kwargs):
        supercall = super(AOpTypeMeta, cls).__call__(*args, **kwargs);
        return supercall;



class AOpTypeMixin(object):
    @classmethod
    def _GetMetaOpManager(cls):
        return type(cls).GetOpManager();
            # return cls.__metaclass__.GetOpManager();



    def __init__(self, nickname=None, op_version=0, hashable_args=None, **other_decorator_args):
        self.op_type = self.op_type_name();
        self.nickname = nickname;
        self.op_version = op_version;
        self.hashable_args = hashable_args;
        self.other_decorator_args = other_decorator_args;

    def __call__(self, func):
        self.ValidateFuncArgs(func);
        self.PreDecorate(func);
        self.original_func_name = func.__name__;
        self.op_name = self.original_func_name;
        if (self.nickname is not None):
            self.op_name = self.nickname;
        self.op_index = self.get_op_index();
        decorated = self.decorate_function(func);
        self.set_op_info(decorated);
        self.register_op(decorated);
        self.PostDecorate(decorated)
        return decorated;

    def _getHashableArgsSubset(self, **kwargs):
        if(self.hashable_args is None):
            return kwargs;
        else:
            hashable_args = self.hashable_args;
            if(not isinstance(self.hashable_args, (tuple, list))):
                hashable_args = [hashable_args];
            return ptlreg.apy.utils.subdict(subkeys=hashable_args, full_dict=kwargs);

    def _GetParamSetForHashableArgs(self, **kwargs):
        hashable_argset = self._getHashableArgsSubset(**kwargs);
        return AOpParamSet(**hashable_argset);

    def ValidateFuncArgs(self, func):
        """
        You can check the arguments of func here
        E.g.:
        self._CheckForRequiredArgs(func, output_path=None);
        self._CheckForReservedArgs(func, 'force_recompute');
        :param func:
        :return:
        """
        return;


    def PreDecorate(self, func):
        """
        this is where you apply any prior decoration;
        :param func:
        :return:
        """
        # return super(AOpType, self).PreDecorate(func);
        return func;

    def PostDecorate(self, decorated):
        """
        this is where you apply any post decoration;
        :param func:
        :return:
        """
        # return super(AOpType, self).PostDecorate(func);
        return decorated;


    def decorate_function(self, func):
        @wraps(func)
        def decorated(*args, **kwargs):
            return func(*args, **kwargs);
        return decorated;

    def _CheckForRequiredArgs(self, func, required_args=None, **kwargs):
        """
        :param func: function to check
        :param required_args: list or tuple of argument names that should be required, but do not need specific default.
        :param kwargs: arguments that should have specific defaults can be provided as kwargs
        :return:
        """
        if(required_args is not None):
            if(not isinstance(required_args, (list, tuple))):
                required_args = [required_args];
            if(len(required_args)>0):
                for r in required_args:
                    assert (r in getfullargspec(func)[0]), "The argument {} must be included in {} decorated as {}.".format(r, func.__name__, self.op_type);

        # check for arguments that should have specific defaults
        if(len(kwargs.keys())>0):
            adfd = _getArgDefaultDictionary(func);
            for k in kwargs.keys():
                assert ((k in adfd) and (kwargs[k]==adfd[k])), "The argument and default {}={} must be included in {} decorated as {}.".format(k, kwargs[k], func.__name__, self.op_type);

    def _check_for_reserved_args(self, func, reserved_args=None):
        if(reserved_args is not None):
            for r in reserved_args:
                assert (r not in getfullargspec(func)[0]), "The argument {} in {} is reserved for functions decorated with {}. Use a different argument name...".format(r, func.__name__, self.op_type);

    def add_op_info_to(self, decorated, **kwargs):
        decorated.op_info.update(kwargs);

    def set_op_info(self, decorated):
        """
        This is where subclasses should set additional op info
        :param decorated:
        :return:
        """
        # if(hasattr(super(AOpType, self), 'SetOpInfo')):
        #     super(AOpType, self).SetOpInfo(decorated);
        decorated.op_info = {}
        decorated.op_index = self.op_index;
        self.add_op_info_to(decorated,
                            op_name=self.op_name,
                            op_type=self.op_type,
                            nickname=self.nickname,
                            op_version=self.op_version,
                            hashable_args=self.hashable_args,
                            op_index=self.op_index,
                            function_name = self.original_func_name,
                            other_decorator_args = self.other_decorator_args);

    def register_op(self, decorated):
        return self._add_registered_op(decorated);

    def _add_registered_op(self, decorated):
        previous_op = self.get_registered_ops().get_item(decorated.op_index);
        assert (previous_op is None), "Op named {} already registered as {}\nold op: {}\nnew op: {}!".format(decorated.op_index, self.op_type_name(), previous_op, decorated);
        self.get_registered_ops().add_item(item_detail_list=decorated.op_index, item_value=decorated);
        self._GetMetaOpManager().REGISTERED_OPS_LIST.append(decorated);

    def __str__(self):
        rstr = "AOpType {}:\n{}".format(self.op_type, self.get_registered_ops().__str__());
        return rstr;

    # def getOpDetailList(self):
    #     # return [self.op_type, self.nickname, self.op_version];
    #     return [self.op_type, self.nickname, self.op_version];

    def get_op_index(self):
        return self.op_name;

    @classmethod
    def op_type_name(cls):
        return cls.__name__;

    @classmethod
    def get_registered_ops(cls):
        # return DictDir(base_dictionary=AOps.REGISTERED_OP_TYPES_DICT.getItem(cls.OpTypeName()));
        return cls._RegisteredOps;


    @classmethod
    def get_op_by_name(cls, op_name):
        return cls.get_registered_ops().get_item(op_name);

    @classmethod
    def get_op_info(cls, op_name):
        return cls.get_registered_ops().get_item(op_name).op_info;


    @classmethod
    def aop_type_details(cls):
        return [cls.op_type_name()];

@six.add_metaclass(AOpTypeMeta)
class AOpType(AOpTypeMixin):
    pass;
    # __metaclass__ = AOpTypeMeta;