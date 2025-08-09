from ptlreg.apy.aops import AOpType
from functools import wraps

class CaptureSessionSelectionOp(AOpType):
    """
    Decorator class to register a function with a op name.
    """
    def __init__(self, op_name, op_version = 0, **decorator_args):
        assert((op_name.lower() != 'all') and (op_name.lower() != 'each')), 'Cannot use feature name {}; it is reserved'.format(op_name);
        super(CaptureSessionSelectionOp, self).__init__(nickname=op_name,
                                                      op_version=op_version,
                                                      hashable_args=None,
                                                      **decorator_args);
        self.op_name = op_name;

    def decorate_function(self, func):
        # self._CheckForReservedArgs('force_recompute');
        @wraps(func)
        def decorated(*args, **kwargs):
            fkwargs = dict(kwargs);
            rval = func(*args, **fkwargs);
            return rval;
        return decorated;

    def set_op_info(self, decorated):
        super(CaptureSessionSelectionOp, self).set_op_info(decorated);
        decorated.op_name = self.op_name;
        return decorated;