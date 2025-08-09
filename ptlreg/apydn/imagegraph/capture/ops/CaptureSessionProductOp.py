import os
from functools import wraps
from ptlreg.apy.aops import AOpType


class CaptureSessionProductOp(AOpType):
    """
    Decorator class to register a function with a op name.
    """
    def __init__(self, op_name, op_version = 0, **decorator_args):
        assert((op_name.lower() != 'all') and (op_name.lower() != 'each')), 'Cannot use feature name {}; it is reserved'.format(op_name);
        super(CaptureSessionProductOp, self).__init__(nickname=op_name,
                                                        op_version=op_version,
                                                        hashable_args=None,
                                                        **decorator_args);
        self.op_name = op_name;

    def decorate_function(self, func):
        self._check_for_reserved_args('force_recompute');
        op_name = self.op_name;
        subdir_name = "{}_session_product".format(op_name);
        @wraps(func)
        def decorated(*args, **kwargs):
            obj = args[0];
            fkwargs = dict(kwargs);
            session_product_subdir_suffix = fkwargs.pop('session_product_subdir_suffix', None);
            session_subdir_name = subdir_name
            if(session_product_subdir_suffix is not None):
                session_subdir_name = subdir_name+"_{}".format(session_product_subdir_suffix)
            force_recompute = fkwargs.pop('force_recompute', None);
            save_result = fkwargs.pop('save_result', True);
            ext = fkwargs.pop('ext', '.png');
            rval = None;
            if(obj._capture_target is not None and save_result):
                ct = obj._capture_target;
                ct.add_images_subdir_if_missing(session_subdir_name);
                subdir_path = ct.get_images_subdir(session_subdir_name);
                product_sample_name = obj.primary_sample.file_name_base;
                product_sample_path = os.path.join(subdir_path, product_sample_name+ext);
                if(force_recompute or not os.path.exists(product_sample_path)):
                    im = func(*args, **fkwargs);
                    im.write_to_file(product_sample_path);
                new_sample = ct.add_sample_for_image_path(product_sample_path);
                ct._add_session_product_labels(new_sample, op_name);
                rval = new_sample.get_image();
            else:
                rval = func(*args, **fkwargs);
            return rval;
        return decorated;

    def set_op_info(self, decorated):
        super(CaptureSessionProductOp, self).set_op_info(decorated);
        decorated.op_name = self.op_name;
        return decorated;