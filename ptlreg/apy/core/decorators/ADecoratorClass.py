from functools import wraps
import six
import inspect
try:
    getfullargspec = inspect.getfullargspec;
except AttributeError:
    getfullargspec = inspect.getargspec;







class DecoratorClassMeta(type):
    def __new__(cls, *args, **kwargs):
        newoptype = super(DecoratorClassMeta, cls).__new__(cls, *args, **kwargs);
        return newoptype;
    def __init__(cls, *args, **kwargs):
        super(DecoratorClassMeta, cls).__init__(*args, **kwargs);
        # return super(AOpTypeMeta, cls).__init__(*args, **kwargs);
    def __call__(cls, *args, **kwargs):
        supercall = super(DecoratorClassMeta, cls).__call__(*args, **kwargs);
        return supercall;



class DecoratorClassMixin(object):
    def __init__(self, **decorator_args):
        '''
        Deal with decorator arguments
        :param decorator_args:
        '''
        self.decorator_args = decorator_args;

    def __call__(self, func, *args, **kwargs):
        '''
        Called when function is declared, to decorate function
        :param func:
        :return:
        '''
        # print(func)
        # print(args)
        # print(kwargs)
        self.original_func_name = func.__name__;
        print("call on {}".format(self.original_func_name))
        decorated = self.decorate_function(func);
        return decorated;

    def decorate_function(self, func):
        '''
        Actual decorator. This is probably what you want to customize.
        :param func:
        :return:
        '''
        @wraps(func)
        def decorated(*args, **kwargs):
            rval = func(*args, **kwargs);
            return rval;
        return decorated;

@six.add_metaclass(DecoratorClassMeta)
class ADecoratorClass(DecoratorClassMixin):
    pass;
    # __metaclass__ = AOpTypeMeta;