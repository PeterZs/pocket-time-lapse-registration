from future.utils import iteritems
from .ParamDict import ParamDict
class FuncDict(ParamDict):
    """FuncDict (class): Extends ParamDict so that functions can be assigned to features and called whenever
        computing those features is necessary.
        Attributes:
            data: name -> name, value, params
            feature funcs: name -> function for evaluating
    """

    # def __init__(self, owner=None, name=None, path=None):
    #     ParamDict.__init__(self, owner=owner, name=name, path=path);
    #     self.functions = {};
    def __init__(self, owner=None, name=None):
        super(FuncDict, self).__init__(owner=owner, name=name);
        self.functions = {};

    def __str__(self):
        rstr = '';
        for k in self.data:
            f = self.functions.get(k);
            if(f is None):
                rstr = rstr + "--{}--\nparams: {}\nmodified: {}\n\n".format(k, self.data[k]['params'], self.data[k]['modified'])
            else:
                rstr = rstr + "--{}--\nparams: {}\nmodified: {}\n".format(k, self.data[k]['params'], self.data[k]['modified'])
                for fk, fv in iteritems(f.op_info):
                    rstr = rstr+"{}: {}\n".format(fk, fv);
                rstr = rstr +'\n'
        return rstr;


    def _get_entry(self, name=None, force_recompute=False, **kwargs):
        d = self.data.get(name);
        if((d is not None) and (not force_recompute)):
            return d;
        else:
            f = self.get_function(name=name);
            if(f is not None):
                value = f(self.owner, force_recompute=force_recompute, **kwargs);
                self.set_value(name=name, value=value, **kwargs);
        return self.data.get(name);

    def get_function(self, name=None):
        return self.functions.get(name);

    def set_function(self, name, function=None):
        self.functions[name]=function;

    def get_function_list(self):
        return list(self.functions.keys());
