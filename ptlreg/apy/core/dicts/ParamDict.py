
# import cPickle as pickle
# import _pickle as cPickle
try:
    import cPickle as pickle
except:
    import pickle

import os
import fnmatch
import copy
from future.utils import iteritems

class ParamDict(object):
    """ParamDict (class): Dictionary that stores values and the parameters used to compute those values. I use this
        class for two things. The first is to store parameters for reproducability. The second is to only recompute
        values when functions are called with different parameters (some of this latter functionality is tied up in
        code that isn't part of the Lite release).

        Attributes:
            data: name -> value, params
    """

    def __init__(self, owner=None, name=None, path=None, **kwargs):
        self.name=name;
        self.data = {};
        self.owner=owner;
        self.modified=False;
        super(ParamDict, self).__init__(**kwargs)

    def __str__(self):
        rstr = '';
        for k in self.data:
            rstr = rstr+"--{}--\nparams: {}\nmodified: {}\n\n".format(k, self.data[k]['params'], self.data[k]['modified'])
        return rstr;

    def _get_entry(self, name=None, force_recompute=False, **kwargs):
        d = self.data.get(name);
        if((d is not None) and (not force_recompute)):
            return d;
        return None;

    def _set_entry(self, name, d):
        assert(name!='all' and name!='each'),"Entry named '{}' is reserved in ParamDict".format(name);
        self.data[name]=d;

    def has_entry(self, name=None):
        return (self.data.get(name) is not None);

    def get_value(self, name=None, force_recompute=False, **kwargs):
        d = self._get_entry(name=name, force_recompute=force_recompute, **kwargs);
        if(d is not None):
            return d.get('value');
        else:
            return None;

    def get_params(self, name=None):
        d = self.data.get(name);
        if(d is not None):
            return d.get('params');
        else:
            return None;

    def set_value(self, name, value=None, modified=True, **kwargs):
        if(self.data.get(name) is None):
            self.data[name] = {};
        self.data[name]['value']=value;
        self.data[name]['params']=kwargs;
        self.data[name]['name']=name;
        self.set_entry_modified(name=name, is_modified=modified)

    def set_entry_info(self, entry_name, **kwargs):#info_label, info_value, modified=True):
        if (self.data.get(entry_name) is None):
            self.data[entry_name] = {};
        if(kwargs == {}):
            return;
        self.set_entry_modified(name=entry_name, is_modified=True);
        for k, v in iteritems(kwargs):
            self.data[entry_name][k] = v;


    def get_entry_info(self, entry_name, info_label=None):
        entry = self.data[entry_name];
        if(entry is None):
            AWARN("requested info_label {} for non-existent entry {} in ParamDict".format(info_label, entry_name))
            return None;
        if(info_label is None):
            return self.data[entry_name];
        return self.data[entry_name].get(info_label);

    def save_entry(self, name, path, force=False):
        """Save one entry to one file."""
        if(self.has_entry(name=name) is None):
            return None;
        if(self.is_entry_modified(name=name) or force or (not os.path.isfile(path))):
            f = open(path, 'wb');
            pickle.dump(self._get_entry(name=name), f, protocol=2);
            f.close();
            self.set_entry_modified(name=name, is_modified=False);
        return True;

    def set_entry_modified(self, name, is_modified=True):
        self.data[name]['modified']=is_modified;
        if(is_modified):
            self.set_modified(is_modified=True);

    def is_entry_modified(self, name):
        entry = self.data.get(name);
        if(entry is not None):
            m=entry.get('modified');
            if(m is not None):
                return m;
            else:
                return True;
        else:
            assert(False), "checking mod bit on entry that does not exist"

    def is_modified(self):
        return self.modified;

    def set_modified(self, is_modified):
        self.modified=is_modified;

    def save(self, path, force=False):
        """save all entries to one file."""
        if(force or self.is_modified()):
            f = open(path, 'wb');
            pickle.dump(self.data, f, protocol=2);
            f.close();
            self.set_modified(is_modified=False);
        return True;

    def load_entry(self, path, assign_name=None):
        """load one entry from one file."""
        f=open(path, 'rb');
        d = pickle.load(f);
        if(assign_name is not None):
            d['name'] = assign_name;
        self._set_entry(name=d['name'], d=d);
        f.close();
        self.set_entry_modified(name=d['name'], is_modified=False);
        return True;

    def load_entries_from_dir(self, dir_path):
        """
        Do not load the all.pkl because that will be at odds with individual
        :param dir_path:
        :return:
        """
        assert (os.path.isdir(dir_path)), "{} is not a directory.".format(dir_path);
        nloaded = 0;
        for filename in os.listdir(dir_path):
            if(fnmatch.fnmatch(filename, '*.pkl')and filename.lower() != 'all.pkl'):
                path = os.path.join(dir_path, filename);
                self.load_entry(path=path);
                nloaded = nloaded+1;

    def deep_clone(self):
        return copy.deepcopy(self);

    def _update_with_param_dict(self, d):
        self.data.update(d.data);
        self.modified = True;

    def load(self, path):
        """Load a set of entries all from one file."""
        f=open(path, 'rb');
        newd = pickle.load(f);
        self.data.update(newd);
        f.close();
        return True;

    def get_key_list(self):
        return list(self.data.keys());