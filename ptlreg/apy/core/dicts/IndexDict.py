

from functools import wraps
from ptlreg.apy.core.aobject.AObject import AObject, AObjectMeta
import ptlreg.apy.utils
from future.utils import iteritems
import six

def _add_index_dict_index(cls, _is_unique_id=False, index_name=None, skip_assert=False):
    # print("adding {} to class {}".format(index_name, cls))
    # assert (issubclass(cls, HasIndexDict)), "Cannot add index because {} does not inherit HasIndexDict".format(cls)
    assert(hasattr(cls, '_CLASS_INDEX_DICT_FUNC_MAP')), "Cannot add index {} because {} does not inherit HasIndexDict".format(index_name, cls);
    def decorator(func):
        ipname = index_name;
        if(ipname is None):
            ipname = func.__name__;
        if(not skip_assert):
            assert(cls._get_class_index_func_map().get(ipname) is None), "Class {} already has index property {}".format(cls, ipname);
        cls._get_class_index_func_map()[ipname]=[func, _is_unique_id];
        # @wraps(func)
        def get_u(self):
            # this should be the property get
            return self._index_dicts[ipname];
        # @wraps(func)
        def set_u(self, value):
            assert (False), "shouldnt be setting this way"
            self._index_dicts[ipname] = value;
        # @wraps(func)
        def del_u(self):
            assert (False), "shouldnt be deleting this way"
            del self._index_dicts[ipname];
        setattr(cls, ipname+'_map', property(get_u, set_u, del_u, "Mapped property {} is used as key in self.{}_map".format(ipname, ipname)))
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally
    return decorator

class HasIndexDictMeta(type):
    def __new__(cls, *args, **kwargs):
        newtype = super(HasIndexDictMeta, cls).__new__(cls, *args, **kwargs);
        # newtype = super(HasIndexDictMeta, cls).__new__(cls);
        assert(newtype._CLASS_INDEX_DICT_FUNC_MAP.get(newtype.__name__) is None), "Already have HasIndexDict class called {}".format(newtype._CLASS_INDEX_DICT_FUNC_MAP.get(newtype.__name__));
        newtype._CLASS_INDEX_DICT_FUNC_MAP[newtype.__name__]={};
        for attr in dir(newtype):
            obj = getattr(newtype, attr);
            if (callable(obj) and hasattr(obj, "index_prop_name")):
                _add_index_dict_index(newtype, _is_unique_id=obj._is_unique_id, index_name=obj.index_prop_name, skip_assert=True)(obj);
        return newtype;

    def __init__(cls, *args, **kwargs):
        super(HasIndexDictMeta, cls).__init__(*args, **kwargs);
        return;
        # return super(HasIndexDictMeta, cls).__init__(*args, **kwargs);

    def __call__(cls, *args, **kwargs):
        supercall = super(HasIndexDictMeta, cls).__call__(*args, **kwargs);
        return supercall;

class IndexPropFunc(object):
    """
    Decorator class. use @IndexPropFunc(prop_name=None, _is_unique_id=None) decorator to register a function with a feature name.
    """
    def __init__(self, prop_name=None, _is_unique_id=False, **decorator_args):
        self.index_prop_name =prop_name;
        self._is_unique_id = _is_unique_id;
    def __call__(self, func):
        if(self.index_prop_name is None):
            self.index_prop_name = func.__name__;
        decorated = func;
        decorated.index_prop_name = self.index_prop_name;
        decorated.__name__ = '_'+func.__name__+'_index_func';
        decorated._is_unique_id = self._is_unique_id;
        return decorated;


@six.add_metaclass(HasIndexDictMeta)
class HasIndexDict(object):
    """
    This class initially written to store maps from names, md5s, etc, to medianodes
    This is basically used to immitate uniqueIDs in a database
    """
    # __metaclass__ = HasIndexDictMeta;
    _CLASS_INDEX_DICT_FUNC_MAP = {};
    @classmethod
    def index_dict_func_map(cls):
        return cls._CLASS_INDEX_DICT_FUNC_MAP[cls.__name__];

    # def __new__(cls, *args, **kwargs):
    #     newo = super(HasIndexDict, cls).__new__(cls);
    #     # newo = super(HasIndexDict, cls).__new__(cls, *args, **kwargs);
    #     # newo = super(HasIndexDict, cls).__new__(*args, **kwargs);
    #     # assert(isinstance(newo, AObject)), "All HasIndexDict must be mixed in with an AObject"
    #     return newo;



    def __init__(self, *args, **kwargs):
        # Maps names of properties to maps from those properties to the object of interest
        self._reset_index_dicts();
        self._set_default_instance_index_map_funcs()
        super(HasIndexDict, self).__init__(*args, **kwargs)
        
    def _set_default_instance_index_map_funcs(self):
        return;

    def _reset_index_dicts(self, **kwargs):
        self._index_dicts = None;
        self._instance_func_map = {};
        self._init_index_dicts(**kwargs);
        # if (hasattr(super(HasDGraph, self), '_resetBlank')):
        #     super(HasIndexDict, self)._resetBlank(**kwargs);


    def _add_index_map_func(self, func, index_name=None, _is_unique_id=False):
        """

        :param func:
        :param index_name:
        :param _is_unique_id:
        :param skip_assert:
        :return:
        """
        allexisting = self._get_all_indexed_objects();
        ipname = index_name;
        if (ipname is None):
            ipname = func.__name__;
        assert(self._get_index_func_map().get(ipname) is None), "Tried to add instance prop {} to {} when it is already has one with that name".format(ipname, self);
        self._instance_func_map[ipname] = [func, _is_unique_id];
        self._init_index_dict(ipname);
        for o in allexisting:
            self._add_object_to_index(object=o, index=ipname);

    def _apply_to_indexed_objects(self, func, **kwargs):
        allexisting = self._get_all_indexed_objects();
        for o in allexisting:
            func(o, **kwargs);

    def _add_index_map_func_to_indexed_objects(self, func, index_name=None, _is_unique_id=False):
        def addfunc(o):
            o._add_index_map_func(func, index_name=index_name, _is_unique_id=_is_unique_id)
        self._apply_to_indexed_objects(addfunc);

    def _remove_index_dict(self, index_name):
        assert(index_name in self._index_dicts.keys()),"Tried to remove index_dict {} that was not in _index_dicts with keys {}".format(index_name, self._index_dicts.keys())
        del self._index_dicts[index_name];


    def _remove_index_map(self, index_name):
        assert(index_name in self._instance_func_map.keys()),"Tried to remove index {} that was not in _instance_func_map with keys {}".format(index_name, self._instance_func_map.keys())
        del self._instance_func_map[index_name];
        del self._index_dicts[index_name];

    def _recompute_index_map(self, index_name):
        assert(index_name in self._get_index_func_map().keys()), "Tried to recompute index {} that was not in func_map with keys {}".format(index_name, self._get_index_func_map().keys())
        allexisting = self._get_all_indexed_objects();
        self._remove_index_dict(index_name);
        self._init_index_dict(index_name);
        for o in allexisting:
            self._add_object_to_index(object=o, index=index_name);

    def _get_index_map(self, by_index):
        return self._index_dicts.get(by_index);

    def get_index_map(self, index_name):
        return self._get_index_map(index_name);

    def _get_instance_index_func_map(self):
        return self._instance_func_map;
    # @IndexPropFunc(_is_unique_id=False)
    # def firstel(self, o):
    #     return o[0];
    #
    # @IndexPropFunc(_is_unique_id=True)
    # def lower(self, o):
    #     return o.lower();

    def _get_index_func_map(self):
        rmap = dict(self._get_instance_index_func_map());
        rmap.update(self._get_class_index_func_map());
        return rmap;

    @classmethod
    def _get_class_index_func_map(cls):
        return cls._CLASS_INDEX_DICT_FUNC_MAP[cls.__name__];

    def _get_index_function(self, index):
        return self._get_index_func_map()[index][0];
    def _eval_index_function(self, index, object):
        return self._get_index_function(index)(self, object);
    def _is_unique_id(self, index):
        return self._get_index_func_map()[index][1];

    # <editor-fold desc="Property: 'index_dicts'">
    @property
    def _index_dicts(self):
        return self._index_dictionaries;
    @_index_dicts.setter
    def _index_dicts(self, value):
        self._index_dictionaries=value;

    def _init_index_dicts(self, **kwargs):
        if(self._index_dicts is None):
            self._index_dicts = {};
            for p in self._get_index_func_map().keys():
                self._init_index_dict(index=p);
    def _init_index_dict(self, index, **kwargs):
        self._index_dicts[index] = {};
    # </editor-fold>

    def _get_all_indexed_objects(self):
        if(len(self._get_index_func_map().keys())==0):
            return [];
        index = list(self._get_index_func_map().keys())[0];
        _is_unique_id = self._is_unique_id(index);
        if(_is_unique_id):
            return list(self._index_dicts[index].values());
        else:
            rlist = [];
            for l in self._index_dicts[index].keys():
                rlist = rlist+self._index_dicts[index][l];
            return rlist;

    @property
    def n_indexed_objects(self):
        return len(self._get_all_indexed_objects());

    def _assert_unique_id_not_taken(self, o):
        for k in self._get_index_func_map().keys():
            _is_unique_id = self._is_unique_id(k);
            if(_is_unique_id):
                ival = self._eval_index_function(index=k, object=o);
                # indexed = self._index_dicts[k].get(ival);
                assert(ival not in self._index_dicts[k]), "In _assert_unique_id_not_taken for {}\nwith index_dicts: {}\n{} with [{}: {}] already indexed in dict {}\nWas trying to index {}".format(self, self._index_dicts, o.__class__.__name__, k, ival, self._index_dicts[k], o);

    def _unique_id_is_taken(self, o):
        for k in self._get_index_func_map().keys():
            _is_unique_id = self._is_unique_id(k);
            if(_is_unique_id):
                ival = self._eval_index_function(index=k, object=o);
                # indexed = self._index_dicts[k].get(ival);
                if(ival in self._index_dicts[k]):
                    return True;
        return False;

    def _add_indexed_object(self, o):
        self._assert_unique_id_not_taken(o);
        for k in self._get_index_func_map().keys():
            self._add_object_to_index(o, index=k)
        return o;

    def _add_object_to_index(self, object, index):
        _is_unique_id = self._is_unique_id(index);
        ival = self._eval_index_function(index=index, object=object)
        if (_is_unique_id):
            self._index_dicts[index][ival] = object;
        else:
            if (self._index_dicts[index].get(ival) is None):
                self._index_dicts[index][ival] = [];
            elif (object in self._index_dicts[index][ival]):
                assert (False), "tried to add {} when already in indexdict list {}".format(object,self._index_dicts[index][ival]);
            self._index_dicts[index][ival].append(object);

    def _remove_indexed_object(self, o):
        for k in self._get_index_func_map().keys():
            self._remove_object_from_index(object=o, index=k);
        self._assert_unique_id_not_taken(o);
        return o;

    def _remove_object_from_index(self, object, index):
        _is_unique_id = self._is_unique_id(index);
        ival = self._eval_index_function(index=index, object=object);
        if (_is_unique_id):
            del self._index_dicts[index][ival];
        else:
            if (self._index_dicts[index].get(ival) is None or (object not in self._index_dicts[index][ival])):
                assert (False), "Tried to remove {} which is not indexed".format(object);
            self._index_dicts[index][ival].remove(object);

    # def to_dictionary(self):
    #     d = super(HasIndexDict, self).to_dictionary();
    #     assert(not d.get('_index_dictionaries')), "How does d already have _index_dictionaries? Is HasIndexDict.to_dictionary being called multiple times?"
    #     d['_index_dictionaries'] ={};
    #     for k in self._index_dicts.keys():
    #         d[k]={};
    #         for nkey, nval in iteritems(self.[k]):
    #             d[k][nkey]=self.getIndexedObjectReg(self._index_dicts[k][nkey]);
    #             assert (d[k][nkey] is not None), "IndexDict has none entry?"
    #     return d;
    #
    # def reconstructChildrenFromDicts(self, d):
    #     """
    #     this will not work unless childRef and getFromChildRef match
    #     :return: a map from childRef's to nodes
    #     """
    #     refmap = {};
    #     for k in d.keys():
    #         self._index_dicts[k]={};
    #         for nkey, nval in iteritems(d[k]):
    #             objref = d[k][nkey];
    #             if(refmap.get(objref) is None):
    #                 newob = self.NewNodeFromChildRef(childref);
    #                 refmap[childref]=lnode;
    #             else:
    #                 lnode = refmap[childref];
    #             self.child_maps[k][nkey]=lnode;

class IndexDictMeta(HasIndexDictMeta, AObjectMeta):
    pass;


class IndexDict(six.with_metaclass(IndexDictMeta, HasIndexDict, AObject)):
    pass;

# @six.add_metaclass(IndexDictMeta)
# class IndexDict(HasIndexDict, AObject):
#     pass;
    # __metaclass__ = IndexDictMeta;