from ptlreg.apy.core.aobject import AObject, AOBJECT_CLASSES_DICT
from ptlreg.apy.core.SavesToJSON import SavesToJSON
import sys
import traceback
import os
import ptlreg.apy.utils

def TestAObjectLists(testdir):
    json_path = os.path.join(testdir, 'testaobjects.json')
    try:
        L = AObjectList();
        for cname in AOBJECT_CLASSES_DICT:
            cl = AOBJECT_CLASSES_DICT[cname];
            testi1 = cl();
            L.append(testi1);

        # json_path = os.path.join(testdir, 'testaobjects.json')
        L.write_to_json(json_path=json_path);
        L2 = AObjectList();
        L2.load_from_json(json_path=json_path);
    except:
        # print("TestAObjectLists Error for class {}".format(cl));
        e = sys.exc_info()[0];
        traceback.print_exc(file=sys.stdout)
        assert (False), "Failed trying to save AObjects in AObjectList.Class: {}\nError: {}".format(cl, e);

        assert (testi2.aobject_type_name() == cname), "Mismatch in AObject type name {} != {}".format(cname,
                                                                                                      testi2.aobject_type_name());
        assert (type(testi2) == type(testi1) and type(
            testi2) == cl), "Class mismatch in serializable for class {}.".format(cl);

class AObjectList(SavesToJSON):
    """
    Parent for classes representing lists of AObjects
    """
    ElementClass = AObject

    @classmethod
    def _validate_element_class(cls, elem):
        assert (isinstance(elem, cls.ElementClass)), "{} cannot be constructed with AObjectList containing {}".format(
            cls.__name__, elem.__class__);

    @classmethod
    def new_element(cls, *args):
        return cls.ElementClass(*args);




    def __init__(self, aobjects=None, zipped_args=False, **kwargs):
        """

        :param aobjects:
        :param zipped_args: if true then this means aobjects should be a list of arrays, each containing the ordered arguments for constructing an instance of self.element_class
        :param kwargs:
        """
        self._aobjects=[];
        super(AObjectList, self).__init__(**kwargs);
        self._init_aobject_list_subclass(aolist=self, aobjects=aobjects, zipped_args=zipped_args, **kwargs);

    @classmethod
    def _init_aobject_list_subclass(cls, aolist, aobjects, zipped_args=False):
        if(aobjects is not None):
            assert (isinstance(aobjects, (list, tuple, AObjectList))), "cannot initialize {} with type {}.".format(aolist.__class__, aobjects.__class__);
            if(zipped_args):
                # This is if the arguments were zipped
                for ao in aobjects:
                    aolist.append(cls.new_element(*ao));
            elif(isinstance(aobjects, AObjectList)):
                for ao in aobjects:
                    cls._validate_element_class(ao);
                    aolist.append(aobjects);
            else:
                for ao in aobjects:
                    aolist.append(ao);



    def as_list(self):
        return self._aobjects;

    def select_indices(self, inds):
        newlist = self.__class__();
        for a in inds:
            newlist.append(self[a]);
        return newlist;

    def clone_indices(self, inds):
        out_list = [];
        for oi in inds:
            out_list.append(self._aobjects[oi].clone());
        return out_list;

    def list_attribute(self, name):
        rlist = [];
        for a in self:
            # rlist.append(a.__getattribute__(name));
            rlist.append(getattr(a, name));
        return rlist;

    def map(self, f):
        return map(f, self._aobjects);

    def to_dictionary(self):
        d=super(AObjectList, self).to_dictionary();
        d['aobjects']=self._serialize_objects();
        return d;

    def _serialize_objects(self):
        re = [];
        for e in self._aobjects:
            re.append(e.to_dictionary());
        return re;

    def init_from_dictionary(self, d):
        super(AObjectList, self).init_from_dictionary(d);
        aobjects = d['aobjects'];
        for o in aobjects:
            newo = AObject.create_from_dictionary(o);
            self._aobjects.append(newo);


    
    # <editor-fold desc="Property: 'aobjects'">
    @property
    def _aobjects(self):
        return self._get_aobjects();
    def _get_aobjects(self):
        return self._aobjects_array;
    @_aobjects.setter
    def _aobjects(self, value):
        self._set_aobjects(value);
    def _set_aobjects(self, value):
        self._aobjects_array = value;
    # </editor-fold>
    

    # <editor-fold desc="Container methods for wrapping container aobjects">
    def __len__(self):
        return self._aobjects.__len__();

    def __repr__(self):
        return str(self._aobjects);

    def __getitem__(self, key):
        if(isinstance(key, (tuple, slice))):
            return self.__class__(aobjects=self._aobjects.__getitem__(key));
        else:
            return self._aobjects.__getitem__(key);



    def __setitem__(self, key, value):
        return self._aobjects.__setitem__(key, value);

    def __delitem__(self, key):
        return self._aobjects.__delitem__(key);

    def __missing__(self, key):
        return self._aobjects.__missing__(key);

    def __iter__(self):
        return self._aobjects.__iter__();

    def __reversed__(self):
        return self._aobjects.__reversed__();

    def __contains__(self, item):
        return self._aobjects.__contains__(item);

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._aobjects == other._aobjects;
        return self._aobjects == other

    def __add__(self, other):
        if(isinstance(other, self.__class__)):
            return self._aobjects.__add__(other._aobjects);
        else:
            return self._aobjects.__add__(other);

    def is_empty(self):
        return (len(self._aobjects)==0);

    def length(self):
        if(self._aobjects is not None):
            return len(self._aobjects);

    def append(self, value):
        if(isinstance(value, self.ElementClass)):
            self._aobjects.append(value);
        else:
            self._aobjects.append(self.new_element(value));
    def head(self):
        # get the first element
        return self._aobjects[0];
    def tail(self):
        # get all elements after the first
        return self._aobjects[1:];
    def init(self):
        # get elements up to the last
        return self._aobjects[:-1];
    def last(self):
        # get last element
        return self._aobjects[-1];
    def drop(self, n):
        # get all elements except first n
        return self._aobjects[n:];
    def take(self, n):
        # get first n elements
        return self._aobjects[:n];

    def pop(self, index=-1):
        return self._aobjects.pop(index);

    def index_of(self, value, **kwargs):
        return self._aobjects.index(value, **kwargs)


    def sort(self, cmp=None, key=None, reverse=False):
        """
        def cmp(a,b):
            return a-b;
        cmp function returns negative for less than, positive for greater than, and zero for equal.
        :param cmp:
        :param key:
        :param reverse:
        :return:
        """
        assert(not ((cmp is not None) and (key is not None))), "Provide only cmp or key";
        if(cmp is not None):
            key = ptlreg.apy.utils.cmp_to_key(cmp);
        return self._aobjects.sort(key=key, reverse=reverse);

    def sort_by_attribute(self, attr_name, **kwargs):
        if(self.is_empty()):
            return;
        def takeattr(elem):
            return getattr(elem, attr_name);
        return self._aobjects.sort(key=takeattr, **kwargs);

    def insert(self, index, elem):
        if(isinstance(elem, self.ElementClass)):
            return self._aobjects.insert(index, elem);
        else:
            return self._aobjects.insert(index, self.new_element(elem));

    def remove(self, elem):
        self._aobjects.remove(elem);


    # </editor-fold>

