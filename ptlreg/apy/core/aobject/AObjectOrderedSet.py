# from abepy.apy.AObject import AObject, AObjectMeta, AOBJECT_CLASSES_DICT
from ptlreg.apy.core import *
from ptlreg.apy.core.aobject.AObjectList import *
from ptlreg.apy.core.dicts.IndexDict import *

class IsAObjectOrderedSet(HasIndexDict):
    '''
    Works like a list that validates items. Items need to be of the relevant class, and the same item cannot be inserted
    more than once.

    The HasIndexDict class means that the class has a dictionary that maps unique indices to elements.

    Note that append() does validation / addition to index, and that if subclassing AObjectList, the constructor will
    call append on each item in an argument list.

    '''


    @classmethod
    def _validate_element_class(cls, elem):
        assert (isinstance(elem, cls.ElementClass)), "{} is not an instance of {}".format(elem.__class__, cls.ElementClass);
        return elem;

    def _validate_element(self, elem):
        return self._validate_element_class(elem);


    def main_index_map_func(self, o):
        return o;

    @property
    def _main_index_map(self):
        return self.get_index_map('main_index');

    def _set_default_instance_index_map_funcs(self):
        self._add_index_map_func(func=self.__class__.main_index_map_func, index_name='main_index', _is_unique_id=True);
        return;

    def __init__(self, *args, **kwargs):
        super(IsAObjectOrderedSet, self).__init__(*args, **kwargs);

    def __contains__(self, key):
        if(isinstance(key, self.ElementClass)):
            return key in self._main_index_map.values();
        else:
            return key in self._main_index_map;

    def add(self, obj):
        self.append(obj);

    def discard(self, obj):
        self.remove(obj);


    ##################//--Reimplemented from AObjectList--\\##################
    # <editor-fold desc="Reimplemented from AObjectList">
    def __add__(self, other):
        # if(isinstance(other, self.ElementClass)):
        #     self.append(other);
        # else:
        for o in other:
            self.append(o);

    def __delitem__(self, key):
        self._remove_indexed_object(self.__getitem__(key));
        return self._aobjects.__delitem__(key);

    def __getitem__(self, key):
        if (isinstance(key, (tuple, slice))):
            return self.__class__(aobjects=self._aobjects.__getitem__(key));
        else:
            return self._aobjects.__getitem__(key);

    def __setitem__(self, key, value):
        vitem = self._validate_element(value);
        to_replace = self.__getitem__(key);
        if(to_replace!=vitem):
            self._add_indexed_object(vitem);
            self._remove_indexed_object(to_replace);
        return self._aobjects.__setitem__(key, vitem);

    def __contains__(self, item):
        return self._main_index_map.get(self.main_index_map_func(item)) is not None;

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._aobjects == other._aobjects;
        return self._aobjects == other

    def append(self, value):
        vitem = self._validate_element(value);
        self._add_indexed_object(vitem);
        super(IsAObjectOrderedSet, self).append(vitem);

    def pop(self, index=-1):
        robj = self._aobjects.pop(index);
        self._remove_indexed_object(robj);
        return robj;

    def remove(self, elem):
        self._remove_indexed_object(elem);
        self._aobjects.remove(elem);

    def remove_subset(self, elems):
        for e in elems:
            self.remove(e);

    def __or__(self, other):
        return self.union_of(self, other);

    @classmethod
    def union_of(cls, *sets):
        def addToUnion(a, b):
            for o in b:
                if(not a._unique_id_is_taken(o)):
                    a.append(o);
        if(len(sets)==0):
            return cls();
        if(len(sets)==1):
            return cls(sets[0].asList())
        union = cls(sets[0].asList());
        for a in sets[1:]:
            addToUnion(union, a);
        return union;

    def union_with(self, *sets):
        for a in sets:
            for b in a:
                if(not self._unique_id_is_taken(b)):
                    self.append(b);
        # return self.__class__.Union(self, *sets);
        # for a in sets:
        #     for b in a:
        #         if(not self._unique_id_is_taken(b)):
        #             self.append(b);

    def intersection_with(self, *sets):
        return self.__class__.intersection_of(self, *sets);

    @classmethod
    def intersection_of(cls, *sets):
        def intersect(a, b):
            rvl = cls();
            for o in b:
                if(a._unique_id_is_taken(o)):
                    rvl.append(o);
            return rvl;
        if(len(sets)==0):
            return cls();
        if(len(sets)==1 and hasattr(sets[0], "asList")):
            return cls(sets[0].asList())
        intersection = sets[0];
        if(len(sets)>1):
            for a in sets[1:]:
                intersection = intersect(intersection, a);
        return intersection;


    def init_from_dictionary(self, d):
        super(IsAObjectOrderedSet, self).init_from_dictionary(d);
        for ai in range(len(self._aobjects)):
            a = self._aobjects[ai];
            vitem = self._validate_element(a);
            if(vitem is not a):
                ptlreg.apy.utils.AWARN("Object changed in AObjectOrderedSet.init_from_dictionary due to validation.")
                self._aobjects[ai]=vitem;
            self._add_indexed_object(vitem);
        # self._recomputeAllIndexMaps();

    @classmethod
    def from_dictionary(cls, d):
        newitem = cls();
        newitem.init_from_dictionary(d);
        return newitem;


    # </editor-fold>
    ##################\\--Reimplemented from AObjectList--//##################

class AObjectOrderedSetMeta(HasIndexDictMeta, AObjectMeta):
    pass;


class AObjectOrderedSet(six.with_metaclass(AObjectOrderedSetMeta, IsAObjectOrderedSet, AObjectList)):
    pass;

#
# @six.add_metaclass(AObjectOrderedSetMeta)
# class AObjectOrderedSet(IsAObjectOrderedSet, AObjectList):
#     pass;
#     # __metaclass__ = AObjectOrderedSetMeta;