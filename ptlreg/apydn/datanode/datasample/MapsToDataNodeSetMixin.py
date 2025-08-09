from ptlreg.apydn import DataNodeConstants
from ptlreg.apydn.datanode.datasample.MapsToDataNodeMixin import MapsToDataNodeMixin
from ptlreg.apydn.datanode.DataNodeSet import DataNodeSet
import pandas as pd

class MapsToDataNodeSetMixin(MapsToDataNodeMixin):
    """
    MapsToDataNodeSet is a mixin for objects that are themselves containers that map to a DataNodeSet.
    It extends MapsToDataNode to handle multiple DataNodes.
    When added to a class, the class must also inherit from AObjectOrderedSet.
    """
    DATANODE_MAP_TYPE = DataNodeSet;
    ElementClass = MapsToDataNodeMixin;
    INDEX_MAP_LABEL_KEY = DataNodeConstants.NODE_ID_KEY;

    # DATANODE_SET_MAP_TYPE = DataNodeSet;
    SUBSET_CLASS = None;

    @classmethod
    def SubsetConstructor(cls, *args, **kwargs):
        """
        Returns the constructor for the subset class.
        :return: SubsetClass constructor
        """
        if(cls.SUBSET_CLASS is None):
            raise NotImplementedError("Subclass {} must implement SubsetClass".format(cls.__name__));
        return cls.SUBSET_CLASS(*args, **kwargs);

    def main_index_map_func(self, o):
        """
        Main index map function for the DataNodeSet.
        :param o:
        :return:
        """
        return o.get_label_value(self.INDEX_MAP_LABEL_KEY);
        # return o.node_id;

    @classmethod
    def _data_node_for_node(cls, node, label_keys=None):
        return cls.DATANODE_MAP_TYPE.DATANODE_CLASS.FromSeries(
            series=pd.Series(node.get_serializable_value_dict_for_label_keys(label_keys=label_keys)));


    def DataNodeSet(self, index_key=None, label_keys = None, **kwargs):
        """
        Creates a DataNodeSet from the current object.
        :param index_key:
        :param label_keys:
        :param kwargs:
        :return:
        """
        df = self.DataFrame(label_keys=label_keys, index_key=index_key, **kwargs);
        return  self.__class__.DATANODE_MAP_TYPE.from_dataframe(df);

    def DataFrame(self, label_keys = None, index_key=None, drop_index_col = True, **kwargs):
        """
        Creates a pandas DataFrame from the current object.
        :param label_keys:
        :param index_key:
        :param drop_index_col:
        :param kwargs:
        :return:
        """
        series_list = [];
        labels_to_use = None;
        if(label_keys):
            labels_to_use = label_keys.copy();
            if(index_key and index_key not in labels_to_use):
                labels_to_use.append(index_key);
        for s in self:
            # self._DataNodeForNode(s, label_keys=label_keys)
            series_list.append(pd.Series(s.get_serializable_value_dict_for_label_keys(label_keys=labels_to_use)));
        df = pd.DataFrame(series_list);
        if (index_key):
            df.set_index(index_key, inplace=True, drop=drop_index_col);
        return df

    @classmethod
    def create_with_id(cls, id, *args, **kwargs):
        rval = cls(*args, **kwargs);
        rval.set_node_id(id);
        return rval;

    def __str__(self):
        rstr = "{} with {} nodes:\n".format(self.__class__.__name__, len(self));
        for a in self:
            rstr = rstr + "{},\n".format(a.node_id);
        return rstr;

    def _get_internal_node_for_node(self, node):
        index = self.main_index_map_func(node);
        return self._main_index_map.get(index);

    @property
    def _main_index_map(self):
        return self.get_index_map('main_index');

    def _validate_element(self, elem):
        # if (hasattr(super(MapsToDataNodeSetMixin, self), '_validate_element')):
        #     super(MapsToDataNodeSetMixin, self)._validate_element(elem);
        velem = super(MapsToDataNodeSetMixin, self)._validate_element(elem);
        return velem;

    def get_selection(self, fn):
        rlist = [];
        for s in self:
            if(fn(s)):
                rlist.append(s);
        return self.__class__.SUBSET_CLASS(rlist);

    def get_selection_for_label_values(self, label_key, label_values):
        '''
        get all samples where the label specified by label_key takes on a value in label_values
        '''
        def selectfunc(imnods):
            return (imnods.get_label_value(label_key) in label_values);
        return self.get_selection(selectfunc);

    def _get_first_with_node_id(self, node_id):
        for s in self:
            if(s.node_id == node_id):
                return s;

    def get_with_label_value(self, key, value):
        map_name = 'label_'+key;
        existing_map = self.get_index_map(map_name);
        def tagindexfunc(set, object):
            return object.get_label_value(key);
        if(existing_map is None):
            self._add_index_map_func(func=tagindexfunc, index_name=map_name, _is_unique_id=False);
        else:
            self._recompute_index_map(map_name);
        return self.__class__.SubsetConstructor(self.get_index_map(map_name).get(value));

    def get_with_timestamp(self, value):
        return self.get_with_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY, value);

    def get_with_tag(self, tag_name):
        map_name = 'tag_'+tag_name;
        existing_map = self.get_index_map(map_name);
        def tagindexfunc(set, object):
            return object.get_label_value(tag_name);
        if(existing_map is None):
            self._add_index_map_func(func=tagindexfunc, index_name=map_name, _is_unique_id=False);
        else:
            self._recompute_index_map(map_name);
        return self.__class__.SubsetConstructor(self.get_index_map(map_name).get(True));

    def get_with_tags(self, tag_names):
        subsets = [];
        for tag in tag_names:
            subsets.append(self.get_with_tag(tag));
        return self.__class__.Intersection(subsets);

    def sort_by_label(self, key, default_value=None, **kwargs):
        if(self.is_empty()):
            return;
        def takelabel(elem):
            if(elem.has_label(key)):
                return elem.get_label_value(key);
            else:
                return default_value;
        self._aobjects.sort(key=takelabel, **kwargs);
        print(self)
        return self;

    def sort_by_timestamp(self, **kwargs):
        if(self.is_empty()):
            return;
        def takelabel(elem):
            return elem.timestamp;
        self._aobjects.sort(key=takelabel, reverse=False, **kwargs);
        return self;

    def get_timestamps(self):
        tslist = [];
        for n in self:
            tslist.append(n.timestamp);
        return tslist;


    def get_flattened_list(self):
        slist = [];
        for n in self:
            if(isinstance(n, MapsToDataNodeSetMixin)):
                slist = slist+n.get_flattened_list();
            else:
                 slist.append(n);
        return slist;






    # def DataNode(self, label_keys=None):
    #     raise NotImplementedError("Subclass needs to implement DataNode method that returns a DataNodeSet instance.")


