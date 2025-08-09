from ptlreg.apydn.datanode.DataNode import *
from ptlreg.apydn.FPath import *

class DataNodeSetValidationError(Exception):
    def __init__(self, nodeset, message=None):
        self.nodeset = nodeset;
        self.message=message;
        super(DataNodeSetValidationError, self).__init__(self.message)

class DuplicateDataNodeError(Exception):
    def __init__(self, nodeset, oldnode, newnode, message=None):
        self.oldnode = oldnode;
        self.newnode = newnode;
        self.nodeset = nodeset;
        self.message=message;
        super(DuplicateDataNodeError, self).__init__(self.message)

class DataNodeSetValidationError(Exception):
    def __init__(self, nodeset, message=None):
        self.nodeset = nodeset;
        self.message=message;
        super(DataNodeSetValidationError, self).__init__(self.message)

class NodeIterator(object):
    def __init__(self, nodeset, node_class):
        self.nodeset = nodeset;
        self.iterrows = nodeset.nodes.iterrows();
        self.node_class = node_class;

    def __next__(self):
        # print(self.iterrows);
        (index, series) = self.iterrows.__next__();
        return self.node_class.from_series(series);

class NodeIndexIterator(object):
    def __init__(self, nodeset, node_class):
        self.nodeset = nodeset;
        self.iterrows = nodeset.nodes.iterrows();
        self.node_class = node_class;

    def __iter__(self):
        return self;

    def __next__(self):
        (index, series) = self.iterrows.__next__();
        return (index, self.node_class.from_series(series));

class NodeSubsetIterator(object):
    def __init__(self, nodeset, node_class, subset_func):
        self.nodeset = nodeset;
        self.node_class = node_class;

    def __iter__(self):
        return self;

    def __next__(self):
        (index, series) = self.iterrows.__next__();
        return (index, self.node_class.from_series(series));

class _NodeSubsetAccessor(object):
    """
    e.g., for returning .iloc so you can slice nodes and ranges of nodes by a number
    """
    def __init__(self, parent_node_set, dataframe, node_set_class):
        """

        :param parent_node_set:
        :param dataframe:
        :param node_set_class:
        """
        self.parent_node_set = parent_node_set;
        self.nodes = dataframe;
        self.node_set_class = node_set_class;
        self.node_class = self.node_set_class.DATANODE_CLASS


    def __getitem__(self, key):
        if isinstance(key, slice):
            # newset = self.get_nodeset_slice(key)
            newset = self.node_set_class.from_dataframe(self.nodes.__getitem__(key));
            newset._set_data_labels(self.parent_node_set.data_labels.copy());
            return newset;
        else:
            return self.node_class.from_series(self.nodes.__getitem__(key))
            # return self.parent_node_set.set_inherited_labels(self.node_class.from_series(self.nodes.__getitem__(key)))

    # def __setitem__(self, key, newvalue):
    #     return self.nodes.__setitem__(key, newvalue);

    def __iter__(self):
        return NodeIterator(self, self.node_set_class.DATANODE_CLASS);


class DataNodeSetMixin(object):
    DATANODESET_SUBSET_CLASS = NotImplementedError; # When a subset is taken, it is this class
    DATANODE_CLASS = DataNode; # individual elements are this class

    DEFAULT_INDEX_KEY= None
    DATA_FRAME_CLASS = pd.DataFrame;
    DATANODESET_FILE_EXTENSIONS = ['.csv'];

    # For dictionary serialization, what keys to use for nodes and for the index key
    DATANODES_DICT_KEY = 'data_nodes';
    DATANODES_INDEX_DICT_KEY = 'data_node_index';

    @classmethod
    def get_default_index_key(cls):
        if(cls.DEFAULT_INDEX_KEY is None):
            return NotImplementedError;
        else:
            return cls.DEFAULT_INDEX_KEY;



    @classmethod
    def from_csv(cls, path, index_key = None):
        if(index_key is None):
            index_key = cls.DEFAULT_INDEX_KEY;
        newSet = cls(path=path, index_key=index_key);
        return newSet;

    def __init__(self, dataframe=None, index_key=None, *args, **kwargs):
        '''

        :param dataframe:
        :param index_key:
        :param path:
        :param args:
        :param kwargs:
        '''
        # print("HasDataNodeSet.__init__ called with dataframe={} and index_key={}".format(dataframe, index_key));
        self.set_nodes(None, validate=False)
        super(DataNodeSetMixin, self).__init__(*args, **kwargs)
        if (dataframe is not None):
            self.set_nodes(dataframe)
            if (index_key is not None):
                self._set_index_key(index_key);
            return;
        else:
            self.init_nodes(index_key=index_key);
            # self._set_index_key(index_key);


    def init_nodes(self, index_key=None, columns=None):
        """
        If you want to initialize with certain columns for a subclass, override this method with one that calls super and provides those columns.
        :param index_key:
        :param columns:
        :return:
        """
        if (index_key is None):
            index_key = self.__class__.get_default_index_key();
        if (index_key is None):
            # check if it's still None, as in there is no default index key for this class and none was provided
            if(columns is None):
                self.set_nodes(self.__class__.DATA_FRAME_CLASS());
            else:
                self.self.set_nodes(self.__class__.DATA_FRAME_CLASS(columns=columns));
        else:
            if(columns is None):
                columns = [index_key];
            if(not (index_key in columns)):
                columns = [index_key] + columns;
            self.set_nodes(self.__class__.DATA_FRAME_CLASS(columns=columns));
        if (index_key is not None):
            self._set_index_key(index_key)


    @classmethod
    def validate_data_node_set(cls, node_set):
        if(not isinstance(node_set, DataNodeSetMixin)):
            raise DataNodeSetValidationError(node_set, "{} is not a DataNodeSet".format(node_set));
        else:
            return cls._validate_data_node_set_dataframe(node_set._nodes);

    @classmethod
    def _validate_data_node_set_dataframe(cls, nodes):
        if (not isinstance(nodes, pd.DataFrame)):
            raise DataNodeSetValidationError(nodes, "{} is not a DataFrame".format(nodes));
        else:
            return cls.DATANODE_CLASS.DATANODE_SCHEMA.validate(nodes);


    @classmethod
    def load_data_node_set(cls, path, index_key=None):
        rval = cls();
        rval._load_data_node_set_from_path(path=path, index_key=index_key);
        return rval;

    def _load_data_node_set_from_path(self, path, index_key=None):
        if(index_key is None):
            index_key = self.__class__.DEFAULT_INDEX_KEY;
        fpath = FPath.From(path);
        if(fpath.file_ext == ".csv"):
            dataframe = pd.read_csv(path);
        elif(fpath.file_ext == ".json"):
            dataframe = pd.read_json(path);
        else:
            dataframe = pd.read_pickle(path);
        self.set_nodes(dataframe);
        self._set_index_key(index_key, drop=False);

    def save_nodes_to_path(self, output_path):
        fpath = FPath.From(output_path);
        tosave = self.nodes.reset_index()
        if (fpath.file_ext == ".csv"):
            tosave.to_csv(output_path);
        elif (fpath.file_ext == ".json"):
            tosave.to_json(output_path);
        else:
            tosave.to_pickle(output_path);

    @property
    def index_key(self):
        return self.nodes.index.name

    def _set_index_key(self, index_key, drop=False):
        if(index_key is None):
            self.nodes.reset_index(inplace=True, names=self.index_key);
        else:
            self.nodes.set_index(index_key, inplace=True, drop=drop)

    def __getitem__(self, key):
        return self.DATANODE_CLASS.from_series(self.nodes.loc.__getitem__(key));

    def __setitem__(self, key, newvalue):
        if(isinstance(newvalue, DataNode)):
            return self.nodes.loc.__setitem__(key, newvalue.data_labels);
        else:
            return self.nodes.loc.__setitem__(key, newvalue);

    def __iter__(self):
        return self._node_iterator();


    def _node_iterator(self):
        return NodeIterator(self, self.__class__.DATANODE_CLASS);

    def index_iterator(self):
        return NodeIndexIterator(self, self.__class__.DATANODE_CLASS);

    @classmethod
    def get_subset_class(cls):
        if(cls.DATANODESET_SUBSET_CLASS is not None):
            return cls.DATANODESET_SUBSET_CLASS;
        else:
            return cls;

    def subset(self, *args, **kwargs):
        """
        Construct a subset
        :param args:
        :param kwargs:
        :return:
        """
        return self.__class__.get_subset_class()(*args, **kwargs)


    @classmethod
    def from_dataframe(cls, dataframe, index_key=None, drop_index=True):
        rval = cls(dataframe=dataframe);
        if(index_key is not None):
            rval._set_index_key(index_key, drop=drop_index);
        return rval;

    def _add_node_with_key(self, node, key=None):
        """

        :param node: dictionary or series
        :param key:
        """
        if(key is not None):
            node = node.copy();
            node[self.index_key]=key;
        self.set_node(key, node);


    def has_datanode_with_index_key(self, key):
        return key in self.nodes.index;



    # def get_nodeset_slice(self, slice):
    #     """
    #     Returns a subset of the nodeset based on the provided slice. The tricky part here is that the full dataframe contains the descendents of children.
    #     :param slice:
    #     :return:
    #     """
    #     raise NotImplementedError;


    def get_node_with_index(self, index):
        return self.nodes.loc[index];

    def get_node(self, index):
        if (isinstance(index, DataNode)):
            return self.get_node_with_index(index.get_label(self.index_key));
        return self.get_node_with_index(index);
        # if (isinstance(key, DataNode)):
        #     return self.nodes.loc[key.get_label(self.index_key)];
        # return self.nodes.loc[key];

    def has_node(self, query):
        if(isinstance(query, DataNode)):
            return self.has_datanode_with_index_key(query.get_label(self.index_key));
        else:
            return self.has_datanode_with_index_key(query);


    def add_node(self, node, error_on_duplicate=True):
        if(self.index_key and error_on_duplicate and self.has_node(node)):
            raise DuplicateDataNodeError(nodeset=self, oldnode=self.get_node(node), newnode=node)
        elif(not self.index_key):
            self._add_node_to_end(node)
        else:
            self.set_node(node.get_label(self.index_key), node);


    def add_label_for_nodes(self, label_key, values=None):
        if(values is None):
            values = np.NaN;
        self.nodes[label_key] = values;

    def calc_label_for_nodes(self, label_key, fn):
        """
        Calculate a label for all nodes in the set using the provided function.
        :param label_key: The key for the label to be calculated.
        :param fn: A function that takes a DataNode and returns a value for the label.
        """
        # warnings.warn("Test this function calc_label_for_nodes, it was AI generated.", UserWarning);
        self.nodes[label_key] = self.nodes.apply(lambda row: fn(self.DATANODE_CLASS.from_series(row)), axis=1);


    def set_node(self, key, node, validate=True):
        if(validate):
            self.__class__.DATANODE_CLASS.validate_data_node(node);

        if (self.has_datanode_with_index_key(key)):
            self.remove_node(key);
        self._set_node(node=node);
        # self.dataframe = pd.concat([self._dataframe, pd.DataFrame([row])], ignore_index=False);

    def _add_node_to_end(self, node:HasDataLabels):
        self._nodes.loc[len(self._nodes)] = node.get_series();

    def _set_node(self, node):
        raise NotImplementedError;

    def remove_node(self, node_or_key):
        if(isinstance(node_or_key, DataNode)):
            return self.nodes.drop(index=node_or_key.get_label(self.index_key), inplace=True);
        else:
            return self.nodes.drop(index=node_or_key, inplace=True);

    @property
    def iloc(self):
        return _NodeSubsetAccessor(parent_node_set=self, dataframe=self.nodes.iloc,
                                   node_set_class=self.get_subset_class());


    @classmethod
    def get_merged_intersect(cls, A, B, **kwargs):
        return cls.from_dataframe(A.nodes.merge(B.nodes, how="inner", **kwargs));

    @classmethod
    def get_merged_union(cls, A, B, **kwargs):
        return cls.from_dataframe(A.nodes.merge(B.nodes, how="outer", **kwargs));

    @classmethod
    def get_merged_left(cls, A, B, **kwargs):
        return cls.from_dataframe(A.nodes.merge(B.nodes, how="left", **kwargs));

    @classmethod
    def get_merged_right(cls, A, B, **kwargs):
        return cls.from_dataframe(A.nodes.merge(B.nodes, how="left", **kwargs));

    @classmethod
    def get_merged_cross(cls, A, B, **kwargs):
        return cls.from_dataframe(A.nodes.merge(B.nodes, how="cross", **kwargs));

    def get_selection_by_function(self, fn):
        # raise NotImplementedError;
        rlist = [];
        for s in self:
            if(fn(s)):
                rlist.append(s);
        return self.__class__.DATANODESET_SUBSET_CLASS(rlist);

    def get_with_label_value(self, key, value):
        raise NotImplementedError;

    def get_with_tag(self, tag_name):
        raise NotImplementedError;

    def sort_by_label(self, key, inplace=True, **kwargs):
        self.nodes.sort_values(by=key, inplace=inplace, **kwargs)
        return self;

    # def get_label_series(self, key):
    #     return self.nodes[key];

    def get_series_for_label(self, key):
        return self.nodes[key];

    @classmethod
    def data_nodes_from_dict(cls, d):
        return pd.DataFrame.from_dict(d);

    @classmethod
    def data_nodes_from_node_list(cls, l):
        return pd.DataFrame.from_dict(pd.Series(l).to_dict()).T;
        # return pd.DataFrame.from_dict(d);

    @classmethod
    def data_nodes_to_node_list(cls, nodes):
        if(nodes.index.name):
            return list(nodes.reset_index().T.to_dict().values());
        else:
            return list(nodes.T.to_dict().values());

    def _sort_node_columns(self, **kwargs):
         self.nodes = self.nodes.reindex(sorted(self.nodes.columns, **kwargs), axis=1);

    def get_label_names(self):
        return self.nodes.columns;

    def to_dictionary(self):
        d = {};
        if (hasattr(super(DataNodeSetMixin, self), 'to_dictionary')):
            d = super(DataNodeSetMixin, self).to_dictionary();
        d[self.__class__.DATANODES_DICT_KEY] = self.nodes.to_dict();
        index_key = self.index_key;
        if(index_key is not None):
            d[self.__class__.DATANODES_INDEX_DICT_KEY] = self.index_key;
        return d;

    def init_from_dictionary(self, d):
        if (hasattr(super(DataNodeSetMixin, self), 'init_from_dictionary')):
            super(DataNodeSetMixin, self).init_from_dictionary(d);
        self.nodes = self.__class__.data_nodes_from_dict(d[self.__class__.DATANODES_DICT_KEY]);
        if(self.__class__.DATANODES_INDEX_DICT_KEY in d):
            self._set_index_key(d[self.__class__.DATANODES_INDEX_DICT_KEY]);


    def to_node_dictionary(self):
        """
        uses `__class__.data_nodes_to_node_list` instead of `self.nodes.to_dict()`,
        This returns a list of nodes in the DATANODES_DICT_KEY entry(?)
        :return:
        """
        d = {};
        if (hasattr(super(DataNodeSetMixin, self), 'to_dictionary')):
            d = super(DataNodeSetMixin, self).to_dictionary();
        d[self.__class__.DATANODES_DICT_KEY] = self.__class__.data_nodes_to_node_list(self.nodes);
        index_key = self.index_key;
        if(index_key is not None):
            d[self.__class__.DATANODES_INDEX_DICT_KEY] = self.index_key;
        return d;

    def init_from_node_dictionary(self, d):
        """
        uses `__class__.data_nodes_from_node_list` instead of `pd.DataFrame.from_dict()`,
        :param d:
        :return:
        """
        if (hasattr(super(DataNodeSetMixin, self), 'init_from_dictionary')):
            super(DataNodeSetMixin, self).init_from_dictionary(d);
        self.nodes = self.__class__.data_nodes_from_node_list(d[self.__class__.DATANODES_DICT_KEY]);
        if(self.__class__.DATANODES_INDEX_DICT_KEY in d):
            self._set_index_key(d[self.__class__.DATANODES_INDEX_DICT_KEY]);

    @classmethod
    def create_set_from_dictionary(cls, d):
        rval = cls();
        rval.init_from_dictionary(d);
        return rval;


    @classmethod
    def create_set_from_node_dictionary(cls, d):
        rval = cls();
        rval.init_from_node_dictionary(d);
        return rval;


    # @classmethod
    # def _SERIALIZERS(cls):
    #     def to_json(dns, obj: None):
    #         if(dns is not None):
    #             return dns.to_dictionary()
    #         else:
    #             return None;
    #     def from_json(d: dict(), obj: None):
    #         return cls.InitFromDictionary(d);
    #     return {
    #         "to_json":to_json,
    #         "from_json":from_json
    #     };

    @classmethod
    def _NODESERIALIZERS(cls):
        def to_json(dns, obj: None):
            if(dns is not None):
                return dns.to_node_dictionary()
            else:
                return None;
        def from_json(d: dict(), obj: None):
            return cls.create_set_from_node_dictionary(d);
        return {
            "to_json":to_json,
            "from_json":from_json
        };

    @classmethod
    def _LISTOFNODESETSERIALIZERS(cls):
        def to_json(nsl, obj: None):
            if(nsl is None):
                return None;
            rlist = [];
            for a in nsl:
                rlist.append(a.to_node_dictionary());
            return rlist;

        def from_json(l, obj: None):
            rlist = [];
            for a in l:
                rlist.append(cls.create_set_from_node_dictionary(a));
            return rlist;

        return {
            "to_json":to_json,
            "from_json":from_json
        };

    @classmethod
    def _DICTOFNODESETSERIALIZERS(cls):
        def to_json(nsd, obj: None):
            if (nsd is None):
                return None;
            rdict = {};
            for a in nsd:
                rdict[a]=nsd[a].to_node_dictionary();
            return rdict;

        def from_json(d, obj: None):
            rdict = {};
            for a in d:
                rdict[a]=cls.create_set_from_node_dictionary(d[a]);
            return rdict;

        return {
            "to_json": to_json,
            "from_json": from_json
        };

    @classmethod
    def _clone_with_nodes(cls, data_node_set, new_nodes):
        new_set = cls.from_dataframe(new_nodes);
        new_set._data_labels = data_node_set.data_labels;
        return new_set;

    @classmethod
    def create_from_data_node_set_dict(cls, d):
        rval = cls();
        rval.init_from_dictionary(d);
        return rval;




def HasDataNodeSetMethod(func):
    setattr(DataNodeSetMixin, func.__name__, func)
