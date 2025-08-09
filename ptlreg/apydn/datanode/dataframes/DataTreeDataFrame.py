from ptlreg.apydn import HasDataLabels
from ptlreg.apydn.DataNodeConstants import *
import pandas as pd
import bigtree as bt


@pd.api.extensions.register_dataframe_accessor("dtree")
class DataTreeAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if DataNodeConstants.NODE_ID_KEY not in obj.columns or DataNodeConstants.NODE_PATH_KEY not in obj.columns:
            raise AttributeError("Must have `{}` and `{}`.".format(DataNodeConstants.NODE_ID_KEY, DataNodeConstants.NODE_PATH_KEY))

    @property
    def children_dataframe(self):
        return self._obj[self._obj[DataNodeConstants.NODE_PATH_KEY].map(lambda x: x.split(DataNodeConstants.NODE_PATH_SEP)[0]) == self._obj[DataNodeConstants.NODE_ID_KEY]];

    @property
    def _child_root_node_ids(self):
        return self._obj[DataNodeConstants.NODE_PATH_KEY].map(lambda x: x.split(DataNodeConstants.NODE_PATH_SEP)[0])

    @property
    def node_ids(self):
        return self._obj[DataNodeConstants.NODE_PATH_KEY]

    @property
    def node_paths(self):
        return self._obj[DataNodeConstants.NODE_PATH_KEY]

    def get_child_subframe(self, node_id):
        """
        Returns dataframe view with the child and its subtree, given the child or its node_id.
        :param node_id: ID of the child node to retrieve (or the child itself).
        :return: DataFrame row corresponding to the child node.
        """
        if (isinstance(node_id, HasDataLabels)):
            node_id = node_id.node_id;
        child_row = self._obj[self._obj["node_id"] == node_id]
        child_path = child_row[DataNodeConstants.NODE_PATH_KEY].values[0]
        # return child_path
        child_df = self._obj[self._obj[DataNodeConstants.NODE_PATH_KEY].map(lambda x: x.startswith(child_path))]
        child_df.loc[:, (DataNodeConstants.NODE_PATH_KEY)] = child_df[DataNodeConstants.NODE_PATH_KEY].map(
            lambda x: node_id + x.split(node_id, 1)[1])
        return child_df


    def get_child_dataframe_copy(self, node_id):
        """
        Returns dataframe of child node with the given node_id. Creates a new dataframe, not a view.
        :param node_id: ID of the child node to retrieve.
        :return: DataFrame row corresponding to the child node.
        """
        if (isinstance(node_id, HasDataLabels)):
            node_id = node_id.node_id;
        child_row = self._obj[self._obj["node_id"] == node_id]
        child_path = child_row[DataNodeConstants.NODE_PATH_KEY].values[0]
        # return child_path
        child_df = self._obj[self._obj[DataNodeConstants.NODE_PATH_KEY].map(lambda x: x.startswith(child_path))].copy()
        child_df.loc[:, (DataNodeConstants.NODE_PATH_KEY)] = child_df[DataNodeConstants.NODE_PATH_KEY].map(
            lambda x: node_id + x.split(node_id, 1)[1])
        return child_df

    def _remove_from_path_roots(self, to_remove):
        """
        removes the given string from the start of all node paths. If the resulting path is empty, the row is removed.
        This can be used to convert a subnode tree back to the node tree (the row for that node should be taken and assigned as labels separately, though).
        :param to_remove:
        :return:
        """
        def remove_first_sep(x):
            if(x and x[0]==DataNodeConstants.NODE_PATH_SEP):
                return x[1:];
            else:
                return x;
        self._obj.loc[:, (DataNodeConstants.NODE_PATH_KEY)] = self._obj[DataNodeConstants.NODE_PATH_KEY].map(
            lambda x: remove_first_sep(x.split(to_remove, 1)[1]))
        self._obj.drop(self._obj.loc[self._obj[DataNodeConstants.NODE_PATH_KEY]==""].index, inplace=True)

    def bigtree_graph(self,
                      path_col = None,
                      attribute_cols = None,
                      sep: str = None,
                      duplicate_name_allowed: bool = True,
                      bt_node_type = None
                      ):
        """

        :param path_col: name of column containing path string
        :param attribute_cols: attribute columns
        :param sep: separator for path segments
        :param duplicate_name_allowed:
        :param node_type:
        :return:
        """
        if(sep is None):
            sep = DataNodeConstants.NODE_PATH_SEP
        if(path_col is None):
            path_col = DataNodeConstants.NODE_PATH_KEY

        if(bt_node_type is None):
            bt_node_type = bt.node.node.Node;

        return bt.dataframe_to_tree(
            data = self._obj,
            path_col =  path_col,
            attribute_cols= attribute_cols,
            sep= sep,
            duplicate_name_allowed= duplicate_name_allowed,
            node_type = bt_node_type
        )

    def pydot_graph(self, path_col = None,
                      attribute_cols = None,
                      sep: str = None,
                      duplicate_name_allowed: bool = True,
                        **kwargs
                    ):
        """
        Creates a pydot graph from the DataFrame, using the specified path and attribute columns.
        :param path_col:
        :param attribute_cols:
        :param sep:
        :param duplicate_name_allowed:
        :param bt_node_type:
        :return:
        """
        tree = self.bigtree_graph(path_col = path_col,attribute_cols = attribute_cols, sep=sep, duplicate_name_allowed=duplicate_name_allowed);
        return bt.tree_to_dot(tree, **kwargs)

    def plot(self):
        # plot this array's data on a map, e.g., using Cartopy
        pass

# class NodeSetDataframe()
#
#



