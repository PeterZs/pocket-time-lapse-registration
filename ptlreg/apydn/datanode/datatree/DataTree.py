from ptlreg.apydn import DataNodeSet, DataNodeConstants
from ptlreg.apydn.datanode.datatree import DataTreeNode
from ptlreg.apydn.datanode.datatree.DataTreeNode import IsDataGraphNode
import pandas as pd


class DataTree(IsDataGraphNode, DataNodeSet):
    """
    A DataNodeGraph is a DataNodeSet that represents a graph of data nodes.
    It is used to represent the relationships between data nodes in a graph structure.
    """

    def __init__(self, *args, **kwargs):
        super(DataTree, self).__init__(*args, **kwargs)
        self.set_node_id(self.node_id)  # Ensure node_id is set correctly

    def get_subnode_dataframe(self):
        df = self.get_data_labels_dataframe();
        if(DataNodeConstants.NODE_PATH_KEY not in df.columns):
            df[DataNodeConstants.NODE_PATH_KEY] = df[self.__class__.NODE_ID_KEY];
        return df


    @property
    def node_path(self):
        return self.get_label(DataNodeConstants.NODE_PATH_KEY);

    @node_path.setter
    def node_path(self, value):
        self.set_label(DataNodeConstants.NODE_PATH_KEY, value);

    @property
    def children_dataframe(self):
        return self.nodes.loc[self.nodes[DataNodeConstants.NODE_PATH_KEY].map(
            lambda x: x.split(DataNodeConstants.NODE_PATH_SEP)[0]) == self.nodes[self.__class__.NODE_ID_KEY]];

    def _set_node(self, node):
        # self._nodes = pd.concat(objs=[self._nodes, node.get_data_labels_subnode_dataframe()], axis=0, ignore_index=True);
        self._nodes = self._nodes.combine_first(node.get_subnode_dataframe());

        # self._nodes.loc[node.get_label(self.index_key)] = node.get_subnode_
        #node.get_series()

    # @property
    # def iloc(self):
    #     return _NodeSubsetAccessor(parent_node_set=self, dataframe=self.children_dataframe.iloc,
    #                                node_set_class=self.get_subset_class());


    def set_node_id(self, id):
        super(DataNodeSet, self).set_node_id(id);
        assert(self.node_path is None), "Tried to set node_id to {} when node_path was already set to {}".format(id, self.node_path);
        if(id is not None):
            self.node_path = self.node_id;

    def get_subnode_dataframe(self):
        """
        Returns a DataFrame representation of the DataNodeSet, including all child nodes.
        Adds the current node to the dataframe, and sets the parent of all the children nodes to the current node.
        :return: DataFrame with all child nodes.
        """
        children = self._nodes.copy()
        if (DataNodeConstants.NODE_PATH_KEY not in children.columns):
            children[DataNodeConstants.NODE_PATH_KEY] = children[self.__class__.NODE_ID_KEY];
        node_path = self.node_path;
        if(node_path is None):
            node_path = self.node_id;
        children[DataNodeConstants.NODE_PATH_KEY] = children[DataNodeConstants.NODE_PATH_KEY].map(
            lambda x: node_path + DataNodeConstants.NODE_PATH_SEP + x);
        label_dataframe = self.get_data_labels_dataframe();
        if (DataNodeConstants.NODE_PATH_KEY not in label_dataframe.columns):
            label_dataframe[DataNodeConstants.NODE_PATH_KEY] = label_dataframe[self.__class__.NODE_ID_KEY];
        # label_dataframe.set_index(self.index_key)
        return pd.concat(objs=[label_dataframe, children], axis=0, ignore_index=True)


    def _add_node_to_end(self, node:DataTreeNode):
        self._nodes = pd.concat(objs=[self._nodes, node.get_subnode_dataframe()], axis=0, ignore_index=True);
        # self._nodes.loc[len(self._nodes)] = node.get_series();