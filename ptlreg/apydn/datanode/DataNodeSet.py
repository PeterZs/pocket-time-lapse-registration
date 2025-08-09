from .DataNodeSetMixin import *
from .DataNode import *
import pandas as pd

DATANODESET_EXTENSIONS_DICT = {}
_DATANODESET_TYPES = {}

class DataNodeSet(DataNodeSetMixin, DataNode):
    """
    You don't need to use DataNodeSet class if you don't want to use the meta class for registering.
    You can just use HasDataNodeSet.
    """

    # DATANODE_SCHEMA = pa.DataFrameSchema({
    #     DataNodeConstants.NODE_ID_KEY: pa.Column(str)
    # });

    DATANODESET_SUBSET_CLASS = None
    # DEFAULT_INDEX_KEY = None
    DATA_FRAME_CLASS = pd.DataFrame

    @classmethod
    def get_default_index_key(cls):
        return cls.DEFAULT_INDEX_KEY

    def set_nodes(self, nodes, validate=True):
        if validate:
            self.__class__._validate_data_node_set_dataframe(nodes)
        self._nodes = nodes

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.nodes.__str__())

    def __repr__(self):
        return self.nodes.__repr__()

    @property
    def nodes(self):
        return self._nodes

    # @nodes.setter
    # def nodes(self, nodes):
    #     self.set_nodes(nodes)

    # def __next__(self):  # Python 2: def next(self)
    #     self.current += 1
    #     if self.current < self.high:
    #         return self.current
    #     raise StopIteration

    # def _childnode_dataframe_from_node(self, child:DataNode)->pd.DataFrame:
    #     """
    #     Converts a DataNode to a DataFrame with the current node's id added to all of the node_paths
    #     :param child: DataNode to convert.
    #     :return: DataFrame representation of the DataNode.
    #     """
    #     return child.get_data_labels_dataframe()

    def _set_node(self, node):
        # self._nodes = pd.concat(objs=[self._nodes, node.get_data_labels_subnode_dataframe()], axis=0, ignore_index=True);
        # self._nodes = self._nodes.combine_first(node.get_subnode_dataframe());
        # self._nodes = self._nodes.merge(node.get_data_labels_dataframe(index=self.__class__.NODE_ID_KEY), how="outer");
        self._nodes.loc[node.get_label(self.index_key)] = node.get_series()




    # def _add_node_to_end(self, node:DataNode):
    #     self._nodes = pd.concat(objs=[self._nodes, node.get_subnode_dataframe()], axis=0, ignore_index=True);
    #     # self._nodes.loc[len(self._nodes)] = node.get_series();




def DataNodeSetMethod(func):
    setattr(DataNodeSet, func.__name__, func)
    return getattr(DataNodeSet, func.__name__)


DataNodeSet.DATANODESET_SUBSET_CLASS = DataNodeSet