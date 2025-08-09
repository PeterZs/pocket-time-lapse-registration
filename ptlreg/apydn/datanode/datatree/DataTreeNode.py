from ptlreg.apydn import DataNode, DataNodeConstants


class IsDataGraphNode:
    """
    Mixin for being a DataGraphNode without inheriting from DataNode.
    A DataGraphNode is a DataNode that represents a node in a DataGraph.
    It is used to represent the relationships between data nodes in a graph structure.
    """

    def get_subnode_dataframe(self):
        df = self.get_data_labels_dataframe();
        if(DataNodeConstants.NODE_PATH_KEY not in df.columns):
            df[DataNodeConstants.NODE_PATH_KEY] = df[self.__class__.NODE_ID_KEY];
        return df




class DataGraphNode(IsDataGraphNode, DataNode):
    """
    DataGraphNode is a class that represents a node in a DataGraph.
    It is used to store and manage data in a graph structure.
    """

    def __init__(self, *args, **kwargs):
        super(DataGraphNode, self).__init__(*args, **kwargs)
        self.set_node_id(self.node_id)  # Ensure node_id is set correctly