import pandas as pd
from .CDBTable import *
from ptlreg.apydn.datanode.DataNode import DataNode

try:
    import networkx as nx
except:
    print("import of networkx failed. Perhaps try installing graphviz? With homebrew on osx, `brew install graphviz`.")

# This is all the matches for one image pair
class COLMAPFeatureMatchesNode(DataNode):
    @property
    def image_ids(self):
        return pair_id_to_image_ids(self.get_label('pair_id'));

    @property
    def pair_id(self):
        return self.get_label('pair_id');

    @property
    def n_matches(self):
        return self.get_label('n_matches');

    @property
    def matches(self):
        return self.get_label('data');


# This is a table of matches for all image pairs
class CDBMatches(CDBTable):
    DATANODE_CLASS = COLMAPFeatureMatchesNode;
    DATANODESET_FILE_EXTENSIONS = ['.cdbmatches'];
    DEFAULT_INDEX_KEY = "pair_id"

    @classmethod
    def from_dataframe_with_binary_info(cls, dataframe: pd.DataFrame):
        matches = dataframe.copy();
        # matches['data'] = matches['data'].map(lambda x: blob_to_array(x, np.uint32, (-1, 2)))
        matches['data'] = matches['data'].map(lambda x: blob_to_array(x, np.uint32, (-1, 2)) if x else np.NaN)
        matches['pair_ids'] = matches['pair_id'].map(lambda x: pair_id_to_image_ids(x))
        return cls.from_dataframe(matches.set_index(cls.DEFAULT_INDEX_KEY, drop=False));



    def get_edge_graph_dataframe(self):
        return pd.DataFrame(dict(
            source=self.nodes.pair_ids.map(lambda x: x[0]),
            target=self.nodes.pair_ids.map(lambda x: x[1])
        ))

    def get_nxgraph(self):
        return nx.Graph(self.get_edge_graph_dataframe().reset_index())



    def get_matches_for_image_id_pair(self, first_id, second_id):
        '''
        Order doesn't matter
        :param first_id:
        :param second_id:
        :return:
        '''
        indx = image_ids_to_pair_id(first_id, second_id);
        if (indx in self.nodes.index and isinstance(self[indx].matches, np.ndarray)):
            pair_order = pair_id_to_image_ids(indx);
            if (pair_order[0] == first_id):
                return self[indx].matches;
            else:
                return self[indx].matches[:,[1,0]];
        else:
            return None;


