import pandas as pd
from .CDBTable import *
from ptlreg.apydn.datanode.DataNode import DataNode

try:
    import networkx as nx
except:
    print("import of networkx failed. Perhaps try installing graphviz? With homebrew on osx, `brew install graphviz`.")



class TVGeometry(DataNode):
    def __init__(self, from_sample, to_sample, tvgnode):
        self.from_sample = from_sample;
        self.to_sample = to_sample;
        self.tvgnode = tvgnode;

    def H(self, from_sample, to_sample):
        if(isinstance(self.tvgnode.H, np.ndarray)):
            if(from_sample.file_name == self.from_sample.file_name and to_sample.file_name == self.to_sample.file_name):
                return self.tvgnode.H;
            elif(from_sample.file_name == self.to_sample.file_name and to_sample.file_name == self.from_sample.file_name):
                return np.linalg.inv(self.tvgnode.H);
        else:
            return None;

    def E(self, from_sample, to_sample):
        if(from_sample.file_name == self.from_sample.file_name and to_sample.file_name == self.to_sample.file_name):
            return self.tvgnode.E;
        elif(from_sample.file_name == self.to_sample.file_name and to_sample.file_name == self.from_sample.file_name):
            return np.linalg.inv(self.tvgnode.E);

    def F(self, from_sample, to_sample):
        if(from_sample.file_name == self.from_sample.file_name and to_sample.file_name == self.to_sample.file_name):
            return self.tvgnode.F;
        elif(from_sample.file_name == self.to_sample.file_name and to_sample.file_name == self.from_sample.file_name):
            return np.linalg.inv(self.tvgnode.F);

    @property
    def _H(self):
        return self.tvgnode.H;

    @property
    def config(self):
        return self.tvgnode.config;

    @property
    def config_string(self):
        return self.tvgnode.config_string;

    @property
    def _F(self):
        return self.tvgnode.F;

    @property
    def _E(self):
        return self.tvgnode.E;

    @property
    def is_homography(self):
        return (self.config in [4, 5, 6]);




CONFIG_TO_STRING_MAP = {
        0: "UNDEFINED",
        1: "DEGENERATE", # Degenerate configuration (e.g., no overlap or not enough inliers).
        2: "CALIBRATED", # Essential matrix.
        3: "UNCALIBRATED", # Fundamental matrix.
        4: "PLANAR", # Homography, planar scene with baseline.
        5: "PANORAMIC", # Homography, pure rotation without baseline.
        6: "PLANAR_OR_PANORAMIC", # Homography, planar or panoramic.
        7: "WATERMARK", # Watermark, pure 2D translation in image borders.
        8: "MULTIPLE", # Multi-model configuration, i.e. the inlier matches result from multiple individual, non-degenerate configurations.
    }

STRING_TO_CONFIG_MAP = {}
for key in CONFIG_TO_STRING_MAP:
    STRING_TO_CONFIG_MAP[CONFIG_TO_STRING_MAP[key]] = key;

class COLMAPTwoViewGeometriesNode(DataNode):
    CONFIG_MAP = STRING_TO_CONFIG_MAP;



    @property
    def config_string(self):
        return CONFIG_TO_STRING_MAP[self.config];

    @property
    def image_ids(self):
        return pair_id_to_image_ids(self.get_label('pair_id'));

    @property
    def pair_id(self):
        return self.get_label('pair_id');

    @property
    def H(self):
        H = self.get_label('H');
        if(isinstance(H, np.ndarray)):
            H.shape = 3, 3;
            return H;
        else:
            return None;

    @property
    def F(self):
        F = self.get_label('F');
        F.shape = 3, 3;
        return F;

    @property
    def E(self):
        E = self.get_label('E');
        E.shape = 3, 3;
        return E;

    @property
    def data(self):
        return self.get_label('data');

    @property
    def config(self):
        return self.get_label('config');


class CDBTVGeometries(CDBTable):
    DATANODE_CLASS = COLMAPTwoViewGeometriesNode;
    DATANODESET_FILE_EXTENSIONS = ['.cdbgeometries'];
    DEFAULT_INDEX_KEY = "pair_id"

    @staticmethod
    def CONFIG_IS_HOMOGRAPHY(config):
        return (config in [4, 5, 6]);

    # def get_edge_graph_dataframe(self):
    #     return pd.DataFrame(dict(
    #         source=self.nodes.pair_ids.map(lambda x: x[0]),
    #         target=self.nodes.pair_ids.map(lambda x: x[1])
    #     ))

    def get_nxgraph(self):
        return nx.Graph(self.get_edge_graph_dataframe().reset_index())

    @classmethod
    def from_dataframe_with_binary_info(cls, dataframe: pd.DataFrame):
        geometries = dataframe.copy();

        def n_matches(x):
            if (isinstance(x, np.ndarray)):
                return len(x);
            else:
                return 0

        # matches['data'] = matches['data'].map(lambda x: blob_to_array(x, np.uint32, (-1, 2)))
        geometries['F'] = geometries['F'].map(lambda x: blob_to_array(x, np.float64) if x else np.NaN)
        geometries['E'] = geometries['E'].map(lambda x: blob_to_array(x, np.float64) if x else np.NaN)
        geometries['H'] = geometries['H'].map(lambda x: blob_to_array(x, np.float64) if x else np.NaN)
        geometries['qvec'] = geometries['qvec'].map(lambda x: blob_to_array(x, np.float64) if x else np.NaN)
        geometries['tvec'] = geometries['tvec'].map(lambda x: blob_to_array(x, np.float64) if x else np.NaN)
        geometries['data'] = geometries['data'].map(lambda x: blob_to_array(x, np.float64) if x else np.NaN)
        geometries['pair_ids'] = geometries['pair_id'].map(lambda x: pair_id_to_image_ids(x))
        geometries['n_inliers'] = geometries['data'].map(lambda x: n_matches(x));
        return cls.from_dataframe(geometries.set_index(cls.DEFAULT_INDEX_KEY, drop=False));

    def get_edge_graph_dataframe(self):
        def getHDet(x):
            if (x is np.nan):
                return x;
            else:
                return np.linalg.det(np.array(x).reshape(3,3));

        return pd.DataFrame(dict(
            source=self.nodes.pair_ids.map(lambda x: x[0]),
            target=self.nodes.pair_ids.map(lambda x: x[1]),
            config = self.nodes.config,
            traslation_norm = self.nodes.tvec.map(lambda x: np.linalg.norm(x)),
            # HDet = self.nodes.H
            HDet = self.nodes.H.map(getHDet),
        ))

    def _get_geometry_for_pair_id(self, pair_id):
        if (pair_id in self.nodes.index):
            return self[pair_id];
        else:
            return None;
