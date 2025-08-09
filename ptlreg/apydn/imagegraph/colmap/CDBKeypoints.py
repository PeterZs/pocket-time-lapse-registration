
import pandas as pd
from .CDBTable import *
from ptlreg.apydn.datanode.DataNode import DataNode

class COLMAPKeypointsNode(DataNode):

    @property
    def image_id(self):
        return self.get_label('image_id');

    @property
    def rows(self):
        return self.get_label('rows');

    @property
    def cols(self):
        return self.get_label('cols');

    @property
    def data(self):
        return self.get_label('data');


    @property
    def point_locations(self):
        return self.data[:,:2];

class CDBKeypoints(CDBTable):
    DATANODE_CLASS = COLMAPKeypointsNode;
    DATANODESET_FILE_EXTENSIONS = ['.cdbkeypoints'];
    DEFAULT_INDEX_KEY = "image_id"
    @classmethod
    def from_dataframe_with_binary_info(cls, dataframe: pd.DataFrame):
        keypoints = dataframe.copy();
        def getData(row):
            return blob_to_array(row['data'], np.float32, (row['rows'], row['cols']))
        keypoints['data'] = keypoints.apply(lambda r: getData(r), axis=1)
        # keypoints['data'] = keypoints['data'].map(lambda x: blob_to_array(x, np.float32, (-1, 2)))
        return cls.from_dataframe(keypoints.set_index(cls.DEFAULT_INDEX_KEY, drop=False));

