import pandas as pd
from .CDBTable import *
from ptlreg.apydn.datanode.DataNode import DataNode


class COLMAPCameraNode(DataNode):
    @property
    def params(self):
        return self.get_label('params');

    @property
    def model(self):
        return self.get_label('model');

    @property
    def params(self):
        return self.get_label('params');

    @property
    def camera_id(self):
        return self.get_label('camera_id');

class CDBCameras(CDBTable):
    DATANODE_CLASS = COLMAPCameraNode;
    DATANODESET_FILE_EXTENSIONS = ['.cdbcameras'];
    DEFAULT_INDEX_KEY = "camera_id";
    @classmethod
    def from_dataframe_with_binary_info(cls, dataframe: pd.DataFrame):
        cameras = dataframe.copy();
        cameras['params'] = cameras['params'].map(lambda x: blob_to_array(x, np.float64))
        cameras['model'] = cameras['model'].map(lambda x: CAMERA_MODEL_IDS[x].model_name)
        return cls.from_dataframe(cameras.set_index(cls.DEFAULT_INDEX_KEY, drop=False));



