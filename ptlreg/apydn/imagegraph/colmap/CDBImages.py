import pandas as pd
from .CDBTable import *
from ptlreg.apydn.datanode.DataNode import DataNode


class COLMAPImageNode(DataNode):
    @property
    def image_id(self):
        return self.get_label('image_id');
    @property
    def name(self):
        return self.get_label('name');
    @property
    def camera_id(self):
        return self.get_label('camera_id');

class CDBImages(CDBTable):
    DATANODE_CLASS = COLMAPImageNode;
    DATANODESET_FILE_EXTENSIONS = ['.cdbimages'];
    DEFAULT_INDEX_KEY = "image_id";
    # def __init__(self):

    @classmethod
    def from_dataframe_with_binary_info(cls, dataframe: pd.DataFrame):
        images = dataframe.copy();
        return cls.from_dataframe(images.set_index(cls.DEFAULT_INDEX_KEY, drop=False));



