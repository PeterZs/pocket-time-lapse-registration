from ptlreg.apy.core import file_created_timestamp_string_from_path
from ptlreg.apydn import HasFilePathLabel
from ptlreg.apydn.datanode.DataNode import *
import pandera.pandas as pa

class FileDataNodePathDoesNotExistError(Exception):
    def __init__(self, filenode, message=None):
        self.filenode = filenode;
        self.message=message;
        super(FileDataNodePathDoesNotExistError, self).__init__(self.message)

class HasFileDataNode_Mixin(HasFilePathLabel):
    """
    Mixin class for data nodes that represent files.
    """
    DATANODE_SCHEMA = pa.DataFrameSchema({
        DataNodeConstants.FILE_PATH_KEY: pa.Column(str)
    });



    def __init__(self, path=None, root_path=None, **kwargs):
        """
        Initialize FileDataNode. This is a node that represents a file with a path.
        :param path: The path
        :param kwargs:
        """
        # print("HasFileDataNode_Mixin.__init__ called with path: {}, root_path: {}".format(path, root_path));
        super(HasFileDataNode_Mixin, self).__init__(path=path, root_path=root_path, **kwargs);
        self.set_node_id(self.file_path);

    def __str__(self):
        return '[{}]:{}'.format(self.__class__.__name__, self.file_path);
    def __repr__(self):
        return '[{}]:{}'.format(self.__class__.__name__, self.file_path);

    @classmethod
    def validate_file_data_node_path(cls, node, root_path=None):
        if(node.file_exists(from_root=root_path)):
            return True;
        else:
            raise FileDataNodePathDoesNotExistError(node, "File Does Not Exist!!!");

    @classmethod
    def file_data_node_from_path(cls, path, root_path=None, validate=True):
        newnode = cls(path=path, root_path=root_path);
        if(validate):
            cls.validate_file_data_node_path(newnode);
        return newnode;

    # <editor-fold desc="Property: 'created_timestamp'">
    @property
    def created_timestamp(self):
        created = self.get_label("created_timestamp");
        if(created is None):
            self.set_label("created_timestamp", file_created_timestamp_string_from_path(self.get_absolute_path()));
    # </editor-fold>





class FileDataNode(HasFileDataNode_Mixin, DataNode):
    pass;