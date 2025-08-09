from ptlreg.apy.core.filepath.HasFilePath import file_created_timestamp_string_from_path, get_file_creation_datetime
from ptlreg.apydn.datanode import *
from ptlreg.apydn.datanode.filedatanode import *



class HasFileDataNodeSet_Mixin(object):
    RELATIVE_FILE_PATH_KEY =HasFilePathLabel.FILE_PATH_KEY;
    # DEFAULT_INDEX_KEY = FILEDATANODESET_DEFULT_INDEX;
    SUBSET_CLASS = None;
    DATANODE_CLASS = FileDataNode;

    def get_absolute_path(self, from_root=None):
        root_path = from_root;
        if(root_path is None):
            root_path = self.root_path;
        if(root_path is not None):
            return os.path.abspath(os.path.join(self.root_path, self.file_path));
        else:
            return os.path.abspath(self.file_path);


    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.nodes.__str__());

    def __repr__(self):
        return self.nodes.__repr__();

    def init_nodes(self, index_key=None, columns=None):
        """
        Initialize the nodes in the set.
        :param index_key: The key to use as the index for the nodes.
        :param columns: The columns to use for the nodes.
        """
        if(index_key is None):
            index_key = DataNodeConstants.FILE_PATH_KEY
        super(HasFileDataNodeSet_Mixin, self).init_nodes(index_key=index_key, columns=[DataNodeConstants.FILE_PATH_KEY, DataNodeConstants.ROOT_PATH_KEY]);


    # <editor-fold desc="Property: 'root_path'">
    @property
    def root_path(self):
        return self.get_label(DataNodeConstants.ROOT_PATH_KEY);
        # return self.get_label(DataNodeConstants.ROOT_PATH_KEY);
    # </editor-fold>


    # def set_root_path(self, root_path):
    #     raise NotImplementedError;


    # def subset(self, *args, **kwargs):
    #     subset = super(HasFileDataNodeSet_Mixin, self).Subset(*args, **kwargs)
    #     subset._set_root_path(self.root_path)
    #     return self.__class__.GetSubsetClass()(*args, **kwargs)

    def calc_timestamps(self, force_use_metadata=False):
        """
        Calculate the timestamps for all file nodes in the set.
        This method should be called after the file paths are set.
        """
        self.calc_label_for_nodes(DataNodeConstants.CREATED_TIMESTAMP_KEY, lambda x: file_created_timestamp_string_from_path(os.path.join(x.root_path, x[DataNodeConstants.FILE_PATH_KEY]), force_use_metadata=force_use_metadata))

    def calc_created_datetime(self, force_use_metadata=False):
        """
        Calculate the timestamps for all file nodes in the set.
        This method should be called after the file paths are set.
        """
        self.calc_label_for_nodes(DataNodeConstants.CREATED_DATETIME_KEY, lambda x: get_file_creation_datetime(os.path.join(x.root_path, x[DataNodeConstants.FILE_PATH_KEY]), force_use_metadata=force_use_metadata))



class FileDataNodeSet(HasFileDataNodeSet_Mixin, DataNodeSet):
    pass;


