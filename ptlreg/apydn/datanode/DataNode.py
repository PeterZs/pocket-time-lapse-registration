import uuid
import pandera.pandas as pa
import six
from ptlreg.apydn.HasDataLabels import *
from ptlreg.apydn.DataNodeConstants import DataNodeConstants
DataNodeClasses = {};


def register_node_class(cls_to_register):
    assert(hasattr(cls_to_register, 'data_node_type_name'));
    if(DataNodeClasses.get(cls_to_register.data_node_type_name()) is not None):
        print("WARNING!!!!!:data_node_type_name {} already used by {}\nold class: {}\nnew class: {}!".format(
        cls_to_register.data_node_type_name(), cls_to_register.__name__, DataNodeClasses.get(cls_to_register.data_node_type_name()), cls_to_register))
    DataNodeClasses[cls_to_register.data_node_type_name()] = cls_to_register;

class DataNodeMeta(type):
    """
    Meta-class that is used to register DataNode classes.
    """
    def __new__(cls, *args, **kwargs):
        newtype = super(DataNodeMeta, cls).__new__(cls, *args, **kwargs);
        register_node_class(newtype);
        return newtype;

    def __init__(cls, *args, **kwargs):
        super(DataNodeMeta, cls).__init__(*args, **kwargs);


    def __call__(cls, *args, **kwargs):
        supercall = super(DataNodeMeta, cls).__call__(*args, **kwargs);
        return supercall;




@six.add_metaclass(DataNodeMeta)
class DataNode(HasDataLabels):
    '''
        Mixin for nodes. IMPORTANT!!!: There should be no properties stored outside of data_labels!!!
    '''
    DATANODE_SCHEMA = pa.DataFrameSchema();
    NODE_ID_KEY = DataNodeConstants.NODE_ID_KEY
    DEFAULT_INDEX_KEY = DataNodeConstants.NODE_ID_KEY

    @classmethod
    def generate_node_id(cls, nchar=None):
        """
        Generates a random node id.
        :return: str
        """
        if(nchar is None):
            return str(uuid.uuid4());
        else:
            return str(uuid.uuid4())[:nchar];

    @property
    def help(self):
        print("{} with node_id: {}\nobj.data_labels is a pandas series, access with get_label(key) and set_label(key, value)".format(self.__class__.name, self.node_id))

    @classmethod
    def validate_data_labels_for_instance(cls, data_labels):
        """
        Uses pandera to validate data labels according to the current class schema.
        This is done by converting the series to a dataframe and passing it to the schema validate function.
        :param data_labels:
        :return:
        """
        return cls.DATANODE_SCHEMA.validate(pd.DataFrame([data_labels]));

    @classmethod
    def validate_data_node(cls, node):
        """
        Validates a data node against the schema of the current class.
        :param node:
        :return:
        """
        return cls.DATANODE_SCHEMA.validate(node.get_data_labels_dataframe());

    @classmethod
    def data_node_type_name(cls):
        """
        If you have class name collisions (e.g. same name different module) you can specify the name to use for this
        class type here. Ideally, you should use the longer version of the class name, e.g., 'apy.AObject'.
        :return:
        """
        return cls.__name__;

    def __init__(self, *args, **kwargs):
        # print("DataNode.__init__ called with args: {}, kwargs: {}".format(args, kwargs));
        super(DataNode, self).__init__(*args, **kwargs);

    @classmethod
    def create_with_id(cls, id, *args, **kwargs):
        rval = cls(*args, **kwargs);
        rval.set_node_id(id);
        return rval;

    @classmethod
    def new_with_random_id(cls, nchar=None, *args, **kwargs):
        rval = cls(*args, **kwargs);
        rval.set_node_id(cls.generate_node_id(nchar=nchar));
        return rval;

    @classmethod
    def from_series(cls, series):
        newNode = cls();
        newNode._set_data_labels(series);
        return newNode;

    @classmethod
    def from_csv(cls, path):
        """
        Creates a DataNode from a CSV file.
        :param path: Path to the CSV file.
        :return: DataNode
        """
        df = pd.read_csv(path, index_col=0).transpose();
        return cls.from_series(df.iloc[0]);

    def set_node_id(self, id):
        self.set_label(self.__class__.NODE_ID_KEY, id);


    def save_data_labels_to_csv(self, path):
        print("Saving DataNOde data labels to CSV: {}".format(path));
        self.data_labels.to_csv(path);

    def load_data_labels_from_csv(self, path)->None:
        df = pd.read_csv(path, index_col=0).transpose();
        # pd.read_csv(csvpath, index_col=0).transpose().iloc[0]
        self._set_data_labels(df.iloc[0]);


    def __getitem__(self, key):
        return self.data_labels.__getitem__(key)

    def __setitem__(self, key, newvalue):
        return self.data_labels.__setitem__(key, newvalue);


    # <editor-fold desc="Property: 'name'">
    @property
    def name(self)->str:
        return self.get_label("name");

    @name.setter
    def name(self, value):
        self.set_label("name", value);
    # </editor-fold>

    # <editor-fold desc="Property: 'node_id'">
    @property
    def node_id(self):
        return self.get_label(self.__class__.NODE_ID_KEY);
    # </editor-fold>

    def get_series(self)-> pd.Series:
        return self.data_labels;