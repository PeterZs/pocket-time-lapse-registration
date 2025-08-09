from ptlreg.apy.amedia.signals.SignalMixin import *
from ptlreg.apy.core import AObjectMeta
from ptlreg.apy.core import AObject

import ptlreg.apy.afileui as ui
import shutil
import os
from ptlreg.apy.core.filepath.FilePath import *
import ptlreg.apy.utils
from functools import wraps
import six

MEDIA_EXTENSIONS_DICT = {}
_MEDIA_OBJECT_TYPES={}

def register_media_object_class(cls_to_register):
    assert(hasattr(cls_to_register, 'MEDIA_FILE_EXTENSIONS')), "Subclass {} of MediaObject must implement MEDIA_FILE_EXTENSIONS().".format(cls_to_register.aobject_type_name());
    exts = cls_to_register.MEDIA_FILE_EXTENSIONS();
    for e in exts:
        existing_class = MEDIA_EXTENSIONS_DICT.get(e);
        if(existing_class is not None):
            # print(cls_to_register)
            # print(existing_class)
            assert(issubclass(cls_to_register, existing_class)), "Cannot register extension {} to both {} and {} classes".format(e, cls_to_register, existing_class);
        MEDIA_EXTENSIONS_DICT[e] = cls_to_register;
        # print("Registered {} to {}".format(e, cls_to_register))

    old_class = _MEDIA_OBJECT_TYPES.get(cls_to_register.__name__);

    #TODO: This should probably be put back if you want to be safe
    # assert(old_class is None), "tried to register MediaObject class {} twice:\nold: {}\nnew: {}".format(cls_to_register.__name__, old_class, cls_to_register);

    _MEDIA_OBJECT_TYPES[cls_to_register.__name__]=cls_to_register;

class MediaObjectMeta(AObjectMeta):
    def __new__(cls, *args, **kwargs):
        newtype = super(MediaObjectMeta, cls).__new__(cls, *args, **kwargs);
        register_media_object_class(newtype);
        return newtype;


    def __init__(cls, *args, **kwargs):
        return super(MediaObjectMeta, cls).__init__(*args, **kwargs);

    def __call__(cls, *args, **kwargs):
        supercall = super(MediaObjectMeta, cls).__call__(*args, **kwargs);
        return supercall;




# class MediaObject(SavesFeatures, HasFilePath):
#     __metaclass__ = MediaObjectMeta;
# _SegmentClass = None;

class MediaObject(six.with_metaclass(MediaObjectMeta, SignalMixin, AObject)):

    @staticmethod
    def MEDIA_FILE_EXTENSIONS():
        """
        MediaObject subclasses must implement MEDIA_FILE_EXTENSIONS to specify what extensions they load.
        Should return a list of file extensions, e.g., ['.mp4', '.mp3', '.txt', '.html'].
        :return:
        """
        return [];

    @staticmethod
    def get_media_object_type_dict():
        return _MEDIA_OBJECT_TYPES;

    @classmethod
    def default_file_name(cls):
        return cls.aobject_type_name()+cls.default_file_extension();

    @classmethod
    def default_file_extension(cls):
        return cls.MEDIA_FILE_EXTENSIONS()[0];

    @property
    def media_type_name(self):
        return self.aobject_type_name();

    def __init__(self, path=None, name=None, manager = None, **kwargs):
        """

        :param path:
        :param kwargs:
        """
        self._name = None;
        self._manager = None;
        super(MediaObject, self).__init__(path=path, **kwargs);

        if (manager is not None):
            self.set_media_manager(manager);
            # self.manager = manager;
        if (name is not None):
            self.name = name;

    @staticmethod
    def open(path=None, initial_path = None, **kwargs):
        if(path is None):
            path = ui.GetFilePath(initial_path=initial_path);
        return MediaObject.load_media_object(path=path, **kwargs);

    @staticmethod
    def load_media_object(path, **kwargs):
        media_path = FilePath(path=path);
        media_class = MEDIA_EXTENSIONS_DICT.get(media_path.file_ext, MediaObject);
        return media_class(path, **kwargs);

    @staticmethod
    def create_from_dictionary(d, load_file=True):
        s_class = AObject.class_from_dictionary(d);
        inst = s_class();
        inst.init_from_dictionary(d=d, load_file=load_file);
        return inst;

    def load_media_from_file(self, *args, **kwargs):
        """
        This should be implemented in subclasses
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError;



    # <editor-fold desc="Property: 'source_metadata'">
    @property
    def source_metadata(self):
        return self._source_metadata;
    @source_metadata.setter
    def source_metadata(self, value):
        self._source_metadata = value;

    @property
    def _source_metadata(self):
        return self.get_info('source_metadata');
    @_source_metadata.setter
    def _source_metadata(self, value):
        self.set_info('source_metadata', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'version_info'">
    @property
    def version_info(self):
        """
        This indicates things like whether the video is a downsampled version of the original
        :return:
        """
        return self._version_info;
    @property
    def _version_info(self):
        return self.get_info("version_info");
    @version_info.setter
    def version_info(self, value):
        self._version_info=value;
    @_version_info.setter
    def _version_info(self, value):
        self.set_info('version_info', value);
    # </editor-fold>


    @property
    def manager(self):
        return self.feature_manager;
    @manager.setter
    def manager(self, value):
        self.set_media_manager(value);

    # <editor-fold desc="Property: '_manager_info'">
    @property
    def _manager_info(self):
        return self.get_info("manager_info");
    @_manager_info.setter
    def _manager_info(self, value):
        self.set_info('manager_info', value);
    # </editor-fold>

    # @property
    # def _clip_manager(self):
    #     if((self.manager is not None) and hasattr(self.manager,'getManagedClipFor')):
    #         return self.manager;

    def set_media_manager(self, manager):
        self._manager_info={};
        if(manager is not None):
            if(hasattr(manager, 'file_path')):
                self._manager_info['file_path']=manager.test_file_path;
            if(hasattr(manager, 'node_id')):
                self._manager_info['node_id']=manager.node_id;
        self._setFeatureManager(manager);

    def init_from_aobject(self, fromobject, share_data = False):
        super(MediaObject, self).init_from_aobject(fromobject);
        # self.manager = fromobject.manager;
        if(share_data):
            self.set_media_manager(fromobject.manager);
        self.name = fromobject.name;

    @property
    def name(self):
        # if(self._name is None):
        #     return self.file_name_base;
        return self._name;

    @name.setter
    def name(self, n):
        self._set_name(name=n);
    def _set_name(self, name):
        self._name = name;


    # # <editor-fold desc="Property: 'node_label'">
    # @property
    # def _node_label(self):
    #     return self.get_info("node_label");
    # @_node_label.setter
    # def _node_label(self, value):
    #     self.set_info('node_label', value);
    # # </editor-fold>


    def copy_media_file_to(self, dst):
        shutil.copy2(self.get_file_path(), dst);

    def to_dictionary(self):
        d = super(MediaObject, self).to_dictionary();
        d['name'] = self.name;
        # Also add for easy media_info reference by MediaNodes
        d['media_type_name']=self.media_type_name;
        return d;

    def init_from_dictionary(self, d):
        super(MediaObject, self).init_from_dictionary(d);
        self.name = d['name'];

    def _get_temp_dir(self):
        if(self.manager is not None):
            # return self.manager.get_dir('temp', absolute_path=True);
            return self.manager.get_temp_dir();
        else:
            default_temp_dir = os.path.join (ptlreg.apy.utils.GetTempDir(), 'temp_', self.__class__.__name__) + os.sep;
            ptlreg.apy.utils.make_sure_dir_exists(default_temp_dir);
            return default_temp_dir;

    def _get_temp_file_path(self, final_path=None, temp_dir=None):
        if(temp_dir is None):
            temp_dir = self._get_temp_dir();
        return ptlreg.apy.utils.get_temp_file_path(final_file_path=final_path, temp_dir_path=temp_dir);

    # def GetManagedClip(self, **kwargs):
    #     return self.manager.getManagedClipFor(self, **kwargs);

    # def BounceClipTo(self, path, **kwargs):
    #     clip = self.GetClip(share_data = True, **kwargs);
    #     clip.write_to_file(output_path=path, **kwargs)
    #     # return MediaObject.LoadMediaObject(path);

    # @property
    # def media_file_path(self):
    #     return self.file_path;

    def write_to_file(self, output_path=None, **kwargs):
        raise NotImplementedError;

    def _bounce_to_path(self, output_path, write_features=True, features_dir=None, **kwargs):
        """

        :param output_path:
        :param write_features:
        :param features_dir:
        :param kwargs:
        :return:
        """
        self.write_to_file(output_path=output_path, **kwargs);
        if(features_dir is None):
            features_dir = os.path.join(FilePath(output_path).get_directory_path(), 'Features');
        if(write_features):
            self._bounce_features_to_dir(features_dir=features_dir);

    def _bounce_features_to_dir(self, features_dir):
        self._saveFeature('each', output_dir=os.path.join(features_dir, self.__class__.__name__));

    def load_features(self, features_to_load=None, **kwargs):
        if(features_to_load is None):
            features_to_load = 'each';
        return super(MediaObject, self).load_features(features_to_load=features_to_load, **kwargs);

    @property
    def mobject_class(self):
        return type(self);

    @property
    def duration(self):
        return None;
    @property
    def width(self):
        return None;
    @property
    def height(self):
        return None;

    ##################//--Hashable--\\##################
    # <editor-fold desc="Hashable">
    def get_file_and_clip_hashable(self):
        if(self.file_path is not None):
            source_hash = self._md5;
        else:
            source_hash = hash(self._samples);
        # return (source_hash,self.getClipRegion());
        raise NotImplementedError;
    # </editor-fold>
    ##################\\--Hashable--//##################

    # def getMediaRef(self):
    #     d = self.to_dictionary();
    #     d['media_object_path'] = self.file_path;
    #     loading_args = self._getLoadingArgs();
    #     if(loading_args is not None):
    #         d['loading_args']=loading_args;
    #     # if(hasattr(self, 'sampling_rate')):
    #     #     d['sampling_rate'] = self.sampling_rate;
    #     return d;

def MediaObjectMethod(media_type=None):
    """
    Decorator to add a method to MediaObject
    :param media_type:
    :return:
    """
    MEDIATYPE = MediaObject;
    if(media_type is not None):
        MEDIATYPE=MediaObject.get_media_object_type_dict()[media_type];
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        setattr(MEDIATYPE, func.__name__, wrapper)
        return func # returning func means func can still be used normally
    return decorator

def MediaObjectStaticMethod(media_type=None):
    MEDIATYPE = MediaObject;
    if(media_type is not None):
        MEDIATYPE=MediaObject.get_media_object_type_dict()[media_type];
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs);
        setattr(MEDIATYPE, func.__name__, staticmethod(wrapper))
        return func # returning func means func can still be used normally
    return decorator

def MediaObjectClassMethod(media_type=None):
    MEDIATYPE = MediaObject;
    if(media_type is not None):
        MEDIATYPE=MediaObject.get_media_object_type_dict()[media_type];
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs);
        setattr(MEDIATYPE, func.__name__, classmethod(wrapper))
        return func # returning func means func can still be used normally
    return decorator



# def AddIndex(cls):
#     def decorator(func):
#         @wraps(func)
#         def get_wrapper(self, *args, **kwargs):
#             # this should be the property get
#             return func(*args, **kwargs)
#         def set_wrapper(self, *args, **kwargs):
#             # this should be the property set
#             return func(*args, **kwargs)
#         def del_wrapper(self, *args, **kwargs):
#             # this should be the property del
#             return func(*args, **kwargs)
#         setattr(cls, func.__name__, property(get_wrapper, set_wrapper, del_wrapper, "I'm the 'x' property."))
#         # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
#         return func  # returning func means func can still be used normally
#     return decorator
