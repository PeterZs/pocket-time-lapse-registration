"""

"""
# from visbeat.VisBeatImports import *
from .dicts import FuncDict
import os
import os.path
from ptlreg.apy.aops.AOps import AOpType
# from SavesToJSON import SavesToJSON
import ptlreg.apy.utils as apyutils
from functools import wraps


class FeatureFunction(AOpType):
    """
    Decorator class. use @FeatureFunction(feature_name) decorator to register a function with a feature name.
    """
    def __init__(self, feature_name, save_result = True, clip_to_region = False, inherited=None, op_version = 0, hashable_args=None, **decorator_args):
        """

        :param feature_name:
        :param save_result:
        :param clip_to_region:
        :param inherited: if False, then not inherited. If None, then it depends on whether it clips to region and a clip is inheriting
        :param op_version:
        :param hashable_args:
        :param decorator_args:
        """
        # print(feature_name)
        assert((feature_name.lower() != 'all') and (feature_name.lower() != 'each')), 'Cannot use feature name {}; it is reserved'.format(feature_name);
        super(FeatureFunction, self).__init__(nickname=feature_name,
                                              op_version=op_version,
                                              hashable_args=None,
                                              **decorator_args);
        self.feature_name = feature_name;
        self.save_result = save_result;
        self.clip_to_region = clip_to_region;
        self.inherited = inherited;

        # self.decorator_args = decorator_args;
        # self.feature_version = feature_version;

    def decorate_function(self, func):
        self._check_for_reserved_args('force_recompute');
        if(not self.save_result):
            @wraps(func)
            def decorated(*args, **kwargs):
                # assert ('force_recompute' not in inspect.getargspec(func)[0]), "The argument 'force_recompute' is reserved for functions decorated with FeatureFunction. Use a different argument name. for function with feature_name '{}'.".format(self.feature_name);
                obj = args[0];
                fkwargs = dict(kwargs);
                force_recompute = fkwargs.pop('force_recompute', None);
                rval = func(*args, **fkwargs);
                if(self.clip_to_region):
                    assert(hasattr(rval, 'clip_bounds')), "Only clippable objects can be returned by features marked with clip_to_region=True. feature {} result {} does not have self.clip_bounds".format(self.feature_name, rval);
                return rval;
        else:
            @wraps(func)
            def decorated(*args, **kwargs):
                # assert('force_recompute' not in inspect.getargspec(func)[0]), "The argument 'force_recompute' is reserved for functions decorated with FeatureFunction. Use a different argument name. for function with feature_name '{}'.".format(self.feature_name);
                obj = args[0];
                fkwargs = dict(kwargs);
                force_recompute = fkwargs.pop('force_recompute', None);
                if ((not obj.has_feature(self.feature_name)) or force_recompute):
                    rval = func(*args, **fkwargs);
                    obj.set_feature(name=self.feature_name, value=rval, **fkwargs);
                rval = obj.get_feature(self.feature_name);
                if (self.clip_to_region):
                    assert (hasattr(rval,'clip_bounds')), "Only clippable objects can be returned by features marked with clip_to_region=True. feature {} result {} does not have self.clip_bounds".format(self.feature_name, rval);
                return rval;
        return decorated;

    def set_op_info(self, decorated):
        super(FeatureFunction, self).set_op_info(decorated);
        self.add_op_info_to(decorated,
                            saves_result = self.save_result,
                            clip_to_region = self.clip_to_region,
                            inherited = self.inherited);
        decorated.feature_name = self.feature_name;
        decorated.clip_to_region = self.clip_to_region;
        # decorated.feature_function_info = ;
        # decorated.is_stale = True;
        return decorated;

class ManagesFeatures(object):
    """
    At a minimum you can implement getFeaturesDir() and it will use said directory with the default SaveFeatures functions
    Customizeable behavior beyond that.
    """
    ##################//--Setting Save/Load/Clear Functions--\\##################
    # <editor-fold desc="Setting Save/Load/Clear Functions">
    def _get_feature_save_function(self):
        def fsaveFeatures(obj, features_to_save = 'each', overwrite = True, ** kwargs):
            return obj.manager._save_features_for(obj=obj, features_to_save=features_to_save, overwrite=overwrite, **kwargs);
        return fsaveFeatures;
    def _get_feature_load_function(self):
        def floadFeatures(obj, features_to_load=None, **kwargs):
            return obj.manager._load_features_for(obj=obj, features_to_load=features_to_load, **kwargs);
        return floadFeatures;
    def _get_feature_clear_function(self):
        def fclearFeatures(obj, features_to_clear=None, **kwargs):
            return obj.manager._clearFeaturesFor(obj, features_to_clear=features_to_clear, **kwargs);
        return fclearFeatures;
    # </editor-fold>
    ##################\\--Setting Save/Load/Clear Functions--//##################

    def get_features_dir(self, **kwargs):
        """
        Gets the directory where features are to be saved
        :param kwargs:
        :return:
        """
        return None;

    def _get_feature_dir_for(self, obj, feature_name=None):
        """
        The default is that features for different classes go to different subdirectories of the Features Root Dir.
        :param obj:
        :param feature_name:
        :return:
        """
        if (self.get_features_dir() is not None):
            return os.path.join(self.get_features_dir(), type(obj).__name__ + os.sep);

    def _get_feature_file_path_for(self, obj, feature_name, features_dir=None):
        if (features_dir is None):
            features_dir = self._get_feature_dir_for(obj, feature_name=feature_name);
        return os.path.join(features_dir, obj._get_feature_file_name(feature_name));

    def _get_features_root_for(self, obj):
        """
        Return None if manager will handle feature file locations
        :param obj:
        :return:
        """
        return None;


    ##################//--Default Save/Load/Clear Functions (just use MediaObject version)--\\##################
    # <editor-fold desc="Default Save/Load/Clear Functions (just use MediaObject version)">



    def _save_features_for(self, obj, features_to_save=None, output_dir=None, overwrite=True):
        """
        ManagesFeatures._saveFeaturesFor()
        :param obj:
        :param features_to_save:
        :param output_dir:
        :param overwrite:
        :return:
        """
        if (features_to_save is None):
            return;
        if (not isinstance(features_to_save, list)):
            features_to_save = [features_to_save];
        success = True;
        for f in features_to_save:
            didsave = self._save_feature_for(obj, feature_name=f,
                                             output_dir=output_dir,
                                             overwrite=overwrite);
            success = didsave and success;
        return success;

    def _save_feature_for(self, obj, feature_name, output_dir=None, overwrite=True):
        """
        ManagesFeatures._saveFeatureFor()
        :param feature_name:
        :param output_dir:
        :param overwrite:
        :return: True if saving is successful
        """
        feature_output_dir = output_dir;
        if (feature_output_dir is None):
            feature_output_dir = self._get_feature_dir_for(obj, feature_name=feature_name);
        if (feature_output_dir is None):
            assert (False), "Cannot save -- no features directory specified"

        if (feature_name == 'each'):
            # Note here that you want to use output_dir, because if it is None you may still use different dirs for different features
            return self._save_features_for(obj, features_to_save=obj.get_features_list(),
                                           output_dir=output_dir,
                                           overwrite=overwrite);
        apyutils.make_sure_dir_exists(feature_output_dir);
        # assert (os.path.isdir(feature_output_dir)), "Directory {} does not exist.".format(feature_output_dir);

        opath = self._get_feature_file_path_for(obj, feature_name=feature_name, features_dir=feature_output_dir);
        apyutils.make_sure_dir_exists(opath);
        if (not os.path.isfile(opath) or overwrite):
            if (feature_name == 'all'):
                return obj._save_all_features_to_path(path=opath);
            else:
                vfeature = obj.get_feature(name=feature_name, force_recompute=False);
                if (vfeature is not None):
                    return obj._save_feature_to_path(name=feature_name, path = opath);

    def _load_features_for(self, obj, features_to_load=None, features_dir = None):
        # print("loadFeaturesFor {}, {}".format(features_to_load, features_dir))
        if (features_to_load is None):
            return;
        if (not isinstance(features_to_load, list)):
            features_to_load = [features_to_load];
        for f in features_to_load:
            self._load_feature_for(obj, feature_name=f, features_dir=features_dir);

    def _load_feature_for(self, obj, feature_name, features_dir = None):
        feature_load_dir = features_dir;
        if(feature_load_dir is None):
            # feature_load_dir = self._getFeatureDirFor(obj, feature_name=feature_name);
            feature_load_dir = obj.features_dir;
        # if(feature_load_dir is None):
            # is this one necessary?
            # feature_load_dir = self._getFeatureDirFor(obj, feature_name=feature_name);
        

        if (feature_name == 'each'):
            if(feature_load_dir is not None):
                # print("loading each feature from {}".format(feature_load_dir));
                return obj._load_each_feature_from_dir(features_dir=feature_load_dir);
            else:
                return self._load_features_for(obj,
                                               features_to_load=obj.get_features_list(),
                                               features_dir=features_dir);

        ipath = self._get_feature_file_path_for(obj, feature_name=feature_name, features_dir=feature_load_dir);
        if (os.path.isfile(ipath)):
            print("Loading {} from {}".format(feature_name, ipath))
            # print(ipath)
            if (feature_name == 'all'):
                obj._load_all_features_from_path(path=ipath);
            else:
                obj._load_feature_from_path(path=ipath);




    # </editor-fold>
    ##################\\--Default Save/Load/Clear Functions (just use MediaObject version)--//##################


class _SavesFeatures(object):
    """
    Base class with property FeatureFuncs. FeatureFuncs looks through the callable attributes of the object
    for functions with a feature_name attribute, and adds them to a
    """
    # @property
    def _feature_funcs(self):
        feature_func_dict = {};
        for attr in dir(self.__class__):
            obj = getattr(self.__class__, attr);
            if(callable(obj) and hasattr(obj, "feature_name")):
                assert(feature_func_dict.get(obj.feature_name) is None), "Tried to register multiple functions to feature name '{}'".format(obj.feature_name);
                feature_func_dict[obj.feature_name]=obj;
        return feature_func_dict;

class SavesFeatures(_SavesFeatures):
    """
    calculated features are stored in self.features, but saved to the directory given by _freatures_root
    So, this class alone is mostly for use as mixin. Adds SavesFeatures functions, but should be mixed in with something that will define _features_root
    """

    def __init__(self, **kwargs):
        self.features = FuncDict(owner=self, name='features');
        self._save_features_func = None;
        self._load_features_func = None;
        self._clear_features_func = None;
        self._feature_manager = None;
        self._features_root = None;
        self.features.functions.update(self._feature_funcs());
        super(SavesFeatures, self).__init__(**kwargs);
    # @classmethod
    # def FeatureClassName(cls):
    #     return cls.__name__;

    @property
    def feature_manager(self):
        return self._feature_manager;
    @feature_manager.setter
    def feature_manager(self, value):
        self._set_feature_manager(value);

    def _set_feature_manager(self, manager):
        """
        If manager._getFeaturesRootFor(self) is None,
        :param manager:
        :return:
        """
        self._feature_manager = manager;
        if (self.feature_manager is not None):
            features_root = self.feature_manager._get_features_root_for(self);  # this is part of ManagesMediaObject
            if(features_root is None):
                self._features_root = None;
            else:
                self.features_root = features_root;
            self._set_feature_save_function(self.feature_manager._get_feature_save_function());
            self._set_feature_load_function(self.feature_manager._get_feature_load_function());
            self._set_feature_clear_function(self.feature_manager._get_feature_clear_function());

    def _set_feature_save_function(self, save_func):
        self._save_features_func = save_func;

    def _set_feature_load_function(self, load_func):
        self._load_features_func = load_func;

    def _set_feature_clear_function(self, clear_func):
        self._clear_features_func = clear_func;

    def get_feature(self, name, force_recompute=False, **kwargs):
        fval = self.features.get_value(name=name, force_recompute=force_recompute, **kwargs)
        ffunc = self.features.get_function(name=name);
        if(ffunc is not None and ffunc.clip_to_region and hasattr(fval, 'clip_bounds')):
            # if ffunc exists, its set as a feature that clips to segment regions, and the evaluated feature has a clip
            fval.clip_bounds = self.clip_bounds;
        return fval;

    def get_feature_params(self, name):
        return self.features.get_params(name=name);

    def set_feature(self, name, value, **kwargs):
        """Can add the params as keyword arguments and they will be stored as params"""
        return self.features.set_value(name=name, value=value, **kwargs);
        # return self.features._setEntry(name=name, d=dict(name=name, value=value, params=params));

    def has_feature(self, name):
        """Just checks to see if it's there."""
        return self.features.has_entry(name=name);

    def get_feature_function(self, feature_name):
        return self.features.get_function(name=feature_name);

    def get_feature_function_info(self, feature_name):
        return self.features.get_function(name=feature_name).op_info;

    def get_features_list(self):
        return self.features.get_key_list();

    def get_registered_feature_names(self):
        return self.features.get_function_list();

    def set_feature_info(self, feature_name, **kwargs):
        self.features.set_entry_info(entry_name=feature_name, **kwargs);
        return;

    def get_feature_info(self, feature_name, info_label=None):
        return self.features.get_entry_info(entry_name=feature_name, info_label=info_label);

    @property
    def features_dir(self):
        # print("features root: {}".format(self._features_root));
        # print("type name: {}".format(type(self).__name__));
        if(self._features_root is not None):
            return os.path.join(self._features_root, type(self).__name__+os.sep);
    @property
    def features_root(self):
        return self._features_root
    @features_root.setter
    def features_root(self, path):
        self.set_features_root(path);

    def create_and_set_features_directory(self, features_dir=None):
        if(features_dir is None):
            features_dir = os.path.join(self.get_directory_path(), self.file_name_base+"_Features"+os.sep);
        apyutils.make_sure_dir_exists(features_dir)
        self.set_features_root(features_dir);

    def set_features_root(self, path):
        assert(os.path.isdir(path)), "{} is not a valid directory to save features.".format(path)
        self._features_root = path;
        # Make sure features_dir exists
        apyutils.make_sure_dir_exists(os.path.join(self._features_root, type(self).__name__ + os.sep));

    ##################//--Save / Load / Clear--\\##################
    # <editor-fold desc="Save / Load / Clear">

    def save_features(self, features_to_save='each', overwrite=True, **kwargs):
        if (self._save_features_func is not None):
            self._save_features_func(self, features_to_save=features_to_save, overwrite=overwrite, **kwargs);
        else:
            self._save_features(features_to_save=features_to_save, output_dir=self.features_dir, overwrite=overwrite);

    def load_features(self, features_to_load=None, **kwargs):
        if (self._load_features_func is not None):
            self._load_features_func(self, features_to_load=features_to_load, **kwargs);
        elif(self.features_dir is not None):
            self._load_features(features_to_load=features_to_load, **kwargs);
        else:
            apyutils.AWARN("LOAD FUNCTION HAS NOT BEEN PROVIDED FOR {} INSTANCE".format(type(self)));

    def clear_features(self, features_to_clear=None, **kwargs):
        if(self._clear_features_func is not None):
            self._clear_features_func(self, features_to_clear=features_to_clear, **kwargs);
        else:
            AWARN("CLEAR FEATURE FILES FUNCTION HAS NOT BEEN PROVIDED FOR {} INSTANCE".format(type(self)));

    # </editor-fold>
    ##################\\--Save / Load / Clear--//##################


    ##################//--Default Save/Load Functions (when simple feature directory is used)--\\##################
    # <editor-fold desc="Default Save/Load Functions (when simple feature directory is used)">

    def _save_features(self, features_to_save=None, output_dir=None, overwrite=True):
        """

        :param features_to_save:
        :param output_dir:
        :param overwrite:
        :return:
        """
        if(output_dir is None):
            output_dir = self.features_dir;
        if(output_dir is None):
            assert(False), "Cannot save -- no features directory specified"

        if (features_to_save is None):
            return;
        if (not isinstance(features_to_save, list)):
            features_to_save = [features_to_save];
        success = True;
        for f in features_to_save:
            didsave = self._save_feature(feature_name=f,
                                         output_dir=output_dir,
                                         overwrite=overwrite);
            success = didsave and success;
        return success;

    def _save_feature(self, feature_name, output_dir=None, overwrite=True):
        """

        :param feature_name:
        :param output_dir:
        :param overwrite:
        :return: True if saving is successful
        """
        if (output_dir is None):
            output_dir = self.features_dir;
        if (output_dir is None):
            assert (False), "Cannot save -- no features directory specified"

        if (feature_name == 'each'):
            return self._save_features(features_to_save=self.get_features_list(),
                                       output_dir=output_dir, overwrite=overwrite);

        assert(os.path.isdir(output_dir)), "Directory {} does not exist.".format(output_dir);

        opath = self._get_feature_file_path(feature_name=feature_name, features_dir=output_dir);
        apyutils.make_sure_dir_exists(opath);
        if (not os.path.isfile(opath) or overwrite):
            if (feature_name == 'all'):
                return self._save_all_features_to_path(path=opath);
            else:
                vfeature = self.get_feature(name=feature_name, force_recompute=False);
                if (vfeature is not None):
                    return self._save_feature_to_path(name=feature_name, path = opath);

    def _load_features(self, features_to_load=None, features_dir = None):
        if (features_to_load is None):
            return;
        if (not isinstance(features_to_load, list)):
            features_to_load = [features_to_load];
        for f in features_to_load:
            self._load_feature(feature_name=f, features_dir=features_dir);

    def _load_feature(self, feature_name, features_dir = None):
        if(features_dir is None):
            features_dir = self.features_dir;
        if (feature_name == 'each'):
            return self._load_each_feature_from_dir(features_dir=features_dir);

        ipath = self._get_feature_file_path(feature_name=feature_name, features_dir=features_dir);
        if (os.path.isfile(ipath)):
            # print(ipath)
            if (feature_name == 'all'):
                self._load_all_features_from_path(path=ipath);
            else:
                self._load_feature_from_path(path=ipath);

    def _get_feature_file_name(self, feature_name):
        return feature_name+'.pkl';

    def _get_feature_file_path(self, feature_name=None, features_dir = None):
        assert (feature_name is not None), 'must provide name of feature to _getFeatureFilePath'
        if(features_dir is None):
            features_dir = self.features_dir;
        if(self.features_dir is None and self.feature_manager is not None):
            return self.feature_manager._get_feature_file_path_for(self, feature_name=feature_name, features_dir=features_dir);
        return os.path.join(features_dir, self._get_feature_file_name(feature_name));

    def _save_feature_to_path(self, name, path):
        """Subclasses can implement version of this that will check members for features if those features arent found here."""
        return self.features.save_entry(name=name, path=path);

    def _save_all_features_to_path(self, path):
        return self.features.save(path=path);

    def _load_feature_from_path(self, path, assign_name = None):
        """Subclasses can implement version of this that will check whether feature is registered before loading."""
        return self.features.load_entry(path=path, assign_name=assign_name);

    def _load_all_features_from_path(self, path):
        return self.features.load(path=path);

    def _load_each_feature_from_dir(self, features_dir):
        self.features.load_entries_from_dir(dir_path = features_dir);
        return;

    # </editor-fold>
    ##################\\--Default Save/Load Functions (when simple feature directory is used)--//##################

    def get_copy_of_features(self):
        # SavesFeatures
        featurescopy = self.features.deep_clone();
        featurescopy.owner = None;
        return featurescopy;
        # clone._save_features_func = self._save_features_func;
        # clone._load_features_func = self._load_features_func;
        # clone._clear_features_func = self._clear_features_func;
        # clone._features_root = None;