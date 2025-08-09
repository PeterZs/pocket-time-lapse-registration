import subprocess
import logging
import os
from ptlreg.apy.core.filepath import *
from ptlreg.apydn.datanode import DataNode
import pandas as pd
from .COLMAPDB import COLMAPDB
from .database import COLMAPDatabase
import numpy as np

USE_PYCOLMAP = False;

try:
    import pycolmap
except ImportError:
    print("PYCOLMAP NOT INSTALLED")


COLMAP_SIFT_OPTIONS = {
    'default': {},
    'sparse': {
        '--SiftExtraction.first_octave': '0'
    }
}

COLMAP_MATCHING_OPTIONS = {
    'default': {},
    'strict': {
        "--SiftMatching.max_num_trials": "100000",
        "--SiftMatching.min_inlier_ratio": "0.05",
        "--SiftMatching.max_error": "2",
    },
}


def runCommands(cmds):
    print(' '.join(cmds));
    subprocess.run(cmds, check=True);


COLMAP_EXE = 'colmap';


class COLMAPper(DataNode):
    SaveName = 'colmapper.csv';

    def __init__(self, root_path=None, scene_name=None, images_path=None, calibration_node = None, **kwargs):
        super(COLMAPper, self).__init__(**kwargs)
        self.root_path = root_path;
        self.scene_name = scene_name;
        if (images_path is None):
            images_path = root_path;
        self._images_path = images_path;
        # if(calibration_node is None):

            # raise ValueError("Must create COLMAPper with calibration node!");
        # elif(calibration_node == -1):
        #     return;
        # else:
        #     self.calibration_params = dict(
        #         string = calibration_node.getCalibrationParametersString(),
        #         values = list(calibration_node.calibration_params),
        #         model = calibration_node.model,
        #         calibration_shape = list(calibration_node.calibration_shape)
        #     )
            # self.writeINIFile(calibration_node);


    @property
    def calibration_params_string(self):
        return self.calibration_params['string'];

    @property
    def calibration_params_values(self):
        return np.array(self.calibration_params['values']);

    @property
    def calibration_params_model(self):
        return self.calibration_params['model'];

    @property
    def calibration_params_calibration_shape(self):
        return np.array(self.calibration_params['calibration_shape']);

    @property
    def calibration_params(self):
        return self.get_label('calibration_params');

    @calibration_params.setter
    def calibration_params(self, value):
        return self.set_label('calibration_params', value);

    def get_colmapdb(self):
        return COLMAPDB.from_colmapdb_file(self.db_path);

    @classmethod
    def load_from_directory(cls, root_path):
        check_path = cls.save_path_for_root(root_path);
        rval = None;
        if (os.path.exists(check_path)):
            rval = cls(calibration_node=-1);
            rval.load_data_labels_from_csv(check_path);
        return rval;

    @classmethod
    def save_path_for_root(cls, root_path):
        return os.path.join(root_path, COLMAPper.SaveName)

    def get_save_path(self):
        return self.__class__.save_path_for_root(self.root_path);

    def save(self):
        print("Saving COLMAPper to {}".format(self.get_save_path()));
        self.save_data_labels_to_csv(self.get_save_path());
        # self.data_labels.to_csv(self.getSavePath());

    def get_project_file_path(self):
        return os.path.join(self.root_path, 'colmap.ini');

    @property
    def db_path(self):
        return os.path.join(self.root_path, 'colmap.db');

    @property
    def root_path(self):
        return self.get_label('root_path');

    @root_path.setter
    def root_path(self, value):
        return self.set_label('root_path', value);

    @property
    def scene_name(self):
        return self.get_label('scene_name');

    @scene_name.setter
    def scene_name(self, value):
        return self.set_label('scene_name', value);

    @property
    def _images_path(self):
        return self.get_label('_images_path');

    @_images_path.setter
    def _images_path(self, value):
        return self.set_label('_images_path', value);

    @property
    def features_computed(self):
        return self.get_label('features_computed');

    @features_computed.setter
    def features_computed(self, value):
        return self.set_label('features_computed', value);

    @property
    def matches_computed(self):
        return self.get_label('matches_computed');

    @matches_computed.setter
    def matches_computed(self, value):
        return self.set_label('matches_computed', value);

    @property
    def match_list(self):
        return self.get_label('match_list');

    @match_list.setter
    def match_list(self, value):
        return self.set_label('match_list', value);

    @property
    def images_path(self):
        if (self._images_path is not None):
            return self._images_path;
        else:
            return self.root_path;

    @property
    def ini_file(self):
        return self.get_label('ini_file');

    @ini_file.setter
    def ini_file(self, value):
        return self.set_label('ini_file', value);


    # def createDatabaseWithCameras(self, images, get_camera_id_func, **kwargs):
    #     with COLMAPDatabase.connect(self.db_path) as db:
    #         db.create_tables()
    #
    #         camdict = dict();
    #         for im in images:
    #             im.
    #
    #         for camera in cameras:
    #             db.add_camera(camera);
    #         old_camera = db.add_camera(model1, width1, height1, params1)
    #         camera_id2 = db.add_camera(model2, width2, height2, params2)

    def create_project(self, remake_db=False, cameras=None, images=None):
        # create_project = [
        #     f'{COLMAP_EXE}',
        #     'project_generator',
        #     '--output_path',
        #     f'{self.getProjectFilePath()}'
        # ]
        # runCommands(create_project);
        # self.ini_file = self.getProjectFilePath();

        # if(calibration_node is None):
        #     raise ValueError("Provide calibration node");

        # self.writeINIFile(calibration_node=calibration_node);
        if(remake_db):
            os.remove(self.db_path);

        # with COLMAPDatabase.connect(self.db_path) as db:
        #     db.create_tables()
        #     for camera in cameras:
        #         db.add_camera(**camera);
        #     for image in images:
        #         db.add_image(image['name'], image['camera_id']);
        #     db.commit()



        # cmds = [
        #     f'{COLMAP_EXE}',
        #     'database_creator',
        #     '--database_path',
        #     f'{self.db_path}',
        # ]
        # '--project_path',
        # f'{self.ini_file}'

        # if (calibration_node):
        #     self.ini_file = self.writeINIFile(calibration_node=calibration_node, recompute=recompute);
        #     cmds.append('--project_path'),
        #     cmds.append(f'{self.ini_file}'),

        # runCommands(cmds);

        # db = COLMAPDatabase.connect(self.db_path)
        # db.create_tables()
        # for camera in cameras:
        #     db.add_camera(camera['model'], camera['width'], camera['height'], camera['params'])
        # for image in images:
        #     db.add_image(image['name'], image['camera_id'])
        # db.commit()
        # db.close()

        # for camera in cameras:
        #     # print(camera)
        #     for c in camera:
        #         print("CAMERA")
        #         print("{}: {}".format(c, camera[c]))

        with COLMAPDatabase.connect(self.db_path) as db:
            db.create_tables()
            db.commit()
            for camera in cameras:
                # print(camera)
                for c in camera:
                    print("CAMERA")
                    print("{}: {}".format(c, camera[c]))
                db.add_camera(**camera);
                db.commit()
            for image in images:
                db.add_image(image['name'], image['camera_id'], use_prior=False);
                db.commit()






    def detect_features(self, colmap_sift_type='default', use_gpu=True, calibration_node=None, single_camera=True, recompute=False):

        if (self.features_computed and (not recompute)):
            return;
        cmds = [
            f'{COLMAP_EXE}', 'feature_extractor', '--database_path', f'{self.db_path}',
            '--image_path', f'{self.images_path}'
        ]
        if (self.ini_file):
            cmds.append('--project_path')
            cmds.append(f'{self.ini_file}')

        if isinstance(colmap_sift_type, dict):
            feature_options = colmap_sift_type
        else:
            assert colmap_sift_type in ['default', 'sparse']
            feature_options = COLMAP_SIFT_OPTIONS[colmap_sift_type]

        if use_gpu:
            feature_options['--SiftExtraction.use_gpu'] = '1'
        else:
            feature_options['--SiftExtraction.use_gpu'] = '0'

        # if(calibration_node):
        #     feature_options['--ImageReader.camera_model'] = calibration_node.model
        #     feature_options['--ImageReader.camera_params'] = calibration_node.getCalibrationParametersString()

        # if(single_camera):
        #     feature_options['--ImageReader.single_camera'] = '1'

        for key, value in feature_options.items():
            cmds.append(key)
            cmds.append(value)

        print(cmds)
        logging.info(' '.join(cmds))
        subprocess.run(cmds, check=True)
        self.features_computed = True;
        self.save()
        return

    def save(self):
        self.get_data_labels_dataframe().to_csv(self.get_save_path())

    @property
    def match_list_path(self):
        custom = self.match_list;
        if (custom is not None):
            return custom;
        else:
            match_list_name = 'match_list.txt';
            return os.path.join(self.root_path, match_list_name);

    def match_features(self, use_gpu=True, colmap_matching_type='default', max_n_features=None, force_homography = True, recompute=False, default_dictionary_mode=False):
        if (self.matches_computed and (not recompute)):
            return;

        if ((not default_dictionary_mode) and not os.path.exists(self.match_list_path)):
            raise Exception("Could not find image match list!");

        if(not default_dictionary_mode):
            cmds = [
                f'{COLMAP_EXE}', 'matches_importer', '--database_path', f'{self.db_path}',
                '--match_list_path', f'{self.match_list_path}'
            ]
        else:
            cmds = [
                f'{COLMAP_EXE}', 'vocab_tree_matcher', '--database_path', f'{self.db_path}',
                '--VocabTreeMatching.vocab_tree_path', f'{"/Volumes/AbeAPFS/Files/UnstructuredTimeLapse/Reconstructions/vocabtrees/vocab_tree_flickr100K_words1M.bin"}'
            ]

        if isinstance(colmap_matching_type, dict):
            matching_options = colmap_matching_type
        else:
            assert colmap_matching_type in ['default', 'strict']
            matching_options = COLMAP_MATCHING_OPTIONS[colmap_matching_type]

        if use_gpu:
            matching_options['--SiftMatching.use_gpu'] = '1'
        else:
            matching_options['--SiftMatching.use_gpu'] = '0'

        # # if (force_homography):
        # #     matching_options['--SiftMatching.planar_scene'] = '1'
        # #     #
        #
        # else:
        #     raise NotImplementedError("COLMAPper assumes planar scenes for now!")
        if(max_n_features is None):
            max_n_features = '100000'
        if (max_n_features):
            matching_options['--SiftMatching.max_num_matches'] = max_n_features;

        for key, value in matching_options.items():
            cmds.append(key)
            cmds.append(value)

        logging.info(' '.join(cmds))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(cmds)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        runCommands(cmds)
        self.matches_computed=True;
        self.save();
        return

    def write_image_match_list(self, filepath_pair_list, output_file_name=None, recompute=False):
        if (output_file_name is None):
            output_file_name = 'match_list.txt';
        output_path = os.path.join(self.root_path, output_file_name);

        if (os.path.exists(output_path) and (not recompute)):
            return output_path;

        f = open(output_path, "a");

        def relpath(fpin: FilePath):
            return FilePath.From(fpin).relative(self.root_path);

        for pair in filepath_pair_list:
            f.write("{} {}\n".format(relpath(pair[0]), relpath(pair[1])));
        f.close()
        self.set_label('match_list', output_path);
        self.save();
        return output_path;

    def write_ini_file(self, recompute=False):
        filecontents = ("log_to_stderr =true\n"+
                        "database_path ={database_path}\n".format(database_path=self.db_path)+
                        # "image_path = {images_path}\n".format(images_path=self.images_path)+
                        "[ImageReader]\n"+
                        "single_camera = true\n"+
                        "single_camera_per_folder =false\n"+
                        "single_camera_per_image = false\n"+
                        "camera_model = {camera_model}\n".format(camera_model=self.calibration_params_model)+
                        "camera_params = {camera_params_string}\n".format(camera_params_string=self.calibration_params_string)+
                        "[SpatialMatching]\n"+
                        "planar_scene=1\n"
                        )

        output_path = os.path.join(self.root_path, 'colmap.ini');
        if (os.path.exists(output_path) and (not recompute)):
            return output_path;
        f = open(output_path, "a");
        f.write(filecontents);
        f.close()
        self.ini_file = output_path;
        self.save();
        return output_path;
