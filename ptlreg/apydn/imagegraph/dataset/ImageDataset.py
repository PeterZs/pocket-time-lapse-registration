import os
import shutil
import warnings

from ptlreg.apy.amedia import Image
from ptlreg.apy.core import SavesFeatures, SavesDirectories, datetime_from_formatted_timestamp_string, get_file_creation_datetime
from ptlreg.apy.core.filepath import HasFilePath, FilePathList
from ptlreg.apy.defines import TIMESTAMP_FORMAT
from ptlreg.apydn.datanode.datasample.ManagesDataSamples import ManagesDataSamples
from ptlreg.apydn.imagegraph.ImageSampleConstants import ImageSampleConstants
from ptlreg.apydn.imagegraph.imagefilesample.ImageFileSampleSet import ImageFileSampleSet
from .ImageDatasetConstants import ImageDatasetConstants
from ptlreg.apydn.FPath import FPath
from ... import DataNodeConstants


class DeleteImageDatasetSampleError(Exception):
    pass;




class ImageDataset(ManagesDataSamples, SavesFeatures, SavesDirectories):
    '''
    `samples` is an ImageDatasetFileSampleSet which has ImageFileSample's as an element class and uses the label
    ImageSampleConstants.DATASET_RELATIVE_PATH_KEY for the main index function
    '''
    SAMPLE_SET_TYPE = ImageFileSampleSet;

    def __init__(self, path=None, **kwargs):
        input_path = path;
        if (isinstance(path, HasFilePath)):
            input_path = path.absolute_file_path;
        super(ImageDataset, self).__init__(path=input_path, **kwargs)
        self.original_sample_categories = [];

    def add_recipe_subdir(self, name, parent_recipe=None, add_images=False):
        recipe_subdir_name = self.get_recipe_subdir_name(name, parent_recipe=parent_recipe);
        if (recipe_subdir_name in self.directories):
            FPath.make_sure_dir_path_exists(self.get_dir(recipe_subdir_name));
            return;
        else:
            self.add_dir(recipe_subdir_name, recipe_subdir_name);
        if (add_images):
            rlist = [];
            subdirimpaths = FilePathList.from_directory(self.get_images_subdir(name),
                                                       extension_list=['.jpeg', '.jpg', '.png']);
            for fp in subdirimpaths:
                has_sample = self.get_image_sample_for_path(fp.absolute_file_path);
                if (not has_sample):
                    rlist.append(self.add_sample_for_image_path(fp.absolute_file_path));
            return self.create_sample_set(rlist);

    def clear_missing_samples(self):
        toremove = [];
        self.update_absolute_file_paths();
        for s in self.samples:
            if (not s.file_exists()):
                toremove.append(s);
        for s in toremove:
            self.samples.remove(s);

    def delete_sample(self, s, for_real=False):
        if (s not in self.samples):
            raise DeleteImageDatasetSampleError("Sample {} not in image dataset {}".format(s.file_path, self));
        s.DELETE(for_real=for_real)
        self.samples.remove(s);
        return s;

    def add_images_subdir_with_tags(self, subdir_name, tags=None):
        """
        Adds the subdir, and if the subdir already exists reads the images in it and adds them as samples with the provided tags.
        Note that TAGS ARE ONLY APPLIED TO NEWLY ADDED IMAGES
        :param subdir_name:
        :param tags:
        :return:
        """
        # print("adding subdir_name: {}\ntags: {}".format(subdir_name, tags));
        new_samples = self.add_images_subdir_if_missing(subdir_name, add_images=True);
        if (tags is None):
            tags = [];
        elif (isinstance(tags, str)):
            tags = [tags];
        for sample in new_samples:
            if (sample.has_label(ImageDatasetConstants.IMAGE_SUBDIR_TAG)):
                sample.set_label_value(ImageDatasetConstants.IMAGE_SUBDIR_TAG, subdir_name);
            else:
                sample.add_string_label(key=ImageDatasetConstants.IMAGE_SUBDIR_TAG, value=subdir_name)
            for t in tags:
                sample.set_tag_label(t);
        return self.create_sample_set(new_samples.asList());

    def _subdir_file_paths(self, subdir_name):
        return FilePathList.from_directory_search(self.get_images_subdir(subdir_name), recursive=False,
                                                  extension_list=['.jpeg', '.jpg', '.png'],
                                                  criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);

    def add_images_subdir_if_missing(self, name, add_images=False):
        """
        Adds the subdir if it is missing.
        :param name:
        :param add_images:
        :return: A list of ImageSample's for new images found in the directory after they have been added
        """
        images_subdir_name = self.get_images_subdir_name(name);
        if (images_subdir_name in self.directories):
            FPath.make_sure_dir_path_exists(self.get_dir(images_subdir_name));
        else:
            self.add_dir(images_subdir_name, images_subdir_name);
        if (add_images):
            rlist = [];
            subdirimpaths = FilePathList.from_directory_search(self.get_images_subdir(name), recursive=False,
                                                               extension_list=['.jpeg', '.jpg', '.png'],
                                                               criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);
            # subdirimpaths = FilePathList.from_directory(self.get_images_subdir(name), extension_list=['.jpeg','.jpg','.png']);
            for fp in subdirimpaths:
                has_sample = self.get_image_sample_for_path(fp.absolute_file_path);
                if (not has_sample):
                    rlist.append(self.add_sample_for_image_path(fp.absolute_file_path));

            return self.create_sample_set(rlist);
        else:
            return;

    def duplicate_sample_set_to_image_subdir_with_tags(self, sample_set, subdir_name, tags=None, ext=None, save=True,
                                                       scale=None, overwrite=False):
        new_samples = [];
        self.add_images_subdir_if_missing(subdir_name);
        subdir = self.get_images_subdir(subdir_name);
        if (tags is None):
            tags = subdir_name;
        if (isinstance(tags, str)):
            tags = [tags];
        for s in sample_set:
            use_ext = s.file_ext;
            if (ext is not None):
                use_ext = ext;
            spath = os.path.join(subdir, s.file_name_base + use_ext);
            if ((not os.path.exists(spath)) or overwrite):
                if (scale is None):
                    s.get_image().write_to_file(spath);
                else:
                    s.get_image().GetScaledByFactor(scale).write_to_file(spath);
            new_sample = self.add_sample_for_image_path(spath);
            for t in tags:
                new_sample.add_tag_label(t);
            new_samples.append(new_sample);
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    def get_image_sample_for_path(self, path):
        return self.get_sample_for_path(path);

    def get_sample_for_path(self, path):
        return self._samples_by_path_map.get(path);

    @property
    def n_samples(self):
        return self.samples.length();

    def update_absolute_file_paths(self):
        for s in self.samples:
            new_absolute_path = os.path.join(self.managed_dir, s.get_label_value(ImageSampleConstants.DATASET_RELATIVE_PATH_KEY));
            # print("Before: {}\nAfter: {}".format(s.file_path, new_absolute_path))
            s.set_label_value(DataNodeConstants.FILE_PATH_KEY, new_absolute_path);

    def _get_image_sample_for_sample(self, sample):
        '''
        This will get whatever sample has the same key as the provided one
        :param sample:
        :type sample:
        :return:
        :rtype:
        '''
        return self.samples._get_internal_node_for_node(sample);

    def delete_dir(self, name, really=False):
        if (not really):
            warnings.warn("DID YOU ACCIDENTALLY TRY TO EMPTY `{}` DIRECTORY???".format(name))
        samples = self.get_subdir_image_samples(name);
        for s in samples:
            self.samples.remove(s);
        super(ImageDataset, self).delete_dir(name);

    def get_subdir_image_samples(self, images_subdir):
        subdirimpaths = FilePathList.from_directory(self.get_images_subdir(images_subdir),
                                                   extension_list=['.jpeg', '.jpg', '.png']);
        rlist = [];
        for fp in subdirimpaths:
            rlist.append(self.get_image_sample_for_path(fp.absolute_file_path));
        return self.create_sample_set(rlist);

    def _sample_dir_name(self, sample):
        return sample.get_directory_name();

    def get_images_subdir(self, name):
        return self.get_dir(self.get_images_subdir_name(name));

    def get_images_subdir_name(self, name):
        return os.path.join(ImageDatasetConstants.IMAGES_SUBDIR_NAME, name);

    def get_recipe_subdir_name(self, name, parent_recipe=None):
        if (parent_recipe is not None):
            recipe_subdir_name = os.path.join(ImageDatasetConstants.RECIPE_SUBDIR_NAME, parent_recipe, name);
        else:
            recipe_subdir_name = os.path.join(ImageDatasetConstants.RECIPE_SUBDIR_NAME, name);
        return recipe_subdir_name;

    def get_recipe_subdir(self, name, parent_recipe=None):
        return self.get_dir(self.get_recipe_subdir_name(name, parent_recipe));

    def delete_recipe_subdir(self, name, tag=None):
        return self.delete_dir(self.get_recipe_subdir_name(name, tag=tag));

    def init_dirs(self, **kwargs):
        super(ImageDataset, self).init_dirs(**kwargs);
        self.add_dir_if_missing(name=ImageDatasetConstants.IMAGES_SUBDIR_NAME, folder_name=ImageDatasetConstants.IMAGES_SUBDIR_NAME);
        self.add_dir_if_missing(name=ImageDatasetConstants.FEATURES_DIR_NAME, folder_name=ImageDatasetConstants.FEATURES_DIR_NAME);
        self.add_dir_if_missing(self.features_dir);

    @property
    def features_dir(self):
        return self.get_dir(ImageDatasetConstants.FEATURES_DIR_NAME)

    @property
    def images_dir(self):
        return self.get_dir(ImageDatasetConstants.IMAGES_SUBDIR_NAME);

    def _add_images_subdir(self, images_subdir):
        self.add_images_subdir_if_missing(images_subdir);

    # def _pull_image(self, source, dest, rename_to_timestamp=True, width='default', can_use_ffmpeg=False, calibration=None, correct_distortion=True,
    #                 exif_data=None):
    #     print("NEED TO IMPLEMENT _pullImage IN SUBCLASS!")
    #     raise NotImplementedError;


    def _pull_image(self, source, dest, width=None, can_use_ffmpeg=False, **kwargs):
        """
        Pull the image, optionally changing its size, undistorting, and putting calibration information into the exif data of the new file
        :param source: path to source image
        :param dest: path to new file
        :param width: width of new image
        :param can_use_ffmpeg: whether it's ok to use ffmpeg. Probably not anymore...
        :return:
        """

        if(can_use_ffmpeg):
            assert(width is None)
            (
                ffmpeg
                .input(source, **{'noautorotate': None})
                .filter('scale', width, -1)
                .output(dest)
                .run()
            )
            return;
        else:
            im = Image(path=source, rotate_with_exif=False);
            if(width is None or im.shape[1]==width):
                # print("Using copy2 to copy {} to {}".format(source, dest));
                shutil.copy2(source, dest);
            else:
                imscaled = im.get_scaled_to_width(width);
                imscaled.write_to_file(dest, exif=im._get_exif());
        return;

        # #
        # im = Image(path=source, rotate_with_exif=False);
        # if (exif_data is None):
        #     exif_data = im._getEXIF();
        # if(width is not None):
        #     raise NotImplementedError("target width is disabled for now");
        #     if (width == 'default'):
        #         small_side = 1080;
        #         if(im.shape[1] < im.shape[0]):
        #             width = small_side;
        #         else:
        #             width = int(np.round(im.shape[1] * (small_side / im.shape[0])));

        # if(calibration and (not correct_distortion)):
        #     if (im.shape[1] == width or width is None):
        #         # save calibration without changing image pixels at all
        #         assert (im.shape[1] == calibration.calibration_shape[0]), "calibration does not match image shape"
        #         # p0.timestamp.strftime('%Y:%m:%d %H:%M:%S')
        #
        #         im.writeWithCalibrationEXIF(output_path=dest, calibration=calibration, exif_in=exif_data);
        #         return;
        #     else:
        #         # resize and save modified calibration
        #         assert (width == calibration.calibration_shape[0]), "calibration does not match image shape"
        #         imscaled = im.GetScaledToWidth(width);
        #         imscaled.writeWithCalibrationEXIF(
        #             output_path=dest,
        #             calibration=calibration.GetScaledToShape(imscaled.shape),
        #             original_calibration=calibration,
        #             exif_in=exif_data
        #         );
        #         return;


        # if(correct_distortion):
        #     assert(calibration is not None), "No calibration provided for distortion correction";
        #     if(calibration.calibration_shape[0]!=im.shape[0]):
        #         sratio = im.shape[0]/calibration.calibration_shape[0];
        #         assert(sratio == im.shape[1]/calibration.calibration_shape[1]), "Calibration shape appears to be wrong! {} with im shape {}".format(calibration.calibration_shape, im.shape)
        #         calibration = calibration.GetScaledToShape(im.shape);
        #     # assert(calibration.calibration_shape[0] == im.shape[0] and calibration.calibration_shape[1] == im.shape[1]), "Calibration shape appears to be wrong! {} with im shape {}".format(calibration.calibration_shape, im.shape)
        #     newImage, newCalibration = calibration.getUndistortedImageAndNewCalibration(im);
        #     if(width is not None and width != newImage.shape[1]):
        #         scaledImage = newImage.GetScaledToWidth(width);
        #         scaledImage.writeWithCalibrationEXIF(
        #             output_path=dest,
        #             calibration=newCalibration.GetScaledToShape(scaledImage.shape),
        #             original_calibration=calibration,
        #             exif_in=im._getEXIF()
        #         );
        #         scaled = width/im.shape[1];
        #         # print("wrote {} undistorted scaling x{}".format(FilePath.From(dest).file_name, scaled));
        #         print("wrote {} undist scaling x{}".format(FilePath.From(dest).file_path, scaled));
        #     else:
        #         newImage.writeWithCalibrationEXIF(
        #             output_path=dest,
        #             calibration=newCalibration,
        #             original_calibration=calibration,
        #             exif_in=im._getEXIF()
        #         );
        #         # print("wrote {} undistorted no scaling".format(FilePath.From(dest).file_name));
        #         print("wrote {} undist no scaling".format(FilePath.From(dest).file_path));
        #     return;




    def _add_dataset_relative_path_label(self, sample):
        sample.add_string_label(key=ImageSampleConstants.DATASET_RELATIVE_PATH_KEY,
                              value=self.relative_path(sample.absolute_file_path));

    def create_sample_for_image_path(self, path):
        # TODO: should add calibration info here?
        pathstring = FPath.From(path).absolute_file_path;
        imsample = self.__class__.SAMPLE_SET_TYPE.new_element(path = pathstring);
        self._add_dataset_relative_path_label(imsample);
        return imsample;

    def add_sample_for_image_path(self, path, on_repeat='skip'):
        newsample = self.create_sample_for_image_path(path)
        if (newsample in self.samples):
            if (on_repeat == 'skip'):
                return self._get_image_sample_for_sample(newsample);
                # return self.samples.get(newsample.node_id);
            else:
                raise NotImplementedError;
        self.add_sample(newsample);
        return newsample;

    def add_sample(self, sample):
        if (sample.has_label(ImageSampleConstants.DATASET_RELATIVE_PATH_KEY)):
            assert (sample.get_label_value(ImageSampleConstants.DATASET_RELATIVE_PATH_KEY) == self.relative_path(
                sample.absolute_file_path));
        self.samples.add(sample);

    # def SaveImageSet(self, image_set, folder_name):

    @staticmethod
    def default_filename_fn(file_name):
        return file_name;

    @property
    def _samples_by_path_map(self):
        return self.samples._main_index_map;

    def pull_directory(self, source_dir, images_subdir=None, width=None, filename_fn=None, on_repeat='skip', rename_to_timestamp=True):
        """
        Pulls all images from the source directory and adds them to the dataset.
        :param source_dir: The source directory to pull images from.
        :param images_subdir: The subdirectory in the dataset to store the images.
        :param width: The width to resize the images to, or 'default' for no resizing.
        :param filename_fn: A function to generate the filename for each image.
        :return: A list of new samples added to the dataset.
        """
        if (images_subdir is not None):
            self.add_images_subdir_if_missing(images_subdir);
        file_paths = FilePathList.from_directory(source_dir, extension_list=['.jpeg', '.jpg', '.png', '.heic'],);
        new_samples = [];
        for fpath in file_paths:
            new_sample = self.pull_new_image(fpath=fpath, images_subdir=images_subdir, width=width,
                                             filename_fn=filename_fn, rename_to_timestamp=rename_to_timestamp, on_repeat=on_repeat);
            if (new_sample is not None):
                new_samples.append(new_sample);
        return new_samples;


    def pull_new_image(self, fpath, images_subdir=None, filename_fn=None, on_repeat='skip', width=None, rename_to_timestamp=None, ext=None, **kwargs):
        original_image_name =fpath.file_name_base;
        timestamp = datetime_from_formatted_timestamp_string(original_image_name);
        if (timestamp is None):
            if(width is not None):
                assert rename_to_timestamp, "If you specify width, you should also rename to timestamp, because the exif data will change when resizing the image"
                # if(rename_to_timestamp is None):
                #     rename_to_timestamp = True;
                # else:

        if(ext is None):
            if(fpath.file_ext.lower() in ['.jpg', '.png']):
                ext = fpath.file_ext.lower();
            else:
                ext = '.jpg';


        if(timestamp is None):
            timestamp = get_file_creation_datetime(fpath.absolute_file_path);

        ofilename = original_image_name;
        if(rename_to_timestamp):
            # dt = datetime.fromtimestamp(timestamp)
            ofilename = timestamp.strftime(TIMESTAMP_FORMAT)

        if(filename_fn is not None):
            if (isinstance(filename_fn, str)):
                ofilename = filename_fn;
            else:
                ofilename = filename_fn(ofilename);
        if (ofilename is None):
            warnings.warn("Problem calculating name for pull_new_image on path {}".format(fpath))
            return None;


        if (images_subdir is None):
            destination = os.path.join(self.images_dir, ofilename+ext);
        else:
            destination = os.path.join(self.get_images_subdir(images_subdir), ofilename+ext);
        newSample = None;

        if (os.path.exists(destination)):
            if (on_repeat == 'skip'):
                # rsample = self.samples.getByFileName(FilePath.From(destination).file_name);
                rsample = self.get_image_sample_for_path(destination);
                if (rsample is not None):
                    return rsample;
                else:
                    newSample = self.add_sample_for_image_path(destination);
        else:
            # self._pull_image(fpath.absolute_file_path, destination, calibration=calibration,
            #                  correct_distortion=correct_distortion);
            self._pull_image(fpath.absolute_file_path, destination, **kwargs);
            newSample = self.create_sample_for_image_path(destination);
            if (newSample in self.samples):
                warnings.warn("{} duplicate node id {}".format(newSample.file_name, newSample.node_id))
            else:
                self.samples.add(newSample);
        return newSample;

    def pull_new_images(self, filePathList, images_subdir=None, with_tag=None, width='default', filename_fn=None,
                        on_repeat='skip'):
        if (images_subdir is not None):
            self.add_images_subdir_if_missing(images_subdir);
        for imgfp in filePathList:
            self.pull_new_image(fpath=imgfp, images_subdir=images_subdir, width=width, filename_fn=filename_fn,
                                on_repeat=on_repeat);



