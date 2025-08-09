from ptlreg.apy.core import file_created_timestamp_string_from_path, get_file_creation_datetime
from ptlreg.apydn import HasFilePathLabel, FPath
from ptlreg.apydn.datanode.datasample.DataSample import DataSampleBase
from ptlreg.apydn.imagegraph.ImageSample import *
from ptlreg.apy.amedia.media.Image import Image
import os
from ptlreg.apydn.imagegraph.ImageSampleConstants import ImageSampleConstants

class DeleteImageSampleError(Exception):
    pass;

class ImageFileSampleMixin(HasFilePathLabel, ImageSampleMixin):
    """
    An ImageSample representing an image file on disk.
    """

    def __init__(self, path=None, root_path=None, **kwargs):
        """
        Initializes the ImageFileSampleMixin with a file path.
        :param path: The file path to the image file.
        :param root_path: The root path for the file.
        """
        super(ImageFileSampleMixin, self).__init__(path=path, root_path=root_path, **kwargs);

    def set_file_path(self, file_path=None, **kwargs):
        super(ImageFileSampleMixin, self).set_file_path(file_path=file_path, **kwargs);
        self.file_name = FPath.From(file_path).file_name;

    @classmethod
    def for_path(cls, path, root_path=None):
        newitem = cls(path=path, root_path=root_path);
        newitem.set_path(path);
        return newitem;

    # def _set_timestamp(self, use_name=None):
    #     """
    #     Sets the timestamp for the image sample.
    #     :param timestamp: The timestamp to set.
    #     """
    #     if(use_name is None):
    #
    #
    #     self.set_label_value(ImageSampleConstants.TIMESTAMP_KEY, timestamp);

    # @property
    # def file_path(self):
    #     return FPath.From(self.get_label_value(ImageSampleConstants.FILEPATH_KEY));

    def DELETE(self, for_real=False):
        fp = self.absolute_file_path
        if(not for_real):
            raise DeleteImageSampleError("You tried to delete {}; if you are sure about this set argument for_real=True...".format(fp));
        try:
            os.remove(fp);
        except OSError as e:
            # If it fails, inform the user.
            print("Error trying to delete image file sample: %s - %s." % (e.filename, e.strerror))


    def _calc_timestamp(self, force_use_metadata=None):
        """
        Calculates the timestamp for the image sample.
        If use_name is provided, it will use the file name to determine the timestamp.
        :param use_name: The name to use for calculating the timestamp.
        """
        self.set_timestamp(file_created_timestamp_string_from_path(self.absolute_file_path, force_use_metadata=force_use_metadata));

    # def setThumbnailPath(self, path):


    def set_path(self, path):

        # First set the full file path
        if(self.file_path is None):
            self.add_string_label(key=DataNodeConstants.FILE_PATH_KEY, value=FPath.From(path).absolute_file_path);
        else:
            self.set_label_value(DataNodeConstants.FILE_PATH_KEY, FPath.From(path).absolute_file_path);

        # Now we can set the file name based on the full path
        if(self.file_name is None):
            self.add_string_label(key=ImageSampleConstants.FILENAME_KEY, value=self.file_name);
        else:
            self.set_label_value(ImageSampleConstants.FILENAME_KEY, self.file_name);


    @property
    def timestamp(self):
        return self.get_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY)
    
    @timestamp.setter
    def timestamp(self, value):
        """
        sets the timestamp for the image sample.
        :param value:
        :return:
        """
        self.set_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY, value);

    # @property
    # def dataset_relative_filepath(self):
    #     return FPath.From(self.get_label_value(ImageSampleConstants.DATASET_RELATIVE_PATH_KEY));

    @property
    def dataset_relative_filepath(self):
        return self.get_label_value(ImageSampleConstants.DATASET_RELATIVE_PATH_KEY);


    def get_image(self, rotate_with_exif=False, **kwargs):
        im=Image(self.absolute_file_path, rotate_with_exif=rotate_with_exif, **kwargs);
        self.image_shape = im.shape;
        return im;

    def set_thumbnail_image(self, thumbnail):
        self.set_string_label(ImageSampleConstants.THUMBNAIL_PATH_KEY, FPath.PathString(thumbnail.file_path));

    @property
    def thumbnail_path(self):
        return self.get_label_value(ImageSampleConstants.THUMBNAIL_PATH_KEY);


    def get_thumbnail_image(self):
        return Image(self.thumbnail_path);
        # self.image_shape = im.shape;

    # @property
    # def file_name(self):
    #     fname = self.get_label_value(ImageSampleConstants.FILENAME_KEY);

    @property
    def file_name(self):
        return FPath.From(self.file_path).file_name;


    @property
    def sample_name(self):
        return FPath.From(self.file_name).file_name_base;


class ImageFileSample(ImageFileSampleMixin, DataSampleBase):
    """
    An ImageSample representing an image file on disk.
    This class is a concrete implementation of the ImageFileSampleMixin.
    """

    def __init__(self, path=None, root_path=None, **kwargs):
        """
        Initializes the ImageFileSample with a file path.
        :param path: The file path to the image file.
        :param root_path: The root path for the file.
        """
        super(ImageFileSample, self).__init__(path=path, root_path=root_path, **kwargs);
        self.init_node_id(**kwargs);

    def init_node_id(self, **kwargs):
        """
        Initializes the node_id if it is not set.
        This method should be called in the constructor of the subclass.
        """
        if ((self.node_id is None) and (self.has_label(DataNodeConstants.FILE_PATH_KEY))):
            self.set_node_id(self.file_path)
        else:
            if (self.node_id != self.file_path):
                raise ValueError("Node ID is not the same as file path: {} != {}".format(self.node_id, self.file_path));