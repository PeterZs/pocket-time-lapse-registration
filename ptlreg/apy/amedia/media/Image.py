from ptlreg.apy.core import IsList
from ptlreg.apy.utils import *
from matplotlib.colors import hsv_to_rgb
import base64
from ptlreg.apy.amedia.media import MediaObject
from ptlreg.apy.amedia.signals import *
from ptlreg.apy.amedia.media.examplefiles import *
# from ptlreg.apymedia.media.mobjects.ImageTextUtils import *
import scipy as sp
from PIL import Image as PIM
from PIL import ImageOps as PImageOps
import enum
import requests, io
import png
import numpy as np

from ...core.filepath import FilePath

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import ptlreg.apy.utils

try:
    import pillow_heif
except ImportError:
    AWARN("`pillow_heif` is not installed. You will not be able to read HEIC images. Try:\npip install pillow-heif\n")

import matplotlib
import matplotlib.pyplot as plt
from ptlreg.apy.core.SavesFeatures import FeatureFunction
from .ImageText import *

from PIL.ExifTags import TAGS as PILEXIFTAGS


CALIBRATION_MAKERNOTE_DATA_KEY = "CurrentCameraCalibration"
ORIGINAL_CALIBRATION_MAKERNOTE_DATA_KEY = "OriginalCurrentCameraCalibration"

MAKERNOTE_EXIF_KEY = 3750;
FocalLengthIn35mmFilmEXIF_KEY = 41989
for k in PILEXIFTAGS:
    if(PILEXIFTAGS[k] == "MakerNote"):
        MAKERNOTE_EXIF_KEY = k;
    if(PILEXIFTAGS[k]=="FocalLengthIn35mmFilm"):
        FocalLengthIn35mmFilmEXIF_KEY = k;
        # print(MAKERNOTE_EXIF_KEY)





_GPyrDownKernel = np.array([[ 1,  4,  6,  4,  1],
                           [ 4, 16, 24, 16,  4],
                           [ 6, 24, 36, 24,  6],
                           [ 4, 16, 24, 16,  4],
                           [ 1,  4,  6,  4,  1]])*np.true_divide(1.0, 256);


class _ColorSpaces(enum.Enum):
    RGB="RGB";
    NormalizedLAB='NormalizedLAB';
    HSV='HSV';

class Image(MediaObject):
    """Image
    """
    IMAGE_TEMP_DIR = None;

    GaussianPyramidKernel = _GPyrDownKernel;

    ColorSpaces = _ColorSpaces;

    @staticmethod
    def MEDIA_FILE_EXTENSIONS():
        return ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'];

    def __init__(self, path = None, pixels = None, convert_to_float=False, hdr = False, rotate_with_exif=False, **kwargs):
        # You can do Image(pixels) or Image(path)
        self._samples = None;
        if(isinstance(path, np.ndarray) and (pixels is None)):
            # if the path looks like pixels and pixels are undefined, treat the path as pixels
            pixels = path;

        self._pixels = None;
        super(Image, self).__init__(path=path, **kwargs);
        self._pixels = pixels;
        if(path and (not pixels)):
            if (self.file_ext == ".heic" or self.file_ext == ".HEIC"):
                self._load_image_data_heic(hdr=hdr);
            elif (hdr):
                self.load_image_data_hdr();
            else:
                self.load_image_data(rotate_with_exif=rotate_with_exif);
        if(convert_to_float):
            self.set_pixel_type_to_float();
        self._colorspace = self.ColorSpaces.RGB;

    @classmethod
    def load_hdr(cls, path):
        return Image(path=path, hdr=True);

    def load_image_data_hdr(self, path = None, force_reload=True):
        if(path):
            self.set_file_path(path);
        if(self.file_path):
            if(force_reload or (not self.pixels)):
                if(self.file_ext == '.png'):
                    self._load_image_data_png();
                elif(self.file_ext == '.tif'):
                    self._load_image_data_tiff();
                elif (self.file_ext == '.tiff'):
                    self._load_image_data_tiff();
                elif (self.file_ext == '.heic' or self.file_ext == '.HEIC'):
                    self._load_image_data_heic(hdr=True);
                else:
                    self.load_image_data(force_reload=force_reload);

    def _load_image_data_heic(self, path=None, hdr=False):
        if(hdr):
            raise NotImplementedError("HDR HEIC not implemented yet.")
        if(path is None):
            path = self.file_path;
        if pillow_heif.is_supported(path):
            heif_file = pillow_heif.open_heif(path)
            # self.set_info("heif_mode", heif_file.mode)
            # self.set_info("heif_info", heif_file.info)
            # info_props = [
            #     'bit_depth',
            # ]
            # print(heif_file)
            #
            # self.set_info("heif_bit_depth", heif_file["bit_depth"])
            # self.set_info("heif_size", heif_file.size)

            np_array = np.asarray(heif_file)
            self._pixels = np_array;

    def _load_image_data_png(self):
        reader = png.Reader(self.file_path);
        pngdata = reader.read()
        self._pixels = np.vstack(map(np.array, pngdata[2]));
        self.update_info(pngdata[3]);
        self._pixels = np.reshape(self._pixels, (pngdata[1], pngdata[0], self.get_info('planes')))
        bitdepth = self.get_info('bitdepth');
        maxval = (np.power(2.0,bitdepth));
        print("HDR with bit depth {}".format(bitdepth));
        self._pixels = np.true_divide(self._pixels, maxval);

    def _load_image_data_tiff(self):
        import tifffile;
        self._pixels = tifffile.imread(self.file_path);

    def init_from_aobject(self, fromobject, share_data = False):
        """
        This may be used in cloning situations where a file is not directly read. Transfer all of the media data.
        Also transfer any attributes from other parent classes in the case of multiple inheritance.
        :param fromobject:
        :return:
        """
        if(share_data):
            self._pixels = fromobject._pixels;
            super(Image, self).init_from_aobject(fromobject);
        else:
            super(Image, self).init_from_aobject(fromobject);
            if(fromobject._pixels is not None):
                self._pixels = fromobject._pixels.copy();

    ##################//--Pixel Access--\\##################
    # <editor-fold desc="Pixel Access">

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self.pixels = np.power(self.pixels, value/self._gamma);
        self._gamma = value;


    @property
    def pixels(self):
        """
        clipped pixel access
        :return:
        """
        return self.samples;
    @pixels.setter
    def pixels(self, data):
        self.samples = data;

    @property
    def _pixels(self):
        """
        unclipped pixel access
        :return:
        """
        return self._samples;
    @_pixels.setter
    def _pixels(self, value):
        self._samples = value;


    @property
    def n_color_channels(self):
        if(len(self.pixels.shape)<3):
            return 1;
        else:
            return self.pixels.shape[2];

    @property
    def dtype(self):
        return self.pixels.dtype;

    @property
    def _is_float(self):
        return (self.dtype.kind in 'f');

    @property
    def _is_int(self):
        return (self.dtype.kind in 'iu');

    @property
    def fpixels(self):
        if(self._is_float):
            return self.pixels;
        else:
            return self.pixels.astype(float)*np.true_divide(1.0,255.0);

    @property
    def _fpixels(self):
        if(self._is_float):
            return self._pixels;
        else:
            return self._pixels.astype(float)*np.true_divide(1.0,255.0);

    @property
    def ipixels(self):
        if (self._is_int):
            return self.pixels;
        else:
            return (self.pixels * 255).astype(np.uint8);

    @property
    def _ipixels(self):
        if (self._is_int):
            return self._pixels;
        else:
            return (self._pixels * 255).astype(np.uint8);

    @property
    def shape(self):
        return np.asarray(self.pixels.shape)[:];

    @property
    def _shape(self):
        return np.asarray(self._pixels.shape)[:];

    def pad(self, left=0, right=0, top=0, bottom=0, **kwargs):
        pad_width = [[top,bottom], [left,right],[0,0]];
        self.pixels = np.pad(self.pixels, pad_width, **kwargs);
        return self;

    def get_padded(self, left=0, right=0, top=0, bottom=0, **kwargs):
        rval = self.clone(share_data=False);
        rval.pad(left=left, right=right, top=top, bottom=bottom, **kwargs);
        return rval;



    def get_scaled(self, shape=None, shape_xy=None):
        shapeis = (shape is not None);
        shapexyis = (shape_xy is not None);
        assert(shapeis != shapexyis), "Must provide only one of shape or shape_xy for Image.get_scaled"
        if(shapeis):
            sz=[shape[0], shape[1]];
        else:
            sz=[shape_xy[1],shape_xy[0]];
        # if(self.n_color_channels>1):
        #     sz = [sz[0], sz[1], self.n_color_channels];
        # imK = sp.misc.imresize(self.pixels, size=sz);
        imK = np.array(PIM.fromarray(self.ipixels).resize((int(sz[1]),int(sz[0]))));
        return self.mobject_class(pixels=imK);

    def get_scaled_to_width(self, width):
        ratio = np.true_divide(width, self.shape[1]);
        new_height = self.shape[0]*ratio;
        imK = np.array(PIM.fromarray(self.ipixels).resize((width,int(np.round(new_height)))));
        return self.__class__(pixels=imK);

    def get_cropped(self, x_range=None, y_range=None):
        full_shape = self.shape;
        if(x_range is None):
            xrange =[0,full_shape[1]];
        else:
            xrange = [x_range[0],x_range[1]];
        if(y_range is None):
            yrange =[0,full_shape[0]];
        else:
            yrange = [y_range[0],y_range[1]];
        if(xrange[0] is None):
            xrange[0]=0;
        if(xrange[1] is None):
            xrange[1]=full_shape[1];
        if(yrange[0] is None):
            yrange[0]=0;
        if(yrange[1] is None):
            yrange[1]=full_shape[0];
        return Image(pixels=self.pixels[yrange[0]:yrange[1], xrange[0]:xrange[1]]);

    def get_scaled_by_factor(self, factor=1.0):
        return self.get_scaled([int(self.shape[0] * factor), int(self.shape[1] * factor)]);

    def set_pixel_type_to_float(self):
        if(self._is_float):
            return;
        else:
            self._samples = self._fpixels;
        return self;

    def set_pixel_type_to_uint(self):
        if(self._is_int):
            return;
        else:
            self._samples = self._ipixels;
        return self;

    def get_uint_copy(self):
        clone = self.clone(share_data=False);
        clone.set_pixel_type_to_uint();
        return clone;

    def get_float_copy(self):
        clone = self.clone(share_data=False);
        clone.set_pixel_type_to_float();
        return clone;
    # </editor-fold>
    ##################\\--Pixel Access--//##################

    def clear(self):
        self.pixels = np.zeros(self.pixels.shape);

    def load_image_data(self, path = None, rotate_with_exif=True, force_reload=True):
        if(path):
            self.set_file_path(path);
        if(self.file_path):
            if(force_reload or (not self.pixels)):
                pim = PIM.open(fp=self.file_path);
                if (rotate_with_exif):
                    pim = PImageOps.exif_transpose(pim)
                self._pixels = np.array(pim);
                # self.pixels = np.array(pim);

    @staticmethod
    def solid_rgba_pixels(shape, color=None):
        if(color is None):
            color = [0,0,0,0];
        rblock = np.ones((shape[0],shape[1],4));
        rblock[:]=color;
        return rblock;

    @staticmethod
    def solid_rgb_pixels(shape, color=None):
        if(color is None):
            color = [0,0,0,0];
        rblock = np.ones((shape[0],shape[1],3));
        rblock[:]=color;
        return rblock;

    @classmethod
    def solid_image(cls, shape, color=None):
        if(color is None):
            return cls(pixels=cls.solid_rgba_pixels(shape, [0, 0, 0]));
        elif(len(color)==3):
            return cls(pixels=cls.solid_rgb_pixels(shape, color));
        elif(len(color)==4):
            return cls(pixels=cls.solid_rgba_pixels(shape, color));
        else:
            raise NotImplementedError;

    @classmethod
    def zeros(cls, shape):
        return Image(pixels=np.zeros(shape));

    @classmethod
    def ones(cls, shape):
        return Image(pixels=np.ones(shape));

    @classmethod
    def gaussian_noise(cls, size, mean=0, std=1):
        return cls(pixels=np.random.normal(mean, std, size));

    @classmethod
    def row(cls, images):
        totalw = 0;
        maxh = 0;
        for im in images:
            totalw = totalw+im.width;
            if(im.height>maxh):
                maxh = im.height;
        rim = cls.zeros([maxh, totalw, 3]);
        currentw = 0;
        for im in images:
            rim.pixels[:im.height,currentw:currentw+im.width,:]=im.fpixels;
            currentw = currentw+im.width;
        return rim;



    @property
    def width(self):
        return self.shape[1];

    @property
    def height(self):
        return self.shape[0];


    @property
    def possible_value_range(self):
        if (self.dtype.kind in 'iu'):
            return [0,255];
        else:
            return [0.0, 1.0];

    def reflect_y(self):
        self.pixels[:,:,:]=self.pixels[::-1,:,:];

    def reflect_x(self):
        self.pixels[:,:,:]=self.pixels[:,::-1,:];

    def pil(self):
        return PIM.fromarray(np.uint8(self.ipixels));

    def _get_rgb_channels(self):
        return self.pixels[:,:,0:3];

    def get_rgb_copy(self, background=None):
        c = self.clone(share_data=False);
        if(c._colorspace is not Image.ColorSpaces.RGB):
            c._converColorSpaceToRGB();
        if(c.n_color_channels==1):
            c._samples = np.stack((c._samples,)*3, axis=-1);
            return c;
        if(c.n_color_channels==4):
            bg = np.zeros_like((self.shape[0],self.shape[1],3));
            if(background is not None):
                bg[:]=background;
            alpha = self.fpixels[:,:,3];
            a = np.dstack([alpha, alpha, alpha]);
            c._samples = bg*(1.0-a)+a*c.fpixels[:,:,:3];
            return c;
        else:
            assert(False), "Not sure how to convert {} channels to RGB".format(self.n_color_channels);

    def get_rgba_copy(self):
        if(self.n_color_channels==4):
            return self.clone(share_data=False);
        if(self.n_color_channels==3):
            clone = self.clone(share_data=False);
            clone._add_channel_to_unclipped_samples(np.ones(self._samples.shape[:2]));
            return clone;

    def _add_channel_to_unclipped_samples(self, channel_data):
        if(channel_data.dtype.kind == self.dtype.kind):
            self._samples = np.dstack((self._samples, channel_data));
            return;

        if(channel_data.dtype.kind in 'iu'):
            if(self._is_int):
                self._samples = np.dstack((self._samples, channel_data));
                return;
            else:
                self._samples = np.dstack((self._samples, channel_data.astype(float)*np.true_divide(1.0, 255.0)));
                return;
        else:
            assert(channel_data.dtype.kind in 'f'), "unknown dtype {}".format(channel_data.dtype);
            if(self._is_float):
                self._samples = np.dstack((self._samples, channel_data));
                return;
            else:
                self._samples = np.dstack((self._samples, (channel_data*255).astype(np.uint8)));
                return;
        assert(False), "Should not get here!\nchannel_data.dtype={}\nself.pixels.dtype={}".format(channel_data.dtype, self.pixels.dtype);

    def get_gray_copy(self):
        if(self.n_color_channels==1):
            return self.clone(share_data=False);
        if(self.n_color_channels==4):
            return self.get_rgb_copy().get_gray_copy();
        if(self.n_color_channels==3):
            clone = self.clone(share_data=False);
            clone.pixels = np.mean(clone.fpixels, 2);
            return clone;

    def get_clip(self, start_x=None, end_x=None, start_y=None, end_y=None):
        clone = self.clone(share_data=False);
        clone.clip_to(start_x=start_x, end_x=end_x, start_y=start_y, end_y=end_y);
        return clone;

    def clip_to(self, start_x=None, end_x=None, start_y=None, end_y=None):
        if(start_x is None):
            start_x=0;
        if(start_y is None):
            start_y=0;
        if(end_x is None):
            end_x=self.shape[1];
        if(end_y is None):
            end_y=self.shape[0];
        self.pixels = self.pixels[start_y:end_y,start_x:end_x];

    def get_fft_image(self):
        fftim = Image.Zeros(self.shape);
        if(self.n_color_channels==1):
            fftim.pixels = np.fft.fftshift(np.fft.fft2(self.fpixels));
        else:
            for c in range(self.n_color_channels):
                fftim.pixels[:,:,c]=np.fft.fftshift(np.fft.fft2(self.fpixels[:,:,c]));
        return fftim;


    def normalize(self, scale=None):
        if(scale is None):
            scale = self.dtype_value_range[1];
        self.pixels = self.pixels/np.max(self.pixels.ravel());
        self.pixels = self.pixels*scale;

    def show(self, title=None, new_figure = True, **kwargs):
        if (ptlreg.apy.utils.is_notebook()):
            Image.show_image(self, new_figure=new_figure, title=title, **kwargs);
        else:
            self.pil().show();

    def show_channel(self, channel=0, **kwargs):
        cim = self.get_channel_image(channel);
        cim.show();

    def get_channel_image(self, channel=0, **kwargs):
        return Image(pixels=self._get_channel_pixels(channel));

    @property
    def dtype_value_range(self):
        """
        Returns the value range of the image.
        :return: a tuple of (possible_value_range, current_value_range)
        """
        if(self._is_float):
            return [0.0, 1.0];
        else:
            return [0, 255];

    def _set_value_range(self, value_range=None):
        if(value_range is None):
            # value_range = [0,1];
            value_range = self.dtype_value_range[1];
        data = self.samples;
        maxval = np.max(data);
        minval = np.min(data)
        currentscale = maxval-minval;
        data = data-minval;
        data = data*(value_range[1]-value_range[0])/currentscale
        data = data+value_range[0];
        self._samples = data;

    def get_with_values_mapped_to_range(self, value_range=None):
        if(value_range is None):
            # value_range = [0,1];
            value_range = self.dtype_value_range[1];
        remap = self.clone();
        remap._set_value_range(value_range=value_range);
        return remap;

    def get_with_values_clipped_to_range(self, value_range=None):
        if(value_range is None):
            value_range = self.dtype_value_range[1];
            # value_range = [0,1];
        remap = self.clone();
        remap.pixels = np.clip(remap.pixels, value_range[0], value_range[1]);
        return remap;

    @staticmethod
    def show_image(im, title=None, new_figure=True, axis=None, **kwargs):
        if(isinstance(im, Image)):
            imdata = im.pixels;
        else:
            imdata = im;

        if(imdata.dtype == np.int64 or imdata.dtype == np.int32):
            imdata = imdata.astype(np.uint8)


        if  (ptlreg.apy.utils.is_notebook()):
            if (new_figure):
                if(title is not None):
                    f = plt.figure(num=title);
                else:
                    f = plt.figure();
            if (len(imdata.shape) < 3):
                if(imdata.dtype==np.uint8):
                    nrm = matplotlib.colors.Normalize(vmin=0, vmax=255);
                else:
                    nrm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0);
                if(axis is not None):
                    axis.imshow(imdata, cmap='gray', norm=nrm, **kwargs);
                else:
                    plt.imshow(imdata, cmap='gray', norm=nrm, **kwargs);


            elif(imdata.shape[2]==2):
                if(axis is not None):
                    axis.imshow(Image._Flow2RGB(imdata), **kwargs);
                else:
                    plt.imshow(Image._Flow2RGB(imdata), **kwargs);
            else:
                if(axis is not None):
                   axis.imshow(imdata, **kwargs);  # divided by 255
                else:
                    plt.imshow(imdata, **kwargs);  # divided by 255
            plt.axis('off');
            if(title):
                plt.title(title);

    def _get_play_html(self, format='png'):
        a = np.uint8(self.samples);
        f = StringIO();
        PIM.fromarray(a).save(f, format)
        ipdob = ptlreg.apy.utils.aget_ipython().display.Image(data=f.getvalue());
        # encoded = base64.b64encode(f.getvalue());
        imghtml='''<img src="data:image/png;base64,{0}"'''.format(ipdob._repr_png_);
        return imghtml;
        # return ipdob._repr_html_();

    def play(self):
        """
        just to be compatible with other MediaObjects...
        :return:
        """
        self.show();

    @classmethod
    def from_url(cls, url):
        response = requests.get(url)
        bytes_im = io.BytesIO(response.content)
        pix = np.array(PIM.open(bytes_im));
        im = cls(pixels=pix);
        return im;

    @classmethod
    def from_plot_fig(cls, fig, shape=None):
        if(shape is not None):
            fig.set_size_inches(shape[1]/fig.dpi, shape[0]/fig.dpi);
        fig.canvas.draw();
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        return Image(pixels=data);

    def write_to_file(self, output_path=None, **kwargs):
        self.pil().save(output_path);

    def write_with_maker_note_dict_data(self, output_path=None, makernote_dict=None, exif_in=None, **kwargs):
        exif = exif_in;
        if(exif is None):
            exif = self._get_exif();
        makernote_data = pickle.dumps(makernote_dict);
        exif[MAKERNOTE_EXIF_KEY]=makernote_data;
        self.PIL().save(output_path, exif=exif);

    # Calibration is an object with calibration data
    # Here I encode it and safve it in the makernote exif data
    def write_with_calibration_exif(self, output_path=None, calibration=None, original_calibration=None, exif_in=None, **kwargs):
        data_dict = dict();
        data_dict[CALIBRATION_MAKERNOTE_DATA_KEY] = calibration.GetCalibrationDataDict();
        if(exif_in is None):
            exif_in = self._get_exif();
        if(original_calibration is not None):
            data_dict[ORIGINAL_CALIBRATION_MAKERNOTE_DATA_KEY] = original_calibration.GetCalibrationDataDict();

        fl35 = int(np.round(calibration.FocalLengthIn35mmFilm));
        # exif_in[FocalLengthIn35mmFilmEXIF_KEY] = np.array([fl35],dtype="int16")[0]
        exif_in[FocalLengthIn35mmFilmEXIF_KEY] = fl35;
        return self.write_with_maker_note_dict_data(output_path=output_path, makernote_dict= data_dict, exif_in=exif_in);

    def write_with_exif(self, output_path=None, exif_in=None, **kwargs):
        exif = exif_in;
        if(exif is None):
            exif = self._get_exif();
        self.PIL().save(output_path, exif=exif);




    def get_exif_focal_length_in_35mm_film(self):
        return self._get_exif()[FocalLengthIn35mmFilmEXIF_KEY]

    def _read_maker_note_dict_data(self):
        exif = self._get_exif();
        if (MAKERNOTE_EXIF_KEY in exif):
            return pickle.loads(exif[MAKERNOTE_EXIF_KEY])
        else:
            return None;


    def _get_exif(self):
        if (self.file_path):
            fp = FilePath.From(self.file_path)
            pim = PIM.open(fp.absolute_file_path);
            exif = pim.getexif();
        else:
            exif = PIM.Exif();
        return exif;


    def get_encoded_base64(self):
        return base64.b64encode(self.pixels);

    def get_data_as_string(self):
        return self.pixels.tostring();

    def _get_channel_pixels(self, channel=None):
        if(channel is None):
            channel = 0;
        return self.pixels[:, :, channel];

    def _get_alpha_pixels(self):
        if(self.n_color_channels<4):
            if(self._is_float):
                return np.ones([self.imshape[0],self.imshape[1]]);
            else:
                return (np.ones([self.imshape[0],self.imshape[1]])*255).astype(np.uint8);
        else:
            return self._get_channel_pixels(3);

    def get_alpha_as_rgb(self):
        alpha = self._get_alpha_pixels();
        return Image(pixels=np.dstack([alpha, alpha, alpha]));
        # return Image._MultiplyArrayAlongChannelDimension(alpha, 3);

    @classmethod
    def from_base64(cls, encoded_data, shape):
        d = base64.decodestring(encoded_data);
        npar = np.frombuffer(d, dtype=np.float64);
        rIm = cls(pixels=np.reshape(npar, shape));
        return rIm;

    @staticmethod
    def from_data_string(data_string, shape, dtype=None):
        if(dtype is None):
            dtype=np.float64;
        img_1d = np.fromstring(data_string, dtype=dtype);
        rimg = img_1d.reshape((shape[1], shape[0], -1))
        return rimg;

    @classmethod
    def _rgbf_2_hsvf(cls, Rf, Gf, Bf):
        """
        code modified from https://www.icaml.org/canon/data/images-videos/HSV_color_space/HSV_color_space.html
        """
        # RGB_normalized = RGB / 255.0                           # Normalize values to 0.0 - 1.0 (float64)
        # R = RGB_normalized[:, :, 0]                            # Split channels
        # G = RGB_normalized[:, :, 1]
        # B = RGB_normalized[:, :, 2]
        R=Rf;
        G=Gf;
        B=Bf;
        RGB_normalized=np.dstack((Rf,Gf, Bf));
        v_max = np.max(RGB_normalized, axis=2)                 # Compute max, min & chroma
        v_min = np.min(RGB_normalized, axis=2)
        C = v_max - v_min

        hue_defined = C > 0                                    # Check if hue can be computed

        r_is_max = np.logical_and(R == v_max, hue_defined)     # Computation of hue depends on max
        g_is_max = np.logical_and(G == v_max, hue_defined)
        b_is_max = np.logical_and(B == v_max, hue_defined)

        H = np.zeros_like(v_max)                               # Compute hue
        H_r = ((G[r_is_max] - B[r_is_max]) / C[r_is_max]) % 6
        H_g = ((B[g_is_max] - R[g_is_max]) / C[g_is_max]) + 2
        H_b = ((R[b_is_max] - G[b_is_max]) / C[b_is_max]) + 4
        H[r_is_max] = H_r
        H[g_is_max] = H_g
        H[b_is_max] = H_b
        H *= 60
        V = v_max                                              # Compute value
        sat_defined = V > 0
        S = np.zeros_like(v_max)                               # Compute saturation
        S[sat_defined] = C[sat_defined] / V[sat_defined]
        return np.dstack((H, S, V))

    @classmethod
    def _hsvf_2_rgbf(cls, Hf, Sf, Vf):
        HSV = np.dstack((Hf,Sf,Vf));
        H = HSV[:, :, 0]                                           # Split attributes
        S = HSV[:, :, 1]
        V = HSV[:, :, 2]

        C = V * S                                                  # Compute chroma

        H_ = H / 60.0                                              # Normalize hue
        X  = C * (1 - np.abs(H_ % 2 - 1))                          # Compute value of 2nd largest color

        H_0_1 = np.logical_and(0 <= H_, H_<= 1)                    # Store color orderings
        H_1_2 = np.logical_and(1 <  H_, H_<= 2)
        H_2_3 = np.logical_and(2 <  H_, H_<= 3)
        H_3_4 = np.logical_and(3 <  H_, H_<= 4)
        H_4_5 = np.logical_and(4 <  H_, H_<= 5)
        H_5_6 = np.logical_and(5 <  H_, H_<= 6)

        R1G1B1 = np.zeros_like(HSV)                                # Compute relative color values
        Z = np.zeros_like(H)

        R1G1B1[H_0_1] = np.dstack((C[H_0_1], X[H_0_1], Z[H_0_1]))
        R1G1B1[H_1_2] = np.dstack((X[H_1_2], C[H_1_2], Z[H_1_2]))
        R1G1B1[H_2_3] = np.dstack((Z[H_2_3], C[H_2_3], X[H_2_3]))
        R1G1B1[H_3_4] = np.dstack((Z[H_3_4], X[H_3_4], C[H_3_4]))
        R1G1B1[H_4_5] = np.dstack((X[H_4_5], Z[H_4_5], C[H_4_5]))
        R1G1B1[H_5_6] = np.dstack((C[H_5_6], Z[H_5_6], X[H_5_6]))
        m = V - C
        RGB = R1G1B1 + np.dstack((m, m, m))                        # Adding the value correction
        return RGB

    def convert_rgb_2_hsv(self):
        fpixels = self.fpixels;
        self.pixels = Image._rgbf_2_hsvf(fpixels[:, :, 0], fpixels[:, :, 1], fpixels[:, :, 2]);
        self._colorspace=self.ColorSpaces.HSV;

    def convert_hsv_2_rgb(self):
        fpixels = self.fpixels;
        self.pixels = Image._hsvf_2_rgbf(fpixels[:, :, 0], fpixels[:, :, 1], fpixels[:, :, 2]);
        self._colorspace=self.ColorSpaces.RGB;


    def get_hsv(self):
        clone = self.get_rgb_copy();
        clone.convert_rgb_2_hsv();
        return clone;

    @classmethod
    def create_fourier_basis_image(cls, n_vectors=100):
        C = np.zeros((n_vectors, n_vectors, 3));
        ns = np.arange(n_vectors)
        one_cycle = 2 * np.pi * ns / n_vectors
        for k in range(n_vectors):
            t_k = k * one_cycle
            C[k, :,0] = np.cos(t_k)
            C[k, :,1] = np.sin(t_k)
            C[k, :,2] = 0
        return Image(pixels=C);

    @classmethod
    def get_coord_image(cls, size, normalized=True):
        if(normalized):
            y = np.linspace(0,1, size[0]);
            x = np.linspace(0,1, size[1]);
        else:
            y = np.arange(size[0]);
            x = np.arange(size[1]);
        yy,xx = np.meshgrid(y,x)
        return Image(pixels=np.dstack((yy,xx)));

    def get_coordinate_im(self, normalized=True):
        return Image.get_coord_image(self.shape, normalized=normalized);

    def plot3d(self):
        xx, yy = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        # create the figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, self.pixels ,rstride=1, cstride=1, cmap=plt.cm.gray,
                        linewidth=0)
        # show it
        plt.show()

    def get_subsampled(self, factor, phase=None):
        if(phase is None):
            p = [0,0];
        elif(isinstance(phase, int)):
            p = [phase, phase];
        else:
            p = phase;
        return Image(pixels=self.pixels[p[0]::factor, p[1]::factor]);

    @classmethod
    def get_kernel_box(cls, size=None):
        if(size is None):
            size=[3,3];
        kernel = np.ones((5,5));
        kernel = kernel/np.sum(kernel);
        return cls(pixels=kernel)

    @classmethod
    def get_kernel_gaussian(cls, size=None, std=None):
        if(size is None):
            size=[3,3];
        elif(isinstance(size, int)):
            size = [size, size];
        if(std is None):
            std=[size[0]/4, size[1]/4];
        if(not isinstance(std, (list, tuple))):
            std = [std, std];

        sigy = sp.signal.windows.gaussian(size[0], std=std[0]);
        sigx = sp.signal.windows.gaussian(size[1], std=std[1]);
        kernel = np.outer(sigy, sigx);
        kernel = kernel/np.sum(kernel);
        return cls(pixels=kernel);

    @classmethod
    def create_from_matplotlib_figure(cls, fig):
        fig.canvas.draw();
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        return cls(pixels=data);


    @classmethod
    def stack_images(cls, images, concatdim=0, **kwargs):
        matchdim = (concatdim + 1) % 2;
        newframe = images[0].clone(share_data=False).pixels;
        for vn in range(1, len(images)):
            addpart = images[vn].pixels;
            partsize = np.asarray(addpart.shape)[:];
            cumulsize = np.asarray(newframe.shape)[:];
            if (partsize[matchdim] != cumulsize[matchdim]):
                sz = partsize[:];
                sz[matchdim] = cumulsize[matchdim];
                addpart = np.array(PIM.fromarray(addpart).resize((int(sz[1]), int(sz[0]))));
            newframe = np.concatenate((newframe, addpart), concatdim);
        return cls(pixels=newframe)


    @classmethod
    def create_checker_image(cls, size=None, grid_size=None, col1=None, col2=None):
        if(size is None):
            size = [256,256,3];
        w = size[1];
        h = size[0];
        if(col1 is None):
            col1 = np.ones(3);
        if(col2 is None):
            col2 = np.zeros(3);
        im = Image(pixels=np.zeros(size).astype(float));
        if(grid_size is None):
            grid_size = int(min(w,h)/25);
        # Make pixels white where (row+col) is odd
        for i in range(w):
            for j in range(w):
                if (i//grid_size + j//grid_size)%2:
                    im.pixels[i,j] = col1
                else:
                    im.pixels[i,j] = col2
        return im;


    def convolve_with(self, kernel, mode='constant', origin = None, **kwargs):
        k = kernel;
        if(isinstance(kernel, Image)):
            k = kernel.pixels;
        if(origin is None):
            origin = [0,0];
            if((kernel.shape[0]%2)==0):
                origin[0]=-1;
            if((kernel.shape[1]%2)==0):
                origin[1]=-1;

        # self.pixels = sp.signal.convolve2d(self.pixels, k, mode=mode);
        if(self.n_color_channels==1):
            self.pixels = sp.ndimage.convolve(self.pixels, k, mode=mode, origin=origin, **kwargs);
        else:
            for c in range(self.n_color_channels):
                self.pixels[:,:,c] = sp.ndimage.convolve(self.pixels[:,:,c], k, mode=mode, origin=origin, **kwargs);


    def get_convolved_with(self, kernel, origin=None, **kwargs):
        from mpl_toolkits.mplot3d import Axes3D
        k = kernel;
        if(isinstance(kernel, Image)):
            k = kernel.pixels;
        # if(self.n_color_channels>1 and len(kernel.shape)<3):
        #     k = np.dstack([k]*self.n_color_channels);
        im = self.clone(share_data=False);
        im.convolve_with(kernel=k, origin = origin, **kwargs);
        return im;

    def get_with_tapered_alpha_boundary(self, border_width=None):
        if (border_width is 'default'):
            border_width = int(min(self.width, self.height) * 0.05);
        rval = self.get_rgba_copy().get_float_copy();
        rval.pixels[:border_width, :, 3] = np.transpose(np.tile(np.linspace(0, 1, border_width), (rval.width, 1)))
        rval.pixels[-border_width:, :, 3] = np.transpose(np.tile(np.linspace(1, 0, border_width), (rval.width, 1)))
        rval.pixels[:, :border_width, 3] = np.tile(np.linspace(0, 1, border_width), (rval.height, 1)) * rval.pixels[:,
                                                                                                        :border_width,
                                                                                                        3]
        rval.pixels[:, -border_width:, 3] = np.tile(np.linspace(1, 0, border_width), (rval.height, 1)) * rval.pixels[:,
                                                                                                         -border_width:,
                                                                                                         3]
        return rval

    def get_with_alpha_divided(self, threshold=0.05):
        tpix = self.fpixels.copy()
        talpha = self.get_alpha_as_rgb().fpixels;
        tpix[:, :, :3] = np.where(talpha > threshold, tpix[:, :, :3] / talpha, tpix[:, :, :3]);
        tpix[:, :, 3] = np.where(tpix[:, :, 3] > threshold, 1, tpix[:, :, 3])
        return Image(pixels=tpix)

    def get_with_radial_alpha(self, exponent, scale, multiply_alpha=False, min_alpha=0):
        def _get_radial_alpha(im):
            [h, w] = im.shape[:2]
            [j, i] = np.mgrid[range(h), range(w)]
            midpoint = np.array([h // 2, w // 2])
            cj = j - midpoint[0];
            ci = i - midpoint[1];
            r = np.sqrt(cj ** 2 + ci ** 2);
            r = r / r.max()
            return 1 - r;

        def _get_with_radial_alpha(im, gamma=None, shift=None):
            alpha = _get_radial_alpha(im)
            if (gamma is not None):
                alpha = np.power(alpha, gamma);
            if (shift is not None):
                alpha = np.clip(alpha * shift, 0, 1);
            alpha = alpha / alpha.max()
            alpha = np.clip(alpha, min_alpha, 1);
            rval = im.get_rgba_copy().get_float_copy();
            rval.pixels[:, :, 3] = alpha;
            return rval;

        rval = _get_with_radial_alpha(self, gamma=exponent, shift=scale)
        if (multiply_alpha):
            rval._multiply_rgb_by_alpha_channel();
        return rval;

    def _multiply_rgb_by_alpha_channel(self):
        rval_alpha = self.get_alpha_as_rgb().fpixels;
        self.pixels[:, :, :3] = self.pixels[:, :, :3] * rval_alpha;

    def splat(self, im):
        """
        Splat im onto this image
        """
        try:
            splatIm = im;
            tpix = self.fpixels.copy();
            falphc = splatIm.fpixels[:, :, 3];
            talphc = tpix[:, :, 3];
            from_alpha_im = np.dstack((falphc, falphc, falphc));
            splatpix = np.clip(splatIm.pixels[:, :, :3], 0, 1);

            self.pixels[:, :, :3] = splatpix * (from_alpha_im) + (1 - from_alpha_im) * tpix[:, :, :3];
            self.pixels[:, :, 3] = np.clip(talphc + falphc, 0, 1);
            return True;
        except ValueError as error:
            print(error);
            return None;





    ##################//--operators--\\##################
    # <editor-fold desc="operators">
    def __add__(self, other):
        if(isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.add(self.samples, other.samples));
        else:
            return self._selfclass(pixels=np.add(self.samples, other));

    def __radd__(self, other):
        return self.__add__(other);

    def __sub__(self, other):
        if(isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.subtract(self.samples, other.samples));
        else:
            return self._selfclass(pixels=np.subtract(self.samples, other));

    def __rsub__(self, other):
        if (isinstance(other, (NDArray))):
            return self._selfclass(pixels=np.subtract(other._ndarray, self._ndarray));
        elif(isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.subtract(other.pixels, self.pixels));
        else:
            return self._selfclass(pixels=np.subtract(other, self.pixels));

    def __mul__(self, other):
        if(isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.multiply(self.samples, other.samples));
        else:
            return self._selfclass(pixels=np.multiply(self.samples, other));

    def __rmul__(self, other):
        return self.__mul__(other);
    # </editor-fold>
    ##################\\--operators--//##################


    ##################//--Image Text--\\##################
    # <editor-fold desc="Image Text">
    def write_text_box(self, text, pos_xy, box_width, font_filename='RobotoCondensed-Regular.ttf',
                       font_size=None, color=(0, 0, 0), place='left',
                       justify_last_line=False):
        ImageText.write_text_box(self, xy=pos_xy, text=text, box_width=box_width, font_filename=font_filename,
                                 font_size=font_size, color=color, place=place,
                                 justify_last_line=justify_last_line);

    def write_shadowed_text(self, pos_xy, text,
                            font_size=None,
                            font_filename=None,
                            max_width=None,
                            max_height=None,
                            color=None,
                            shadow_color=None,
                            shadow_offset_xy=None,
                            encoding='utf8', draw_context=None):
        ImageText.write_shadowed_text(img=self, text=text, xy=pos_xy, font_filename=font_filename,
                                      font_size=font_size,
                                      color=color,
                                      shadow_color = shadow_color,
                                      shadow_offset_xy = shadow_offset_xy,
                                      max_width=max_width,
                                      max_height=max_height,
                                      encoding=encoding, draw_context=draw_context);

    def write_text_no_shadow(self, text, pos_xy, font_filename='RobotoCondensed-Regular.ttf',
                             font_size='fill',
                             color=(0, 0, 0),
                             max_width=None,
                             max_height=None,
                             encoding='utf8', draw_context=None):

        ImageText.write_text(img=self, xy=pos_xy, text=text, font_filename=font_filename,
                             font_size=font_size,
                             color=color,
                             max_width=max_width,
                             max_height=max_height,
                             encoding=encoding, draw_context=draw_context);

    def write_text(self, text, pos_xy, font_filename='RobotoCondensed-Regular.ttf',
                   max_width=None,
                   max_height=None,
                   color=(255, 255, 255),
                   shadow=(0,0,0),
                   shadow_offset_xy=(1,1)):

        ImageText.write_shadowed_text(img=self, text=text, xy=pos_xy, font_filename=font_filename,
                                      font_size='fill',
                                      color=color,
                                      shadow_color=shadow,
                                      shadow_offset_xy=shadow_offset_xy,
                                      max_width=max_width,
                                      max_height=max_height);

    @staticmethod
    def imshow(im, new_figure=True):
        if(isinstance(im, Image)):
            imdata = im.pixels;
        else:
            imdata = im;

        if(imdata.dtype == np.int64 or imdata.dtype == np.int32):
            imdata = imdata.astype(np.uint8)


        if  (ptlreg.apy.utils.is_notebook()):
            if (new_figure):
                plt.figure();
            if (len(imdata.shape) < 3):
                if(imdata.dtype==np.uint8):
                    nrm = matplotlib.colors.Normalize(vmin=0, vmax=255);
                else:
                    nrm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0);
                plt.imshow(imdata, cmap='gray', norm=nrm);


            elif(imdata.shape[2]==2):
                plt.imshow(Image._Flow2RGB(imdata));
            else:
                plt.imshow(imdata);  # divided by 255
            plt.axis('off');
    # </editor-fold>
    ##################\\--Image Text--//##################

##################//--Decorators--\\##################
# <editor-fold desc="Decorators">
class ImageFeature(FeatureFunction):
    def __call__(self, func):
        decorated = super(ImageFeature, self).__call__(func);
        setattr(Image, func.__name__, decorated);
        return decorated;

def ImageMethod(func):
    setattr(Image, func.__name__, func)
    return getattr(Image, func.__name__);

def ImageStaticMethod(func):
    setattr(Image, func.__name__, staticmethod(func))
    return getattr(Image, func.__name__);

def ImageClassMethod(func):
    setattr(Image, func.__name__, classmethod(func))
    return getattr(Image, func.__name__);
# </editor-fold>
##################\\--Decorators--//##################


try:
    import acimops
    @ImageMethod
    def get_bilateral_filtered(self, sigma):
        return self._selfclass(pixels=acimops.bilateral.getBilateralFiltered(self.pixels, sigma));

    @ImageMethod
    def get_cross_bilateral_filtered(self, guide, sigmas):
        gim = guide;
        if(isinstance(guide, Image)):
            gim = guide.pixels;
        im = self.pixels;
        return self._selfclass(pixels=acimops.bilateral.getCrossBilateralFiltered(im, gim, sigmas));

except ImportError:
    pass
    # ptlreg.apy.utils.AWARN("acimops not installed; won't be able to use Cython Image Ops");