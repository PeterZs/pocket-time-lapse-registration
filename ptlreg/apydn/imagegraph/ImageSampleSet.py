from ptlreg.apy.amedia import VideoWriter
from ptlreg.apy.amedia.media import Image
from ptlreg.apy.core import AObjectOrderedSet
from ptlreg.apydn.datanode.datasample.MapsToDataNodeSetMixin import MapsToDataNodeSetMixin
from ptlreg.apydn.datanode.filedatanode import FileDataNodeSet
from .HasImageSampleLabelsMixin import HasImageSampleLabelsMixin
from .ImageSample import ImageSample
import numpy as np

from ptlreg.apydn.datanode import DataNode, DataNodeSet
from ptlreg.apydn.datanode.datasample.DataSampleSet import DataSampleSetBase, DataSampleSetMixin


class ImageSampleSetMixin(HasImageSampleLabelsMixin, DataSampleSetMixin):
    ElementClass = ImageSample;
    # @classmethod
    # def FromNodeSet(cls, node_set):
    #     return cls(node_set.getSampleList());


    # def getByFileName(self, key):
    #     return self._main_index_map.get(key);

    # <editor-fold desc="BInfo Property: 'median_image'">
    def get_basic_median_image(self):
        pix = [];
        for a in self:
            aim = a.GetImage().get_rgba_copy();
            pix.append(aim.pixels);
        stack = np.stack(pix);
        composed = np.median(stack, axis=0).astype(dtype='uint8')
        medianIm = Image(pixels=composed);
        return medianIm;

    # </editor-fold>

    def composite_image_column_strips(self, filter_size=None, filter_sigma=None, rotate90x=None):
        images = [x.GetImage() for x in self];
        if (rotate90x):
            for i in images:
                i.rotate90x(x=rotate90x);
        return Image.MakeColumnComposite(images, filter_size=filter_size, filter_sigma=filter_sigma);

    def write_gaussian_filtered_video(self, output_path, frames_per_sample=5, sigma=1, fps=30, x_range=None, y_range=None,
                                      alpha_blur=10):
        """
        filter function should take a distance and return a weight that will be normalized
        """

        example_image = self[0].GetImage().get_float_copy();
        if (x_range is not None):
            x_start, x_stop = x_range;
            if (x_start is None):
                x_start = 0;
            if (x_stop is None):
                x_stop = example_image.shape[1];
            x_range = [x_start, x_stop];

        if (y_range is not None):
            y_start, y_stop = y_range;
            if (y_start is None):
                y_start = 0;
            if (y_stop is None):
                y_stop = example_image.shape[0];
            y_range = [y_start, y_stop];

        def crop(pixels):
            rpix = pixels;
            if (x_range is not None):
                rpix = rpix[:, x_range[0]:x_range[1]];
            if (y_range is not None):
                rpix = rpix[y_range[0]:y_range[1], :];
            return rpix;

        example_pixels = crop(example_image.pixels);
        vw = VideoWriter(output_path, fps=fps);
        n_samples = len(self);
        n_frames = n_samples * frames_per_sample;
        window_radius = int(np.round(3 * sigma));
        frame_values = np.linspace(0, n_samples - 1, n_frames);
        var = sigma ** 2;

        for fi in range(n_frames):
            frame_val = frame_values[fi];
            cur = int(np.round(frame_val));
            frame = np.zeros_like(example_pixels);
            sum_weights = 0;
            for ni in range(max(0, cur - window_radius), min(cur + window_radius, len(self) - 1)):
                framen = crop(self[ni].GetImage().fpixels);
                diff = frame_val - ni;
                nw = np.exp(-0.5 * (diff ** 2) / var);
                frame = frame + framen * nw;
                sum_weights = sum_weights + nw;
            if (sum_weights > 0):
                frame = frame / sum_weights;
            frame_image = Image(pixels=np.clip(frame, 0, 1));
            frame_image = frame_image.GetRGBCopy().GetUIntCopy();
            vw.writeFrame(frame_image);
        vw.close();
        return output_path;

    def write_bilateral_filtered_video(self, output_path, frames_per_sample=5, sigma=2, value_sigma=0.2, fps=30,
                                       x_range=None, y_range=None, image_scale=0.1):
        """
        filter function should take a distance and return a weight that will be normalized
        """

        # we don't want to normalize by super small values around the boundary of panoramas where alpha is between zero and 1
        boundary_threshold = 0.1

        def scaled(im):
            return im.GetScaledByFactor(image_scale)

        example_image = scaled(self[0].GetImage()).get_float_copy();

        if (x_range is not None):
            x_start, x_stop = x_range;
            if (x_start is None):
                x_start = 0;
            if (x_stop is None):
                x_stop = example_image.shape[1];
            x_range = [x_start, x_stop];

        if (y_range is not None):
            y_start, y_stop = y_range;
            if (y_start is None):
                y_start = 0;
            if (y_stop is None):
                y_stop = example_image.shape[0];
            y_range = [y_start, y_stop];

        def crop(pixels):
            rpix = pixels;
            if (x_range is not None):
                rpix = rpix[:, x_range[0]:x_range[1]];
            if (y_range is not None):
                rpix = rpix[y_range[0]:y_range[1], :];
            return rpix;

        def get_frame(n):
            """
            convenience function to get smaller version of frame
            :param n:
            :return:
            """
            fr = crop(scaled(self[n].GetImage()).fpixels)
            fr[:, :, 3] = np.where(fr[:, :, 3] < 1, 0, fr[:, :, 3])
            return fr

        example_pixels = get_frame(0)
        vw = VideoWriter(output_path, fps=fps);
        n_samples = len(self);
        n_frames = n_samples;
        window_radius = int(np.round(3 * sigma));
        frame_values = np.linspace(0, n_samples - 1, n_frames);
        # frame_values = np.linspace(0,n_samples-1, n_frames);
        var = sigma ** 2;

        # iterate over frames
        for fi in range(n_frames):
            frame_val = frame_values[fi];
            cur = int(np.round(frame_val));
            frame = np.zeros_like(example_pixels);
            sum_weights = np.zeros_like(example_pixels[:, :, 1]);

            framefi = get_frame(fi)
            # iterate over neighborhood
            for ni in range(max(0, cur - window_radius), min(cur + window_radius, len(self) - 1)):
                framen = get_frame(ni)

                # calculate time difference
                diff_t = frame_val - ni;

                # calculate per-pixel differences
                diffim = np.linalg.norm(framefi[:, :, :3] - framen[:, :, :3], axis=2)
                diffim = np.where(framefi[:, :, 3] < boundary_threshold, 0, diffim)

                # get gaussian falloff with value differences
                diff_v = np.exp(-0.5 * (np.power(diffim, 2)) / value_sigma) * framen[:, :, 3]

                # get gaussian falloff with time
                nw = np.exp(-0.5 * (diff_t ** 2) / var);

                framen *= nw * np.repeat(diff_v[:, :, np.newaxis], 4, axis=2)
                frame = frame + framen
                sum_weights = sum_weights + framen[:, :, 3];
            sumweights_broadcast = np.repeat(sum_weights[:, :, np.newaxis], 3, axis=2)
            frame = np.where(sumweights_broadcast > boundary_threshold, frame[:, :, :3] / sumweights_broadcast, 0)
            frame_image = Image(pixels=np.clip(frame, 0, 1));
            frame_image = frame_image.GetRGBCopy().GetUIntCopy();
            vw.writeFrame(frame_image);
        vw.close();
        return output_path;

    def write_iir_filtered_video(self, output_path, attack=0.8, decay=0.8, frames_per_sample=5, fps=30, x_range=None,
                                 y_range=None, use_thumbnails=False):
        """
        filter function should take a distance and return a weight that will be normalized
        """

        if (use_thumbnails):
            example_image = self[0].get_thumbnail_image().get_float_copy();
        else:
            example_image = self[0].GetImage().get_float_copy();
        if (x_range is not None):
            x_start, x_stop = x_range;
            if (x_start is None):
                x_start = 0;
            if (x_stop is None):
                x_stop = example_image.shape[1];
            x_range = [x_start, x_stop];

        if (y_range is not None):
            y_start, y_stop = y_range;
            if (y_start is None):
                y_start = 0;
            if (y_stop is None):
                y_stop = example_image.shape[0];
            y_range = [y_start, y_stop];

        def crop(pixels):
            rpix = pixels;
            if (x_range is not None):
                rpix = rpix[:, x_range[0]:x_range[1]];
            if (y_range is not None):
                rpix = rpix[y_range[0]:y_range[1], :];
            return rpix;

        example_pixels = Image(pixels=crop(example_image.pixels)).get_rgba_copy();
        current_frame = Image.Zeros(example_pixels.shape);
        vw = VideoWriter(output_path, fps=fps);
        n_samples = len(self);
        n_frames = n_samples * frames_per_sample;
        for next_sample in self:
            if (use_thumbnails):
                new_frame = next_sample.get_thumbnail_image().get_rgba_copy().get_float_copy();
            else:
                new_frame = next_sample.GetImage().get_rgba_copy().get_float_copy();
            for i in range(frames_per_sample):
                current_frame = current_frame * decay;
                current_frame.AdditiveSplat(new_frame * attack);
                next_frame = current_frame.GetWithAlphaDivided();
                vw.writeFrame(next_frame.GetRGBCopy());
        vw.close();
        return output_path;

    def write_simple_spat_frames_video(self, output_path, fps=30, x_range=None, y_range=None, use_thumbnails=False,
                                       n_frames_to_write=None):
        """
        filter function should take a distance and return a weight that will be normalized
        """

        if (n_frames_to_write is None):
            n_frames_to_write = len(self)
        if (use_thumbnails):
            example_image = self[0].get_thumbnail_image().get_float_copy();
        else:
            example_image = self[0].GetImage().get_float_copy();
        if (x_range is not None):
            x_start, x_stop = x_range;
            if (x_start is None):
                x_start = 0;
            if (x_stop is None):
                x_stop = example_image.shape[1];
            x_range = [x_start, x_stop];

        if (y_range is not None):
            y_start, y_stop = y_range;
            if (y_start is None):
                y_start = 0;
            if (y_stop is None):
                y_stop = example_image.shape[0];
            y_range = [y_start, y_stop];

        def crop(pixels):
            rpix = pixels;
            if (x_range is not None):
                rpix = rpix[:, x_range[0]:x_range[1]];
            if (y_range is not None):
                rpix = rpix[y_range[0]:y_range[1], :];
            return rpix;

        example_pixels = Image(pixels=crop(example_image.pixels)).get_rgba_copy();
        current_frame = Image.Zeros(example_pixels.shape);
        vw = VideoWriter(output_path, fps=fps);
        n_samples = len(self);

        n_frames = n_samples;
        n_written = 0;
        for next_sample in self:
            # if(n_written>len(self)):
            if (n_written > n_frames_to_write):
                break;
            if (use_thumbnails):
                new_frame = next_sample.get_thumbnail_image().get_rgba_copy().get_float_copy();
            else:
                new_frame = next_sample.GetImage().get_rgba_copy().get_float_copy();
            current_frame.splat(new_frame);
            # next_frame = current_frame.GetWithAlphaDivided();
            # current_frame
            vw.writeFrame(current_frame.GetRGBCopy());
            n_written = n_written + 1;
            if (n_written % 10 == 0):
                print(f"Written {n_written} frames to video {output_path}");
        vw.close();
        return output_path;


class ImageSampleSet(ImageSampleSetMixin, DataSampleSetBase):
    DATANODE_MAP_TYPE = DataNodeSet;
    ElementClass = ImageSample;
    SUBSET_CLASS = None;

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        """
        if ((self.node_id is None)):
            self.set_node_id(DataNode.generate_node_id());




# ImageSampleSet.DATANODE_SET_MAP_TYPE = ImageSampleSet;
ImageSampleSet.SUBSET_CLASS = ImageSampleSet;
# ImageSampleSetMixin.SampleSetType = ImageSampleSet;
