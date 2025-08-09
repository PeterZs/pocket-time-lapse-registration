import os
import io
import base64
import numpy as np
from IPython.display import HTML
from ptlreg.apy.core.filepath.FilePath import *
from ptlreg.apy.core.SavesFeatures import FeatureFunction
from .MediaObject import MediaObject
from .Audio import Audio
from .Image import Image
# import moviepy.editor as mpy
# from moviepy.video.fx.all import crop as mpycrop
import imageio
import ptlreg.apy.utils
import time
from ptlreg.apy.amedia.signals.TimeSignal import TimeSignalMixin
import ptlreg.apy.amedia.media.MediaUtils as MediaUtils
from ptlreg.apy.utils import AWARN


try:
    import moviepy as mpy
except ImportError:
    warnings.warn("Failed to import moviepy. Install moviepy to use apy Video features")



class Video(TimeSignalMixin, MediaObject):
    """Video (class): A video, and a bunch of convenience functions to go with it.
        Attributes:
            sampling_rate
            sample_duration
            _samples is a VideoReader hooked up the video file
    """
    AudioClass = Audio;
    ImageClass = Image;

    @staticmethod
    def MEDIA_FILE_EXTENSIONS():
        return ['.mp4', '.avi', '.mov'];

    def __init__(self, path=None, name=None, num_frames_in_source=None, **kwargs):
        self._audio = None;
        self._reader = None;
        super(Video, self).__init__(path=path, name=name, **kwargs);
        if(path):
            self.load_file(num_frames_in_source=num_frames_in_source);

    def get_frame(self, f):
        return self.samples[int(np.round(f))];

    # @staticmethod
    def write_combined_with_audio(self, audio_path=None, audio=None, output_path=None, clip_to_video_length=True,
                                  return_vid=True, codec='libx264', bitrate=None, **kwargs):
        return MediaUtils.create_video_from_video_and_audio(audio_path=audio_path, video=self, audio=audio,
                                                            output_path=output_path, clip_to_video_length=clip_to_video_length, return_vid=return_vid, codec=codec,
                                                            bitrate=bitrate, **kwargs);


    def load_file(self, file_path=None, num_frames_in_source=None):
        if(file_path):
            self.setPath(file_path=file_path);
        if(self.file_path):
            reader =  VideoReader(self.file_path, num_frames_in_source=num_frames_in_source);
            self._samples = reader;
            self.file_meta_data = reader.meta_data;
            self.sampling_rate = reader.meta_data['fps'];
            self.unclipped_frame_shape = reader.reader.get_data(0).shape;

            # if (num_frames_in_source is not None):
                # self.num_frames_in_source = num_frames_in_source;
            self._set_num_frames_in_source(self.num_frames_in_source);

            # if the below code is not run to calculate the number of frames in the source,
            # then the error "OSError: [Errno 26] Text file busy:" can happen when you try to delete
            # videos that habe not run this...
            # if(self.num_frames_in_source is None):
            #     self.num_frames_in_source = self._reader.num_frames_in_source;

            try:
                self.audio = self.AudioClass(self.file_path);
                self.audio.name =self.name;
            except Exception:
                # ptlreg.apy.utils.AINFORM("Issue loading audio for {}".format(self.file_path));
                audio_sampling_rate = 16000;
                self.audio = self.AudioClass(samples=np.zeros(int(np.round(audio_sampling_rate*self.duration))), sampling_rate=audio_sampling_rate);
            # self.audio.clip_bounds = self.clip_bounds;
            if(self.audio.samples is None):
                # print("HERERERERERERERE!!!!!\n\n\n\n")
                AWARN("No Audio");
                audio_sampling_rate = 16000;
                self.audio = self.AudioClass(samples=np.zeros(int(np.round(audio_sampling_rate*self.duration))), sampling_rate=audio_sampling_rate);
                # print(self.audio);
                # print(self.audio.samples);

    def init_from_dictionary(self, d, load_file = True, **kwargs):
        super(Video, self).init_from_dictionary(d);
        if(self._samples is None):
            if(self.file_path is not None and load_file):
                self.load_file();

    def init_from_aobject(self, fromobject, share_data = False, init_audio=True):
        """
        This can be executed instead of init_from_dictionary
        :param fromobject:
        :return:
        """
        if(share_data):
            self._samples = fromobject._samples;
            super(Video, self).init_from_aobject(fromobject);
        else:
            super(Video, self).init_from_aobject(fromobject);
        if(fromobject.audio):
            self.audio = self.AudioClass();
            self.audio.init_from_aobject(fromobject.audio, share_data=share_data);

    # Must define _samples
    # <editor-fold desc="Property: 'samples'">
    def _get_samples(self):
        return self._reader.accessor_slice();
    def _set_samples(self, value):
        assert (False), "no setting samples directly for video";
    # </editor-fold>
    # <editor-fold desc="Property: '_samples'">
    @property
    def _samples(self):
        return self._reader;
    @_samples.setter
    def _samples(self, value):
        self._reader = value;
    # </editor-fold>

    def _get_unclipped_start_time(self):
        return 0.0;

    def _get_unclipped_duration(self):
        return self.num_frames_in_source*self.sample_duration;

    @property
    def sample_duration(self):
        if (self.sampling_rate == 0 or self.sampling_rate is None):
            return None;
        return np.true_divide(1.0, self.sampling_rate);

    @property
    def is_clipped(self):
        return False;
        # return ((self.clip_bounds is not None) and (self.clip_bounds.bound_is_active))

    def get_duration(self):
        return self._get_unclipped_duration();

    # <editor-fold desc="Property: 'file_meta_data'">
    @property
    def file_meta_data(self):
        return self._get_file_meta_data();
    def _get_file_meta_data(self):
        return self.get_info('file_meta_data');
    @file_meta_data.setter
    def file_meta_data(self, value):
        self._set_file_meta_data(value);
    def _set_file_meta_data(self, value):
        self.set_info('file_meta_data', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'unclipped_frame_shape'">
    @property
    def unclipped_frame_shape(self):
        return self._get_frame_shape();
    def _get_frame_shape(self):
        unclipped_frame_shape = self.get_info('unclipped_frame_shape');
        if (unclipped_frame_shape is None):
            self.unclipped_frame_shape = self.get_frame(0).shape;
        return self.get_info('unclipped_frame_shape');
    @unclipped_frame_shape.setter
    def unclipped_frame_shape(self, value):
        self._set_frame_shape(value);
    def _set_frame_shape(self, value):
        self.set_info('unclipped_frame_shape', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'num_frames_in_source'">
    @property
    def num_frames_in_source(self):
        """
        Total number of frames in source, as opposed to number of frames between start and end time
        :return:
        """
        return self._get_num_frames_in_source();

    def _get_num_frames_in_source(self):
        nf = self.get_info('num_frames_in_source');
        if (nf is None):
            self._set_num_frames_in_source(self._reader.num_frames_in_source)
        return self.get_info('num_frames_in_source')
    def _set_num_frames_in_source(self, value):
        self.set_info('num_frames_in_source', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'n_frames'">
    @property
    def n_frames(self):
        """
        This is the number of samples between start and end time
        :return:
        """
        return self.num_frames_in_source;
        # return self._clip_end_sample-self._clip_start_sample;
        # return self._getNFrames();
    # </editor-fold>

    # <editor-fold desc="Property: 'audio'">
    @property
    def audio(self):
        return self._get_audio();
    def _get_audio(self):
        audio = self._audio;
        if(self._audio is not None):
            if(hasattr(self, 'clip_bounds')):
                self._audio.clip_bounds = self.clip_bounds;
            return self._audio;
        else:
            return None;

    @audio.setter
    def audio(self, value):
        self._set_audio(value);
    def _set_audio(self, value):
        self._audio = value;
    # </editor-fold>

    @property
    def shape(self):
        return np.array(self.get_frame(0).pixels.shape + (self.n_frames,));

    def _get_unclipped_width(self):
        return self.unclipped_frame_shape[1];

    def _get_unclipped_height(self):
        return self.unclipped_frame_shape[0];

    def _write_video_only_to_file(self, output_path, output_sampling_rate=None, target_duration=None, frame_shape=None):
        """
        WHY IS IT SEEMING NOT TO WRITE SOME FRAMES???
        :param output_path:
        :param output_sampling_rate:
        :param target_duration:
        :param frame_shape:
        :return:
        """
        speed_factor = 1;
        if(output_sampling_rate is None):
            output_sampling_rate=self.sampling_rate;
        vwriter = VideoWriter(output_path=output_path, fps=output_sampling_rate);
        input_duration = self.duration;
        output_duration = input_duration;
        assert(not ((speed_factor is not None) and (target_duration is not None))), "do not provide both speed factor {} and duration {}".format(speed_factor, target_duration);
        if((speed_factor is not None) and (speed_factor!=1)):
            output_duration = np.true_divide(input_duration, speed_factor);
        if(target_duration is not None):
            output_duration = target_duration;
        nsamples = np.round(output_sampling_rate * output_duration);

        frame_start_times = np.linspace(0, input_duration, num=int(nsamples), endpoint=False);
        frame_index_floats = frame_start_times * self.sampling_rate;
        start_timer = time.time();
        last_timer = start_timer;
        fcounter = 0;
        for nf in range(len(frame_index_floats)):
            frame_to_write = self.samples[frame_index_floats[nf]];
            assert(frame_to_write is not None), "frame samples[{}] {}/{} for time {} is {}\nspeed_factor is {}".format(frame_index_floats[nf],nf,len(frame_index_floats)-1, frame_start_times[nf], frame_to_write, speed_factor);
            if((frame_shape is not None) and ((frame_shape[0]==self.height) and (frame_shape[1]==self.width))):
                assert(frame_shape[0] is not None or frame_shape[1] is not None), "shape was {}".format(frame_shape);
                resample_shape = frame_shape;
                if(resample_shape[0] is None):
                    resample_shape[0]=frame_to_write.height;
                if(resample_shape[1] is None):
                    resample_shape[1]=frame_to_write.width;
                frame_to_write = frame_to_write.get_scaled(resample_shape);

            vwriter.write_frame(frame_to_write);
            fcounter += 1;
            if (not (fcounter % 50)):
                if ((time.time() - last_timer) > 10):
                    last_timer = time.time();
                    print("{}%% done after {} seconds...".format(100.0 * np.true_divide(fcounter, len(frame_index_floats)),
                                                                 last_timer - start_timer));
        vwriter.close();

    def write_to_file(self, output_path, output_sampling_rate=None, **kwargs):
        assert output_path, "MUST PROVIDE OUTPUT PATH FOR VIDEO"
        if(output_sampling_rate is None):
            output_sampling_rate = self.sampling_rate;
        tempfilepath = self._get_temp_file_path(final_path=output_path);
        self._write_video_only_to_file(output_path=tempfilepath, output_sampling_rate=output_sampling_rate)
        rvid = MediaUtils.create_video_from_video_and_audio(video_path=tempfilepath, audio=self.audio, output_path=output_path);
        os.remove(tempfilepath);
        return rvid;

    @property
    def n_time_samples(self):
        """
        This is the number of samples between start and end time
        :return:
        """
        return self.num_frames_in_source;
        # return np.divide(self.duration, self.sample_duration);

    def _get_mpy_clip(self, get_audio=True):
        if(self.is_clipped):
            # effective_clip = self.effectiveClipRegion();
            clip_region = self.clip_bounds;
            clip = mpy.VideoFileClip(self.file_path, audio=get_audio).subclip(clip_region.start_time, clip_region.end_time);
            if (self.clip_bounds.space_bound_is_active):
                # clipvals = self.clip_bounds.space_region.as_numpy_array.astype(int);
                # return mpycrop(clip, width=clipvals[2], height=clipvals[3], x_center=clipvals[0], y_center=clipvals[1]);
                return mpy.fx.Crop(clip, x1=int(self.clip_spatial_origin_x), y1=int(self.clip_spatial_origin_y), width=int(self.width), height=int(self.height));
            else:
                return clip;
        else:
            return mpy.VideoFileClip(self.file_path, audio=get_audio);

    @staticmethod
    def play_video_file(path):
        vidhtml = HTML(data=Video._play_video_from_path_html(path));
        ptlreg.apy.utils.aget_ipython().display.display(vidhtml);

    def _get_string_for_html_streaming_base64(self):
        svideo = io.open(self.file_path, 'r+b').read()
        encoded = base64.b64encode(svideo)
        return "data:video/mp4;base64,{0}".format(encoded.decode('ascii'));

    @staticmethod
    def _play_video_from_path_html(path):
        video = io.open(path, 'r+b').read()
        encoded = base64.b64encode(video)
        vidhtml='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'));
        return vidhtml;

    def _guess_frame_offset(self):
        reader_n_frames = self._reader.meta_data.get('nframes');
        if(reader_n_frames is not None and (self.num_frames_in_source is not None)):
            return reader_n_frames-self.num_frames_in_source;

    def _get_play_html(self):
        if(not self.is_clipped):
            return Video._play_video_from_path_html(self.file_path);
        else:
            if((self.manager is not None) and hasattr(self.manager, 'bounce_clip_for')):
                ptlreg.apy.utils.AWARN("You should bounce the clip {} if you are going to play it".format(self.getClipRegion()));
            temp_file_name = '';
            if(self.name):
                temp_file_name=temp_file_name+self.name;
            temp_file_name = temp_file_name+str(self.getClipRegion().__hash__())+'.mp4';
            temp_path = self._get_temp_file_path(final_path = temp_file_name);
            self.write_to_file(output_path=temp_path);
            return Video._play_video_from_path_html(self.file_path);

    def play(self):
        if  (ptlreg.apy.utils.is_notebook()):
            if(not self.is_clipped):
                # print("Playing video:")
                Video.play_video_file(self.file_path);
            else:
                # print("Playing video Clip {}:".format(self.clip_bounds));
                if((self.manager is not None) and hasattr(self.manager, 'bounce_clip_for')):
                    # clip_path = self.manager.getClipPath(self);
                    # Video.PlayVideoFile(clip_path);
                    ptlreg.apy.utils.AWARN("You should bounce the clip {} if you are going to play it".format(self.getClipRegion()));
                # else:
                # print("playing as clip")
                temp_file_name = '';
                if(self.name):
                    temp_file_name=temp_file_name+self.name;
                temp_file_name = temp_file_name+str(self.get_clip_region().__hash__())+'.mp4';
                temp_path = self._get_temp_file_path(final_path = temp_file_name);
                self.write_to_file(output_path=temp_path);
                Video.play_video_file(temp_path);
                # IPython.display.display(self.get_mpy_clip().ipython_display(fps=self.sampling_rate, maxduration=self.duration + 1));
        else:
            print("HOW TO PLAY VIDEO? NOT A NOTEBOOK.")

    def load_features(self, features_to_load=None, **kwargs):
        super(Video, self).load_features(features_to_load=features_to_load, **kwargs);
        self.audio.load_features(features_to_load=features_to_load, **kwargs);

    def set_features_root(self, path):
        super(Video, self).set_features_root(path);
        self.audio.set_features_root(path);

    @classmethod
    def from_frames_in_path(cls, directory, output_path=None, extensions=None, fps=15):
        input_dir=FilePath.from_arg(directory);
        if(output_path is None):
            output_path = os.path.join(input_dir.absolute_file_path, 'video.mp4');
        if(extensions is None):
            extensions = ['.jpg', '.png', '.jpeg'];
        ims = FilePathList.from_directory(input_dir.absolute_file_path, extension_list=extensions);
        vwriter = VideoWriter(output_path=output_path, fps=fps);
        for ifn in ims:
            if(ifn.file_name[0]=='.'):
                print("skipping {}".format(ifn));
            else:
                print(ifn.absolute_file_path)
                newim = Image(ifn.absolute_file_path);
                if(newim.height%2):
                    newim.pixels = newim.pixels[1:,:,:];
                if(newim.width%2):
                    newim.pixels = newim.pixels[:,1:,:];
                vwriter.write_frame(newim.get_rgb_copy());
        vwriter.close()
        v = cls(output_path);
        return v;

    @classmethod
    def create_with_function_and_params(cls, func, output_path=None, fps=None, frame_parameters=None):
        if(output_path is None):
            output_path = apy.afileui.uiGetSaveFilePath(file_extension='.mp4');
            assert(output_path), "Must provide output path for writing video";
        if(fps is None):
            fps = 15;
            ptlreg.apy.utils.AWARN("No fps provided, using {}".format(fps));

        vw = VideoWriter(output_path = output_path, fps=fps);
        for cval in frame_parameters:
            pim = func(cval)
            vw.write_frame(pim);
        vw.close();
        v = Video(output_path);
        return v;

class VideoFeature(FeatureFunction):
    def __call__(self, func):
        decorated = super(VideoFeature, self).__call__(func);
        setattr(Video, func.__name__, decorated);
        return decorated;

def VideoMethod(func):
    setattr(Video, func.__name__, func)
    return getattr(Video, func.__name__);

def VideoStaticMethod(func):
    setattr(Video, func.__name__, staticmethod(func))
    return getattr(Video, func.__name__);

def VideoClassMethod(func):
    setattr(Video, func.__name__, classmethod(func))
    return getattr(Video, func.__name__);

# <editor-fold desc="VideoWriter">
class VideoWriter(object):
    def __init__(self, output_path, fps):
        self._writer = None;
        self.open(output_path=output_path, fps=fps);

    def open(self, output_path, fps=None):
        assert (self._writer is None), "Tried to open videowriter when it is already open."
        if (fps is None):
            fps = self.sampling_rate;
        self._writer = imageio.get_writer(output_path, 'ffmpeg', macro_block_size=None, fps=fps);

    def close(self):
        self._writer.close();
        self._writer = None;

    def write_frame(self, img):
        assert(self._writer and (not self._writer.closed)), 'ERROR: Vid writer object is closed.';
        if(isinstance(img, Image)):
            self._writer.append_data(img.ipixels);
        else:
            self._writer.append_data(img.astype(np.uint8))
# </editor-fold>

# <editor-fold desc="VideoReader">
class VideoReader(object):
    '''
    accessor_slice(...) returns an accessor which can be indexed like video.samples[]

    '''
    def __init__(self, file_path, num_frames_in_source=None):
        self.reader = imageio.get_reader(file_path, 'ffmpeg');
        self.file_path = file_path;
        self.num_frames_in_source = num_frames_in_source;
        self._meta_data = self.reader.get_meta_data();
        # self.clip_bounds = clip_bounds;
    # <editor-fold desc="Property: 'meta_data'">
    @property
    def meta_data(self):
        return self.reader.get_meta_data();
    # </editor-fold>

    @property
    def num_frames_in_source(self):
        if(self._num_frames_in_source is not None):
            return self._num_frames_in_source;
        else:
            self._num_frames_in_source = self.calc_num_valid_frames();
            return self._num_frames_in_source;

    @num_frames_in_source.setter
    def num_frames_in_source(self, value):
        self._num_frames_in_source = value;

    class FrameAccessor(object):
        def __init__(self, vreader, clip_slice, clip_region = None):
            self.vreader = vreader;
            self.clip_slice = clip_slice;
            self.clip_bounds = clip_region;

        def __getitem__(self, i):
            if(i<0):
                return None;
            ri = i;
            if(self.clip_slice is not None):
                ri = i+self.clip_slice.start;
            if(ri>self.clip_slice.stop):
                return None;
            ri = int(np.round(ri));
            rim = Video.ImageClass(pixels = self.vreader.get_data(ri));
            rim.clip_bounds = self.clip_bounds;
            return rim;

    def __getitem__(self, i):
        if(i<0):
            return None;
        return self.reader.get_data(i);

    def accessor_slice(self, clip_slice=None, clip_region = None):
        if(clip_slice is None):
            clip_slice=slice(0, self.num_frames_in_source);
        return VideoReader.FrameAccessor(vreader = self.reader, clip_slice=clip_slice, clip_region=clip_region);

    def get_metadata_nframes_estimate(self):
        """
        Estimate number of frames from metadata
        :return:
        """
        reader_length = self.reader.get_length();
        if (reader_length == np.inf or reader_length is None):
            if('duration' in self.meta_data):
                return int(np.ceil(self._meta_data['duration']*self._meta_data['fps']));
            else:
                return 0
        return reader_length;



    def calc_num_valid_frames(self):
        file_name = FilePath(self.file_path).file_name;
        ptlreg.apy.utils.AINFORM("Loading {}...".format(file_name))
        valid_frames = 1

        # Old version used get_length, was replaced with count_frames()... will this still work in python 2.7?
        # reader_length = self.reader.get_length();
        # abepy.utils.AWARN("Reader length type={}\nValue is {}".format(type(reader_length), reader_length));

        # This is for python 2.7 vs python 3.7 compatibility
        reader_length = None;
        if(hasattr(self.reader, 'count_frames')):
            try:
                reader_length = self.reader.count_frames();
            except RuntimeError as e:
                print(e);
        if(reader_length is None):
            reader_length = self.get_metadata_nframes_estimate();

        previous_frame_guess = int(reader_length);

        # This is apparently important, because missing on first guess seems to close the door?
        if (previous_frame_guess > 3):
            previous_frame_guess = previous_frame_guess - 3

        search_direction = 1;
        try:
            self.reader.get_data(previous_frame_guess);
        except Exception as e:
            AWARN("POTENTIAL ISSUE READING VIDEO: {}".format(e));
            search_direction = -1;

        previous_frame_guess = previous_frame_guess+search_direction;
        lastReached = False;
        while(not lastReached):
            try:
                self.reader.get_data(previous_frame_guess);
                if(search_direction<0):
                    # this is the first frame we read while scanning backward
                    lastReached = True;
                else:
                    # still haven't fallen off cliff searching forward
                    previous_frame_guess = previous_frame_guess + search_direction;
            # except imageio.core.format.CannotReadFrameError as e:
            except Exception as e:
                # print("ERROR: ", e)
                # AWARN("could not read frame {}; Error {}".format(previous_frame_guess, e));
                if(search_direction<0):
                    # Still not a valid frame number
                    previous_frame_guess = previous_frame_guess+search_direction;
                else:
                    lastReached = True;
                    # Fell off cliff. Backtrack one.
                    previous_frame_guess = previous_frame_guess - 1

        lastFrameReached = previous_frame_guess
        return lastFrameReached+1;

    # def calcNumValidFrames(self):
    #     file_name = FilePath(self.file_path).file_name;
    #     ptlreg.apy.utils.AINFORM("Loading {}...".format(file_name))
    #     valid_frames = 1
    #
    #     # Old version used get_length, was replaced with count_frames()... will this still work in python 2.7?
    #     # reader_length = self.reader.get_length();
    #     # abepy.utils.AWARN("Reader length type={}\nValue is {}".format(type(reader_length), reader_length));
    #
    #     # This is for python 2.7 vs python 3.7 compatibility
    #     if(hasattr(self.reader, 'count_frames')):
    #         reader_length = self.reader.count_frames();
    #     else:
    #         reader_length = self.reader.get_length();
    #
    #     lastFrame = int(reader_length);
    #     lastReached = False;
    #     while(lastFrame>0 and (not lastReached)):
    #         try:
    #             self.reader.get_data(lastFrame);
    #             print("Last Frame Found was {}".format(lastFrame));
    #             lastReached = True;
    #         # except imageio.core.format.CannotReadFrameError as e:
    #         except Exception as e:
    #             print(e)
    #             print("could not read frame {}".format(lastFrame));
    #             lastFrame = lastFrame-1;
    #     return lastFrame
# </editor-fold>