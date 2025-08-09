import ptlreg.apy.utils
from ptlreg.apy.amedia.media.examplefiles import *
from .MediaObject import *
from IPython.display import HTML
from scipy.io.wavfile import write
import base64
import librosa
import librosa.display
import audioread

import numpy as np
import scipy as sp

import copy

import matplotlib
import matplotlib.pyplot as plt
from ptlreg.apy.core.SavesFeatures import FeatureFunction

from ptlreg.apy.amedia.signals import TimeSignalMixin

class Audio(TimeSignalMixin, MediaObject):
    DEFAULT_HOP_LENGTH = 512;
    DEFAULT_TEMPOGRAM_WINDOW_SECONDS = 5;
    USING_PYRUBBERBAND=False;

    @staticmethod
    def MEDIA_FILE_EXTENSIONS():
        return ['.wav','.mp3','.ogg'];

    def __init__(self, path=None, sampling_rate=None, samples=None, name=None,**kwargs):
        # super(Audio, self).__init__(path=path, sampling_rate=sampling_rate, samples=samples, name=name, **kwargs);
        super(Audio, self).__init__(path=path, name=name, sampling_rate=sampling_rate, samples=samples, **kwargs);
        # self._init_samples(samples=samples, sampling_rate = sampling_rate);

        if(self.n_channels is not None):
            assert (self.n_channels < 3), "Audio with {} channels not yet supported".format(self.n_channels);
        if (path):
            self.load_file();
        if (self.name is None):
            self.name = self.file_name;

    def _init_samples(self, samples, sampling_rate = None):
        self._original_samples = samples;
        self.reset_mono_using_multi_channel();
        if(sampling_rate is not None):
            self.sampling_rate = sampling_rate;
            self.origin_sampling_rate = sampling_rate;

    @property
    def original_samples(self):
        return self._original_samples;

    @original_samples.setter
    def original_samples(self, value):
        self._original_samples = value;

    @property
    def original_sampling_rate(self):
        return self.get_info('original_sampling_rate');

    @original_sampling_rate.setter
    def original_sampling_rate(self, value):
        self.set_info('original_sampling_rate', value);

    @property
    def n_channels(self):
        if (self._original_samples is not None):
            if (len(self._original_samples.shape) == 1):
                return 1;
            else:
                # return self._original_samples.shape[len(self._original_samples.shape)-1];
                return self._original_samples.shape[0];
        else:
            return None;

    @classmethod
    def _convert_to_mono_samples(cls, samples):
        if(samples is None):
            return None;
        if (len(samples.shape) == 1):
            return samples.copy();
        else:
            return np.mean(samples, axis=0);

    @property
    def samples(self):
        return self._samples;

    @samples.setter
    def samples(self, value):
        self._samples=value;

    @property
    def n_samples(self):
        return len(self.samples);

    @property
    def audio(self):
        return self._audio;

    @property
    def _audio(self):
        return self;

    def load_file(self, file_path=None):
        if(file_path):
            self.set_file_path(file_path=file_path);
        if(self.get_file_path()):
            if(self.file_ext=='.mp4'):
                # TODO: Should probably get rid of this hack and use something more general.
                x, sr = Audio.__audioread_load(self.get_file_path());
            else:
                x, sr = librosa.load(self.get_file_path(), sr=None, mono=False);
            self._init_samples(samples=x, sampling_rate = sr);

    @classmethod
    def __audioread_load(cls, path,
                         offset: float = 0.0,
                         duration= None,
                         dtype=np.float32
                         ):
        """Load an audio buffer using audioread.Taken from librosa to byepass warning
        :param path:
        This loads one block at a time, and then concatenates the results.
        :param offset:
        :param duration:
        :param dtype:
        :return:
        """
        buf = []

        if isinstance(path, tuple(audioread.available_backends())):
            # If we have an audioread object already, don't bother opening
            reader = path
        else:
            # If the input was not an audioread object, try to open it
            reader = audioread.audio_open(path)

        with reader as input_file:
            sr_native = input_file.samplerate
            n_channels = input_file.channels

            s_start = int(sr_native * offset) * n_channels

            if duration is None:
                s_end = np.inf
            else:
                s_end = s_start + (int(sr_native * duration) * n_channels)

            n = 0

            for frame in input_file:
                frame = librosa.util.buf_to_float(frame, dtype=dtype)
                n_prev = n
                n = n + len(frame)

                if n < s_start:
                    # offset is after the current frame
                    # keep reading
                    continue

                if s_end < n_prev:
                    # we're off the end.  stop reading
                    break

                if s_end < n:
                    # the end is in this frame.  crop.
                    frame = frame[: int(s_end - n_prev)]  # pragma: no cover

                if n_prev <= s_start <= n:
                    # beginning is in this frame
                    frame = frame[(s_start - n_prev):]

                # tack on the current frame
                buf.append(frame)

        if buf:
            y = np.concatenate(buf)
            if n_channels > 1:
                y = y.reshape((-1, n_channels)).T
        else:
            y = np.empty(0, dtype=dtype)

        return y, sr_native

    @classmethod
    def stack_channels(cls, *channels):
        readytostack = [];
        for c in channels:
            assert(len(c.shape)==1), "Trying to stack a channel with shape {}".format(c.shape);
            readytostack.append(c.reshape(1,len(c)));
        return np.concatenate((readytostack), axis=0);


    def write_to_file(self, output_path=None, n_channels=None, **kwargs):
        assert(output_path), "must provide path to save audio."
        if(n_channels is None):
            n_channels = self.n_channels;
        assert (n_channels<3), "Support for more than 2 audio channels not yet implemented";
        if(n_channels == 2):
            signal_out =  self.stereo_copy.T;
        else:
            signal_out = self.samples;
        write(output_path, self.sampling_rate, signal_out);
        # data = self.get_samples();
        # scaled = np.int16(data/np.max(np.abs(data)) * 32767)
        # if(output_sampling_rate is None):
        #     output_sampling_rate=44100
        # write(output_path, output_sampling_rate, scaled)

    @property
    def stereo_copy(self):
        # if (self.n_channels == 2 and (not (check_modified and (not self.signal_modified)))):
        if (self.n_channels == 2):
            return copy.deepcopy(self.original_samples);
        else:
            # yes, this does return a copy...
            return np.stack((self.samples, self.samples), axis=0);

    @property
    def _stereo_copy(self):
        # if (self.n_channels == 2 and (not self.signal_modified)):
        if (self.n_channels == 2):
            return copy.deepcopy(self._original_samples);
        else:
            # yes, this does return a copy...
            return np.stack((self._samples, self._samples), axis=0);

    def get_original_channel(self, channel):
        return self.original_samples[channel,:];
    def _get_original_channel(self, channel):
        return self._original_samples[channel,:];

    def to_dictionary(self):
        d = super(Audio, self).to_dictionary();
        return d;

    def init_from_aobject(self, fromobject, share_data = False):
        if(share_data):
            self._original_samples = fromobject._original_samples;
            self._samples = fromobject._samples;
            self._init_from_aobject_no_samples(fromobject);
        else:
            self._init_from_aobject_no_samples(fromobject);
            if(fromobject._samples is not None):
                self._original_samples = copy.deepcopy(fromobject._original_samples);
                self._samples = copy.deepcopy(fromobject._samples);

    def init_from_dictionary(self, d, load_file = True, **kwargs):
        super(Audio, self).init_from_dictionary(d);
        if(self._original_samples is None):
            if(self.file_path is not None and load_file):
                self.load_file();

    @classmethod
    def silence(cls, duration, sampling_rate=44000, n_channels=2, name=None):
        if(n_channels==1):
            sig = np.zeros(int(np.ceil(duration * sampling_rate)));
        else:
            sig = np.zeros((n_channels,int(np.ceil(duration * sampling_rate))));
        a = cls(samples=sig, sampling_rate=sampling_rate, name=name);
        return a;

    @classmethod
    def ping_sound(cls, duration=None, freqs=None, damping=None, n_channels= 2, sampling_rate = 16000):
        ping = cls.PingSignal(duration=duration, freqs=freqs, damping=damping, sampling_rate=sampling_rate);
        s = ping.samples;
        if(n_channels>1):
            snew = s;
            for c in range(n_channels-1):
                snew = Audio.stack_channels(snew, s);
            s = snew;
        sa = cls(samples=s, sampling_rate=sampling_rate, name = 'ping');
        sa.setMaxAbsValue(1.0);
        return sa;

    def play(self, normalize=False, autoplay=None):
        # self.playStereo();
        if (ptlreg.apy.utils.is_notebook()):
            if(normalize):
                audiodisp = ptlreg.apy.utils.aget_ipython().display.Audio(data=self.samples, rate=self.sampling_rate, autoplay=autoplay);
                ptlreg.apy.utils.aget_ipython().display.display(audiodisp);
            else:
                htmlstr = self._get_string_for_html_streaming_base64(stereo=False);
                ahtml = HTML(data='''<audio alt="{}" controls><source src="{}" type="audio/wav" /></audio>'''.format(self.name, htmlstr));
                ptlreg.apy.utils.aget_ipython().display.display(ahtml);
        else:
            print("Can't play audio with audio.play() outside of notebook");

    def _get_play_html(self, autoplay=None):
        audiodisp = ptlreg.apy.utils.aget_ipython().display.Audio(data=self.stereo_copy, rate=self.sampling_rate, autoplay=autoplay);
        return audiodisp._repr_html_();

    def play_stereo(self, normalize=False, autoplay = None):
        if (ptlreg.apy.utils.is_notebook()):
            if(normalize):
                audiodisp = ptlreg.apy.utils.aget_ipython().display.Audio(data=self.stereo_copy, rate=self.sampling_rate, autoplay=autoplay);
                ptlreg.apy.utils.aget_ipython().display.display(audiodisp);
            else:
                htmlstr = self._get_string_for_html_streaming_base64()
                ahtml = HTML(data='''<audio alt="{}" controls><source src="{}" type="audio/wav" /></audio>'''.format(self.name, htmlstr));
                ptlreg.apy.utils.aget_ipython().display.display(ahtml);
        else:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=self.n_channels,
                            rate=self.sampling_rate,
                            output=True,
                            output_device_index=1
                            )
            stream.write(self.get_samples())
            stream.stop_stream();
            stream.close()
            p.terminate()

    def plot_waveform(self, **kwargs):
        plt.figure(figsize=(14, 5))
        self.plot(**kwargs);

    def _clip_to_time(self, start=None, end=None):
        if(start is None):
            start_index = 0;
        else:
            start_index=int(np.round(start*self.sampling_rate));
        if(end is None):
            self._samples=self.samples[start_index:];
        else:
            self._samples=self.samples[start_index:int(np.round(end*self.sampling_rate))];

    def get_clip(self, start=None, end=None):
        clone = self.clone(share_data=False);
        clone._clip_to_time(start=start, end=end);
        return clone;



    def _is_unmodified(self):
        x = Audio._convert_to_mono_samples(self._original_samples);
        return np.array_equal(x, self._samples);

    def _get_string_for_html_streaming_base64(self, stereo=True):
        if(stereo):
            encoded = self.get_stereo_encoded_base64_wav();
            return "data:audio/wav;base64,{0}".format(encoded.decode('ascii'));
        else:
            encoded = self.get_mono_encoded_base64_wav();
            return "data:audio/wav;base64,{0}".format(encoded.decode('ascii'));

    @property
    def stereo_sampling_rate(self):
        if (self.n_channels == 2):
            if(self.original_sampling_rate is not None):
                return self.original_sampling_rate;
        return self.sampling_rate;

    def get_stereo_encoded_base64_wav(self):
        sig = self.stereo_copy;
        sr = self.stereo_sampling_rate
        if (sig is None):
            sig = self.samples;
            sr = self.sampling_rate;
        saudio = _make_wav(sig, sr);
        encoded = base64.b64encode(saudio);
        return encoded;

    def get_mono_encoded_base64_wav(self):
        # sig = self.getSignal();
        sig = self.samples;
        sr = self.sampling_rate;
        saudio = _make_wav(sig, sr);
        encoded = base64.b64encode(saudio);
        return encoded;

    def reset_mono_using_multi_channel(self):
        self._samples = self._convert_to_mono_samples(self._original_samples);

    ##################//--Features--\\##################
    # <editor-fold desc="Features">
    @FeatureFunction(feature_name='spectrogram', save_result=False)
    def get_spectrogram(self, power=2, **kwargs):
        """
        :param power: 1 for energy spectrogram, 2 for power spectrogram by default
        :param kwargs:
        :return:
        """
        spec = librosa.stft(self.audio.samples, **kwargs);
        D = np.abs(spec) ** power;
        return D;


    @FeatureFunction('melspectrogram', save_result=False)
    def get_mel_spectrogram(self, n_mel=None, **kwargs):
        if (n_mel is not None):
            return librosa.feature.melspectrogram(y=self.audio.samples, sr=self.audio.sampling_rate, n_mel=n_mel, **kwargs);
        else:
            return librosa.feature.melspectrogram(S=self.audio.get_spectrogram(), **kwargs);
    # </editor-fold>
    ##################\\--Features--//##################

    ##################//--Visualization--\\##################
    # <editor-fold desc="Visualization">
    def show_spectrogram(self, title=None, fig = None, force_recompute=True, hop_length=None, time_range=None, frequency_range=None, gamma=None, linear_frequency=False, ax=None, **kwargs):
        if (hop_length is None):
            hop_length = self._default_hop_length;
            # hop_length = self.DEFAULT_HOP_LENGTH;

        if(fig is None):
            fig = plt.figure();
        S = self.get_spectrogram(hop_length=hop_length, power=1, force_recompute=force_recompute, **kwargs);
        pwr = librosa.amplitude_to_db(S, ref=np.max)
        if(gamma is not None):
            print(pwr)
            pwr = np.power(pwr,gamma);
            print(pwr)

        if(ax is None):
            ax = plt.gca();

        lrargs = dict(sr=self.sampling_rate, hop_length=hop_length,
                      y_axis='log', x_axis='time', ax=ax);
        if(linear_frequency):
            lrargs['y_axis']='linear';
        if(ax is not None):
            lrargs['ax']=ax;

        librosa.display.specshow(pwr, **lrargs)
        if(title is None):
            plt.title('Power spectrogram')
        else:
            plt.title(title);
        # plt.colorbar(format='%+2.0f dB')

        if (frequency_range is not None):
            plt.ylim(frequency_range);
        plt.xlabel('Time (s)')
        if (time_range is not None):
            plt.xlim(time_range);
        plt.tight_layout()
        return fig;

    def show_mel_spectrogram(self, hop_length=None, time_range=None, force_recompute=False):
        if (hop_length is None):
            hop_length = self.DEFAULT_HOP_LENGTH;
        S = self.get_mel_spectrogram(hop_length=hop_length, force_recompute=force_recompute);
        lamp_S = librosa.amplitude_to_db(S, ref=np.max);
        # Make a new figure
        f = plt.figure();
        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(lamp_S, sr=self.sampling_rate, x_axis='time', y_axis='mel')

        if (not (time_range is None)):
            plt.xlim(time_range);

        # Put a descriptive title on the plot
        plt.title('Mel Spectrogram')
        # # draw a color bar
        # plt.colorbar(format='%+02.0f dB')
        # # Make the figure layout compact
        plt.tight_layout()
        return f;

    # </editor-fold>
    ##################\\--Visualization--//##################

class AudioFeature(FeatureFunction):
    """
    Decorator to mark a function as an audio feature.
    """
    def __call__(self, func):
        decorated = super(AudioFeature, self).__call__(func);
        setattr(Audio, func.__name__, decorated);
        return decorated;

def AudioMethod(func):
    """
    Decorator to mark a function as an audio method.
    :param func:
    :return:
    """
    setattr(Audio, func.__name__, func)
    return getattr(Audio, func.__name__);

def AudioStaticMethod(func):
    """
    Decorator to mark a function as an audio static method.
    :param func:
    :return:
    """
    setattr(Audio, func.__name__, staticmethod(func))
    return getattr(Audio, func.__name__);

def AudioClassMethod(func):
    """
    Decorator to mark a function as an audio class method.
    :param func:
    :return:
    """
    setattr(Audio, func.__name__, classmethod(func))
    return getattr(Audio, func.__name__);


def _make_wav(data, rate, normalize=False):
    """ Transform a numpy array to a PCM bytestring """
    import struct
    from io import BytesIO
    import wave
    try:
        import numpy as np

        data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            nchan = 1
        elif len(data.shape) == 2:
            # In wave files,channels are interleaved. E.g.,
            # "L1R1L2R2..." for stereo. See
            # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
            # for channel ordering
            nchan = data.shape[0]
            data = data.T.ravel()
        else:
            raise ValueError('Array audio input must be a 1D or 2D array')
        # scaled = np.int16(data / np.max(np.abs(data)) * 32767).tolist()
        if(normalize):
            scaled = np.int16(data / np.max(np.abs(data)) * 32767).tolist()
        else:
            scaled = np.int16(data * 32767).tolist()
    except ImportError:
        # check that it is a "1D" list
        idata = iter(data)  # fails if not an iterable
        try:
            iter(idata.next())
            raise TypeError('Only lists of mono audio are '
                            'supported if numpy is not installed')
        except TypeError:
            # this means it's not a nested list, which is what we want
            pass
        maxabsvalue = float(max([abs(x) for x in data]))
        if(normalize):
            scaled = [int(x / maxabsvalue * 32767) for x in data]
        else:
            scaled = [int(x * 32767) for x in data]
        nchan = 1

    fp = BytesIO()
    waveobj = wave.open(fp, mode='wb')
    waveobj.setnchannels(nchan)
    waveobj.setframerate(rate)
    waveobj.setsampwidth(2)
    waveobj.setcomptype('NONE', 'NONE')
    waveobj.writeframes(b''.join([struct.pack('<h', x) for x in scaled]))
    val = fp.getvalue()
    waveobj.close()

    return val
# </editor-fold>
##################\\--make_wav--//##################