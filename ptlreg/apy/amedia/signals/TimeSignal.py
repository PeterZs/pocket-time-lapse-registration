from ptlreg.apy.core import AObject
from ptlreg.apy.core import SavesToJSON
import numpy as np
from .maths.NDArray import NDArray
from .SignalMixin import *
import matplotlib
import matplotlib.pyplot as plt
import ptlreg.apy.utils
import scipy as sp
import math



def _spotgt_shift_bit_length(x):
    #smallest power of two greater than
    assert(isinstance(x,int)), "In 'spotgt_shift_bit_length(x)' x must be integer."
    return 1<<(x-1).bit_length()

class TimeSignalMixin(SignalMixin):
    # <editor-fold desc="Property: 'sampling_rate'">

    def __init__(self, samples=None, sampling_rate=None, path=None, **kwargs):
        # self.start_time=start_time;
        super(TimeSignalMixin, self).__init__(path=path, **kwargs);
        self._init_samples(samples=samples, sampling_rate=sampling_rate);

    def _init_samples(self, samples, sampling_rate = None):
        self._samples = samples;
        if(sampling_rate is not None):
            self.sampling_rate = sampling_rate;

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

    def _init_from_aobject_no_samples(self, fromobject):
        self.sampling_rate = fromobject.sampling_rate;
        super(TimeSignalMixin, self).init_from_aobject(fromobject);

    def init_from_aobject(self, fromobject, share_data = False):
        if(share_data):
            self._samples = fromobject._samples;
            self._init_from_aobject_no_samples(fromobject);
        else:
            self._init_from_aobject_no_samples(fromobject);
            if(fromobject._samples is not None):
                self._samples = fromobject._samples.copy();

    def write_to_file(self, output_path=None, n_channels=None, **kwargs):
        assert(output_path), "must provide path to save TimeSignal."
        self.saveJSON(json_path=output_path);



    @property
    def shape(self):
        return np.shape(self.samples);

    @property
    def _shape(self):
        return np.shape(self._samples);

    @property
    def sample_times(self):
        return self._sample_times;
    @property
    def _sample_times(self):
        return np.linspace(0, self.duration,num=self.n_time_samples,endpoint=False);

    def _set_value_range(self, value_range=None):
        if(value_range is None):
            value_range = [0,1];
        data = self.samples[:];
        currentscale = np.max(data)-np.min(data);
        data = (data/currentscale)*(value_range[1]-value_range[0]);
        data = data-np.min(data)+value_range[0]
        self._samples = data;

    def get_duration(self):
        return np.true_divide(len(self.samples), self.sampling_rate);

    @property
    def n_time_samples_unclipped(self):
        return len(self._samples);

    @property
    def n_time_samples(self):
        return len(self.samples);

    @staticmethod
    def MEDIA_FILE_EXTENSIONS():
        return ['.timesignal'];

    @property
    def _sampling_rate(self):
        return self.get_info('sampling_rate');

    @_sampling_rate.setter
    def _sampling_rate(self, value):
        self.set_info('sampling_rate', value);

    @property
    def sampling_rate(self):
        return self._getSamplingRate();
    def _getSamplingRate(self):
        sr = self._sampling_rate;
        if (sr is not None):
            return sr;
        else:
            return 0.0;
    @sampling_rate.setter
    def sampling_rate(self, value):
        self._setSamplingRate(value);
    def _setSamplingRate(self, value):
        self._sampling_rate=value;
    # </editor-fold>

    @property
    def sample_duration(self):
        if (self.sampling_rate == 0):
            return None;
        return np.true_divide(1.0, self.sampling_rate);

    @property
    def duration(self):
        return self.get_duration();


    @property
    def _default_frame_time(self):
        return np.true_divide(self._default_frame_length, self.sampling_rate);

    @property
    def _default_frame_length(self):
        return max(1, int(self.sampling_rate * 0.05));

    @property
    def _default_hop_length(self):
        return int(np.floor(self._default_frame_length * 0.5));

    def roll(self, shift_time=None, shift_samples=None, **kwargs):
        shift_samples_provided = (shift_samples is not None);
        shift_time_provided = (shift_time is not None);
        assert(shift_samples_provided != shift_time_provided), "provide exactly one of shift_time or shift_samples.\nshift_time:{}\nshift_samples:{}".format(shift_time, shift_samples);
        if(shift_samples_provided):
            self.samples = np.roll(self.samples, shift_samples, **kwargs);
        else:
            self.samples = np.roll(self.samples, int(np.round(shift_time*self.sampling_rate)), **kwargs);

    def get_rolled(self, shift_time=None, shift_samples=None, **kwargs):
        rval = self.clone(share_data=False);
        rval.roll(shift_time, shift_samples, **kwargs);
        return rval;


    def resample(self, sampling_rate, **kwargs):
        """
        We resample the full underlying signal, not just the clip
        :param sampling_rate:
        :return:
        """
        new_n_samples = sampling_rate * self.duration; # duration of source signal
        self.samples = sp.signal.resample(self.samples, int(new_n_samples), **kwargs);
        self.sampling_rate = sampling_rate;

    def get_resampled(self, sampling_rate, **kwargs):
        rval = self.clone(share_data=False);
        rval.resample(sampling_rate=sampling_rate, **kwargs);
        return rval;

    ##################//--Waveforms--\\##################
    # <editor-fold desc="Waveforms">
    @classmethod
    def _simple_sin_samples(cls, frequency, duration=None, amplitude=None, phase=None, sampling_rate=1600, **kwargs):
        if(phase is None):
            phase = 0;
        if (duration is None):
            duration = 3.0;
        x= np.sin(np.linspace(0, duration * frequency * 2 * np.pi, int(np.round(duration * sampling_rate))) + phase);
        if(amplitude is not None):
            x = amplitude*x;
        return x;

    @classmethod
    def _damped_sin_samples(cls, frequency, duration=None, sampling_rate=16000, damping=0.1, noise_floor=0.001, phase=None, **kwargs):
        """

        :param frequency:
        :param duration: if None, will set length necessary to damp to noise floor, up to 10.0s.
        :param sampling_rate:
        :param damping: Every period the amplitude is multiplied by 0.5^damping
        :param noise_floor:
        :return:
        """
        if(phase is None):
            phase = 0;
        if (duration is None):
            if (damping > 0.005):
                nosc = np.true_divide(math.log(noise_floor, 0.5), damping);
                duration = np.true_divide(nosc, frequency);
            else:
                duration = 10.0;

        x = np.sin(np.linspace(0, duration * frequency * 2 * np.pi, int(np.round(duration * sampling_rate))) + phase);
        dmp = np.linspace(0, duration * frequency, int(np.round(duration * sampling_rate)));
        dmp = np.power(0.5, dmp * damping);
        return np.multiply(x, dmp);

    @classmethod
    def _saw_ramp_samples(cls, freq_start, freq_end=None, duration=None, sampling_rate=16000, **kwargs):
        """

        :param freq_start:
        :param freq_end:
        :param duration:
        :param sampling_rate:
        :return:
        """

        if (duration is None):
            duration = 3.0;
        n_samples = np.round(duration * sampling_rate);
        t = np.linspace(0, duration, n_samples);
        peak_times, peak_vals = cls._ramp_peaks(freq_start=freq_start, freq_end=freq_end, duration=duration, **kwargs);
        out_sig_interp = sp.interpolate.interp1d(x=peak_times, y=peak_vals, kind='linear', axis=-1);
        out_sig = (out_sig_interp(t));
        # out_sig = np.interp(t, peak_times, peak_vals, left=0.0, right=0.0);
        return out_sig;

    @classmethod
    def _damping_samples(cls, duration=None, sampling_rate=16000, halflife=0.1, **kwargs):
        t = np.linspace(0, duration, int(np.round(duration * sampling_rate)));
        return np.power(2,-t/halflife);

    @classmethod
    def _square_ramp_samples(cls, freq_start, freq_end=None, duration=None, sampling_rate=16000, **kwargs):
        """

        :param freq_start:
        :param freq_end:
        :param duration:
        :param sampling_rate:
        :return:
        """

        if (freq_end is None):
            freq_end = freq_start * 2.0;
        if (duration is None):
            duration = 3.0;

        n_samples = np.round(duration * sampling_rate);
        t = np.linspace(0, duration, n_samples);
        peak_times, peak_vals = cls._ramp_peaks(freq_start=freq_start, freq_end=freq_end, duration=duration, **kwargs);
        out_sig_interp = sp.interpolate.interp1d(x=peak_times, y=peak_vals, kind='nearest', axis=-1);
        out_sig = (out_sig_interp(t));
        # out_sig = np.interp(t, peak_times, peak_vals, left=0.0, right=0.0);
        return out_sig;

    @classmethod
    def _chirp_samples(cls, freq_start, freq_end=None, duration=None, amplitude=0.2, sampling_rate = 16000, **kwargs):
        if (freq_end is None):
            freq_end = freq_start * 2.0;
        if (duration is None):
            duration = 3.0;

        t = np.linspace(0, duration, int(np.round(duration * sampling_rate)));
        chirp = sp.signal.chirp(t, freq_start, duration, freq_end, **kwargs);
        return chirp*amplitude;

    @classmethod
    def create_random(cls, duration=2, sampling_rate=44100, amplitude=1.0, **kwargs):
        return cls(samples=np.random.rand(int(np.round(duration*sampling_rate)))*amplitude, sampling_rate=sampling_rate);


    @classmethod
    def create_sine(cls, frequency=None, duration=2, amplitude=0.25, sampling_rate = 44100, phase = None, **kwargs):
        s = cls._simple_sin_samples(duration=duration, frequency=frequency, sampling_rate=sampling_rate, phase=phase);
        s = s*amplitude;
        sa = cls(samples=s, sampling_rate=sampling_rate, name = '{}Hz'.format(frequency));
        # sa.setMaxAbsValue(amplitude);
        return sa;

    @classmethod
    def create_saw(cls, frequency=None, duration=2, amplitude=0.25, sampling_rate = 44100, phase=None, **kwargs):
        if(phase is None):
            phase = 0;
        t = np.linspace(0, duration, int(np.round(duration * sampling_rate)));
        return cls(samples=amplitude*sp.signal.sawtooth(2 * np.pi * frequency * (t + (phase / (2 * np.pi)))), sampling_rate=sampling_rate, name ='Saw {}Hz'.format(frequency));

    @classmethod
    def create_triangle_wave(cls, frequency=None, duration=2, amplitude=0.25, sampling_rate = 44100, phase=None, **kwargs):
        if(phase is None):
            phase = 0;
        t = np.linspace(0, duration, int(np.round(duration * sampling_rate)));
        return cls(samples=amplitude*sp.signal.sawtooth(2 * np.pi * frequency * (t + (phase / (2 * np.pi))), width=0.5), sampling_rate=sampling_rate, name ='Saw {}Hz'.format(frequency));

    @classmethod
    def create_square(cls, frequency=None, duration=2, amplitude=0.25, sampling_rate = 44100, phase=None, **kwargs):
        if(phase is None):
            phase = 0;
        t = np.linspace(0, duration, int(np.round(duration * sampling_rate)));
        return cls(samples=amplitude*sp.signal.square(2 * np.pi * frequency * (t + (phase / (2 * np.pi)))), sampling_rate=sampling_rate, name ='Square {}Hz'.format(frequency));

    @classmethod
    def create_damped_exponential(cls, duration=2, halflife=1, initial_value = 1.0, sampling_rate = 44100):
        dmp = np.linspace(0, duration, int(np.round(duration * sampling_rate)));
        dmp = np.exp(-dmp / (np.true_divide(halflife, 0.69314718056)));
        return cls(samples=dmp/initial_value, sampling_rate=sampling_rate);

    @classmethod
    def create_sum_sines(cls, frequencies, amplitudes=None, damps = None, **kwargs):
        rval = cls.Sine(frequencies[0], **kwargs);
        if(amplitudes is None):
            amplitudes = np.ones(len(frequencies)) * (np.true_divide(1, len(frequencies)));
        for f in range(1, len(frequencies)):
            rval = rval+cls.Sine(frequencies[f], amplitude=amplitudes[f], **kwargs);
        # rval.setValueRange([-1,1]);
        return rval;

    @classmethod
    def create_zeros(cls, duration=None, sampling_rate=None, like=None):
        assert(((duration is not None) and (sampling_rate is not None)) != (like is not None)), "must provide (duration: {} and sampling_rate: {}) xor like:{}".format(duration, sampling_rate, like);
        if((duration is not None) and (sampling_rate is not None)):
            return cls(samples=np.zeros(int(np.round(duration*sampling_rate))), sampling_rate=sampling_rate);
        return cls(samples=np.zeros_like(like.samples), sampling_rate=like.sampling_rate);

    @classmethod
    def create_impulse_train(cls, duration, sampling_rate, frequency=0, amp=1, start=0):
        comb = cls.create_zeros(duration=duration, sampling_rate=sampling_rate);
        t = start;
        step = np.true_divide(1.0, frequency);
        while(t<=(duration-step)):
            comb.set_value_for_time(t, amp);
            t = t+step;
        return comb;


    @classmethod
    def create_ping(cls, duration=None, frequencies=None, damping=None, sampling_rate = 16000):
        if(frequencies is None):
            # freqs = [400,500,600, 700, 800, 900, 1000, 1100, 1200, 1300];
            # freqs = np.arange(4, 25) * 100
            frequencies = np.arange(5, 25) * 75; # just kind of thought this sounded fine...
        if(damping is None):
            damping = [0.05]*len(frequencies);
        if(not isinstance(damping,list)):
            damping = [damping]*len(frequencies);
        s = cls._damped_sin_samples(frequency=frequencies[0], duration = duration, sampling_rate = sampling_rate, damping=damping[0]);
        for h in range(1, len(frequencies)):
            new_s = cls._damped_sin_samples(frequency=frequencies[h], duration = duration, sampling_rate = sampling_rate, damping=damping[h]);
            if(len(new_s)>len(s)):
                new_s[:len(s)]=new_s[:len(s)]+s;
                s = new_s;
            else:
                s[:len(new_s)] = s[:len(new_s)]+new_s;

        return s;

    @classmethod
    def create_pings_at_times(cls, ping_times, duration, sampling_rate=44100, **kwargs):
        ping = cls.create_ping(**kwargs);
        a = cls.Zeros(duration=duration, sampling_rate=sampling_rate);
        for p in ping_times:
            a._addSoundToOriginalSamples(ping, p);
        a.reset_mono_using_multi_channel()
        return a;


    @classmethod
    def create_square_ramp(cls, freq_start, freq_end=None, duration=None, sampling_rate=16000, gain = None, **kwargs):
        """

        :param freq:
        :param duration: if None, will set length necessary to damp to noise floor, up to 10.0s.
        :param sampling_rate:
        :param damping: Every period the amplitude is multiplied by 0.5^damping
        :param noise_floor:
        :return:
        """

        sig = cls._square_ramp_samples(freq_start=freq_start,
                                       freq_end=freq_end,
                                       duration=duration,
                                       sampling_rate=sampling_rate, **kwargs);
        if (gain is not None):
            sig = sig * gain;
        squareramp = cls(samples=sig, sampling_rate=sampling_rate,
                         name='square_ramp_{}hz_{}hz_{}s'.format(freq_start, freq_end, duration));
        return squareramp;

    @classmethod
    def create_chirp(cls, freq_start, freq_end=None, duration=None, sampling_rate=16000, gain = None, **kwargs):
        sig = cls._chirp_samples(freq_start=freq_start,
                                 freq_end=freq_end,
                                 duration=duration,
                                 sampling_rate=sampling_rate, **kwargs);
        if (gain is not None):
            sig = sig * gain;
        chirp = cls(samples=sig, sampling_rate=sampling_rate,
                    name='chirp_{}hz_{}hz_{}s'.format(freq_start, freq_end, duration));
        return chirp;

    @classmethod
    def create_kernel_box(cls, duration, sampling_rate, width, **kwargs):
        sig = cls.create_zeros(duration=duration, sampling_rate=sampling_rate);
        halftime = duration*0.5;
        halfwidth = width*0.5;
        sig.set_time_range_to_value(time_range=[halftime - halfwidth, halftime + halfwidth], value=1.0, norm_to_value=True);
        return sig;

    @classmethod
    def create_kernel_sinc(cls, duration, sampling_rate, cutoff_frequency, **kwargs):
        nsamples=duration*sampling_rate;
        t = np.linspace(-nsamples*0.5,nsamples*0.5,nsamples);
        B = cutoff_frequency/sampling_rate;
        b2 = B*2;
        b2t=b2*t
        sincsamps = (b2)*np.sin(np.pi*(b2t))/(np.pi*b2t);
        return cls(samples=sincsamps, sampling_rate=sampling_rate);

    @classmethod
    def create_kernel_impulse(cls, duration, sampling_rate, **kwargs):
        sig = cls.create_zeros(duration=duration, sampling_rate=sampling_rate);
        sig.samples[int(np.round(len(sig.samples)*0.5))]=1.0;
        return sig;

    @classmethod
    def create_triangle(cls, duration, sampling_rate, **kwargs):
        n_samples = int(np.round(duration*sampling_rate));
        if(n_samples%2):
            triangle=np.arange((n_samples+1)/2);
            triangle=np.append(triangle,triangle[-2::-1])
        else:
            triangle=np.arange((n_samples)/2);
            triangle=np.append(triangle,triangle[::-1])

        normt = np.true_divide(1.0,np.max(triangle));
        triangle = triangle*normt;
        return cls(samples=triangle, sampling_rate=sampling_rate, **kwargs);

    @classmethod
    def create_kernel_triangle(cls, duration, sampling_rate, **kwargs):
        triangle = cls.create_triangle(duration=duration, sampling_rate=sampling_rate, **kwargs);
        normt = np.true_divide(1.0,np.sum(triangle.samples));
        triangle.samples = triangle.samples*normt;
        return triangle;

    def set_time_range_to_value(self, time_range, value, norm_to_value=False):
        startIndex = self._sample_index_for_time(time_range[0]);
        endIndex = self._sample_index_for_time(time_range[1]);
        if(norm_to_value):
            self.samples[startIndex:endIndex]=value/(endIndex-startIndex);
        else:
            self.samples[startIndex:endIndex]=value;




    # </editor-fold>
    ##################\\--Waveforms--//##################


    ##################//--Operators--\\##################
    # <editor-fold desc="Operators">
    def __len__(self):
        return self._samples.__len__();

    # def __repr__(self):
    #     return str(self._aobjects);

    def __getitem__(self, key):
        return self._samples.__getitem__(key);

    def __setitem__(self, key, value):
        return self._samples.__setitem__(key, value);

    def __delitem__(self, key):
        return self._samples.__delitem__(key);

    def __missing__(self, key):
        return self._samples.__missing__(key);

    def __iter__(self):
        return self._samples.__iter__();

    def __reversed__(self):
        return self._samples.__reversed__();

    def __contains__(self):
        return self._samples.__contains__();

    def _samples_equal(self, other):
        if isinstance(other, self.__class__):
            return self._samples == other._samples;
        return self._samples == other

    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #         return self._samples == other._samples;
    #     return self._samples == other

    def __add__(self, other):
        if(isinstance(other, self.__class__)):
            return self._selfclass(samples=np.add(self.samples, other.samples), sampling_rate=self.sampling_rate);
        else:
            return self._selfclass(samples=np.add(self.samples, other), sampling_rate=self.sampling_rate);

    def __radd__(self, other):
        return self.__add__(other);

    def __sub__(self, other):
        if(isinstance(other, self.__class__)):
            return self._selfclass(samples=np.subtract(self.samples, other.samples), sampling_rate=self.sampling_rate);
        else:
            return self._selfclass(samples=np.subtract(self.samples, other), sampling_rate=self.sampling_rate);

    def __rsub__(self, other):
        if (isinstance(other, (NDArray, self.__class__))):
            return self._RClass(np.subtract(other._ndarray, self._ndarray));
        else:
            return self._RClass(np.subtract(other, self._ndarray));

    def __mul__(self, other):
        if(isinstance(other, self.__class__)):
            return self._selfclass(samples=np.multiply(self.samples, other.samples), sampling_rate=self.sampling_rate);
        else:
            return self._selfclass(samples=np.multiply(self.samples, other), sampling_rate=self.sampling_rate);

    def __rmul__(self, other):
        return self.__mul__(other);
    # </editor-fold>
    ##################\\--Operators--//##################

    ##################//--PlotSignalForVis--\\##################
    # <editor-fold desc="PlotSignalForVis">
    def _plot_signal_for_vis(self, width_pixels=None, height_pixels=None, time_range=None, value_range=None, dpi = 250, show_axes=True, fig=None, vlines=None, vlinewidth=1, markersize = None, events=None, **kwargs):
        if(width_pixels is None):
            width_pixels = max(min(self.n_time_samples, np.true_divide(65534-dpi,dpi)), 640);#65536 is 2^16
        else:
            width_pixels = min(width_pixels, 30000);
        if(height_pixels is None):
            height_pixels = int(width_pixels*0.2);

        if(fig is None):
            fig = plt.figure(figsize=(np.true_divide(width_pixels, dpi),np.true_divide(height_pixels, dpi)), dpi=dpi);
        if(markersize is None):
            markersize = 0.02*height_pixels;
        xvals = self._sample_times;
        if(vlines is not None):
            for v in vlines:
                plt.axvline(x=v, color='#888888', linestyle='dotted', linewidth=vlinewidth);
        plt.plot(xvals,self.samples,'-');
        if(events is not None):
            etimes = [];
            evals = [];
            for e in events:
                if(hasattr(e, 'event_time')):
                    etimes.append(e.event_time);
                    evals.append(self.get_value_at_time(e.event_time));
                else:
                    etimes.append(e);
            plt.plot(etimes, evals, 'o', markersize=markersize);
        axes = plt.gca()
        if(time_range is not None):
            axes.set_xlim(time_range)
        if(value_range is not None):
            axes.set_ylim(value_range)

        axes.set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        return fig;

    def _sample_index_for_time(self, t, round=True):
        if(round):
            return int(np.round(t * self.sampling_rate));
        else:
            return t * self.sampling_rate;

    def get_value_at_time(self, t):
        return self.samples[int(self._sample_index_for_time(t))];

    def set_value_for_time(self, t, value):
        self.samples[int(self._sample_index_for_time(t))]=value;


    @classmethod
    def _plot_signals_for_vis(cls, signals, width_pixels=None, height_pixels=None, time_range=None, value_range=None, dpi = 250, fig=None, vlines=None, vlinewidth=1, markersize = None, events=None, linewidth=None, show_axes=True, **kwargs):
        sig0 = signals[0];
        if(width_pixels is None):
            width_pixels = max(min(sig0.n_time_samples, np.true_divide(65534-dpi,dpi)), 640);#65536 is 2^16
        if(height_pixels is None):
            height_pixels = int(width_pixels*0.2);
        if(fig is None):
            fig = plt.figure(figsize=(np.true_divide(width_pixels, dpi),np.true_divide(height_pixels, dpi)), dpi=dpi);
        if(markersize is None):
            markersize = 0.02*height_pixels;
        if(vlines is not None):
            for v in vlines:
                plt.axvline(x=v, color='#888888', linestyle='dotted', linewidth=vlinewidth);
        # ####################################
        if(linewidth is not None):
            if(not isinstance(linewidth, (list, tuple))):
                linewidth = [linewidth]*len(signals);
        else:
            linewidth=[None]*len(signals);

        for signali in range(len(signals)):
            signal = signals[signali];
            plt.plot(signal._sample_times, signal.samples, linewidth=linewidth[signali]);
        if(events is not None):
            etimes = [];
            evals = [];
            for e in events:
                if(hasattr(e, 'event_time')):
                    etimes.append(e.event_time);
                    evals.append(self.get_value_at_time(e.event_time));
                else:
                    etimes.append(e);
            plt.plot(etimes, evals, 'o', markersize=markersize);
        axes = plt.gca()
        if(time_range is not None):
            axes.set_xlim(time_range)
        if(value_range is not None):
            axes.set_ylim(value_range)
        if(not show_axes):
            axes.set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
            plt.margins(0,0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
        return fig;
    # </editor-fold>
    ##################\\--PlotImageFromSignal--//##################

    ##################//--GetSignalPlotPixels--\\##################
    # <editor-fold desc="PlotPixels">

    # @FeatureFunction('fft')
    def get_fft(self, **kwargs):
        return np.fft.fftshift(sp.fftpack.fft(self.samples));

    # @FeatureFunction('fft_freqs')
    def get_fft_freqs(self):
        return sp.fftpack.fftfreq(self.n_time_samples, self.sample_duration);

    def get_freqs(self):
        f = sp.fftpack.fftfreq(self.n_time_samples, self.sample_duration);
        fshift = np.fft.fftshift(f);
        return fshift;


    @property
    def fft(self):
        return self.get_fft();

    @property
    def fftReal(self):
        return np.real(self.fft);

    @property
    def fftImag(self):
        return np.imag(self.fft);

    def plot_fft_real(self, frequency_range=None, show_axes=True, axis=None, margins=None, figsize=None, title=None, **kwargs):
        if (title is None):
            title = self.name+" FFT Real";
        assert(not((figsize is not None)and(axis is not None))), "Conflict: provided axis and figsize to {}.plot()".format(self.__class__);
        if(frequency_range is None):
            frequency_range=[0,0.5*self.sampling_rate];
        X = self.fftReal;
        f = sp.fftpack.fftfreq(self.n_time_samples, self.sample_duration);
        fshift = np.fft.fftshift(f);
        vals = X;


        if(title is not None):
            plt.title(title);

        if(axis is None):
            if(title is not None):
                numarg = dict(num=title);
            else:
                numarg={};
            if(figsize is not None):
                figure = plt.figure(figsize=figsize, **numarg)
            else:
                figure = plt.figure(figsize=[9,3], **numarg);

        if(axis is not None):
            axis.plot(fshift, vals) # magnitude spectrum
        else:
            plt.plot(fshift, vals) # magnitude spectrum

        if(axis is None):
            axis = plt.gca()

        if(not show_axes):
            axis.set_axis_off()
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            #                     hspace = 0, wspace = 0)
            axis.margins(0,0)
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())


        if(margins is not None):
            axis.margins(margins[0], margins[1]);
        # f = np.linspace(-self.sampling_rate*0.5, self.sampling_rate*0.5, len(self.samples))  # frequency variable

        if(frequency_range != 'full'):
            axis.set_xlim(frequency_range[0], frequency_range[1]);
        axis.set_xlabel('Frequency (Hz)')


    def plot_amplitude_spectrum(self, frequency_range=None, show_axes=True, axis=None, margins=None, figsize=None, new_figure=True, title = None, **kwargs):
        assert(not((figsize is not None)and(axis is not None))), "Conflict: provided axis and figsize to {}.plot()".format(self.__class__);
        X = sp.fftpack.fft(self.samples);
        X_mag = np.absolute(X)        # spectral magnitude
        f = sp.fftpack.fftfreq(self.n_time_samples, self.sample_duration);
        fshift = np.fft.fftshift(f);
        mags = np.fft.fftshift(X_mag);


        if(axis is None):
            if(figsize is not None):
                figure = plt.figure(figsize=figsize)
            else:
                if(new_figure):
                    figure = plt.figure(figsize=[9,3]);
        if(title is not None):
            plt.title(title);

        if(axis is not None):
            axis.plot(fshift, mags) # magnitude spectrum
        else:
            plt.plot(fshift, mags) # magnitude spectrum

        if(axis is None):
            axis = plt.gca()

        if(not show_axes):
            axis.set_axis_off()
            axis.margins(0,0)
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())

        if(margins is not None):
            axis.margins(margins[0], margins[1]);

        if(frequency_range and frequency_range != 'full'):
            axis.set_xlim(frequency_range[0], frequency_range[1]);
        axis.set_xlabel('Frequency (Hz)')


    def plot_power_spectrum(self, frequency_range=None, show_axes=True, axis=None, margins=None, figsize=None, new_figure=True, title = None, **kwargs):
        assert(not((figsize is not None)and(axis is not None))), "Conflict: provided axis and figsize to {}.plot()".format(self.__class__);
        # if(frequency_range is None):
        #     frequency_range=[0,0.5*self.sampling_rate];
        X = sp.fftpack.fft(self.samples);
        X_mag = np.absolute(X)        # spectral magnitude
        f = sp.fftpack.fftfreq(self.n_time_samples, self.sample_duration);
        fshift = np.fft.fftshift(f);
        mags = np.fft.fftshift(X_mag**2);


        if(axis is None):
            # if(title is not None):
            #     numarg = dict(num=title);
            # else:
            #     numarg={};
            if(figsize is not None):
                figure = plt.figure(figsize=figsize)
            else:
                if(new_figure):
                    figure = plt.figure(figsize=[9,3]);
        if(title is not None):
            plt.title(title);

        if(axis is not None):
            axis.plot(fshift, mags) # magnitude spectrum
        else:
            plt.plot(fshift, mags) # magnitude spectrum

        if(axis is None):
            axis = plt.gca()

        if(not show_axes):
            axis.set_axis_off()
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            #                     hspace = 0, wspace = 0)
            axis.margins(0,0)
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())


        if(margins is not None):
            axis.margins(margins[0], margins[1]);
        # f = np.linspace(-self.sampling_rate*0.5, self.sampling_rate*0.5, len(self.samples))  # frequency variable

        if(frequency_range and frequency_range != 'full'):
            axis.set_xlim(frequency_range[0], frequency_range[1]);
        axis.set_xlabel('Frequency (Hz)')

    def plot_phase(self, frequency_range=None, show_axes=True, axis=None, margins=None, figsize=None, title=None, **kwargs):
        assert(not((figsize is not None)and(axis is not None))), "Conflict: provided axis and figsize to {}.plot()".format(self.__class__);
        if(frequency_range is None):
            frequency_range=[0,0.5*self.sampling_rate];
        X = sp.fftpack.fft(self.samples);
        X_mag = np.angle(X)        # spectral magnitude
        f = sp.fftpack.fftfreq(self.n_time_samples, self.sample_duration);
        fshift = np.fft.fftshift(f);
        mags = np.fft.fftshift(X_mag);


        if(axis is None):
            if(title is not None):
                numarg = dict(num=title);
            else:
                numarg={};
            if(figsize is not None):
                figure = plt.figure(figsize=figsize, **numarg)
            else:
                figure = plt.figure(figsize=[9,3], **numarg);

        if(axis is not None):
            axis.plot(fshift, mags) # magnitude spectrum
        else:
            plt.plot(fshift, mags) # magnitude spectrum

        if(axis is None):
            axis = plt.gca()

        if(not show_axes):
            axis.set_axis_off()
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            #                     hspace = 0, wspace = 0)
            axis.margins(0,0)
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())


        if(margins is not None):
            axis.margins(margins[0], margins[1]);
        # f = np.linspace(-self.sampling_rate*0.5, self.sampling_rate*0.5, len(self.samples))  # frequency variable

        if(frequency_range != 'full'):
            axis.set_xlim(frequency_range[0], frequency_range[1]);
        axis.set_xlabel('Frequency (Hz)')

    def plot(self, time_range=None, value_range=None, show_axes=True, axis=None, margins=None, figsize=None, title=None, new_figure=True, start_time=None, **kwargs):
        assert(not((figsize is not None)and(axis is not None))), "Conflict: provided axis and figsize to {}.plot()".format(self.__class__);
        if (title is None):
            title = self.name;
        if(start_time is None):
            start_time=0;

        if(axis is None):
            # if(title is not None):
            #     numarg = dict(num=title);
            # else:
            #     numarg={};
            if(figsize is not None):
                figure = plt.figure(figsize=figsize)
            else:
                if(new_figure):
                    figure = plt.figure(figsize=[9,3]);
        if(title is not None):
            plt.title(title);


        if(self.sampling_rate is None):
            ptlreg.apy.utils.AINFORM("This {} instance does not have a sampling rate set, so plotting with sampling rate = 1Hz".format(type(self).__name__));
            clone = self.clone(share_data=True);
            clone.sampling_rate = 1.0;
            clone.plot(axis=axis, margins=margins, value_range=value_range, time_range=time_range);
            return;

        if(axis is None):
            axis = plt.gca()
        if(len(self.shape)==1):
            if(axis is not None):
                axis.plot(self._sample_times+start_time, self.samples, **kwargs);
            else:
                plt.plot(self._sample_times+start_time, self.samples, **kwargs);
        elif(len(self.shape)==2):
            self._show2d(**kwargs)

        if(title is not None):
            plt.title(title);

        if (time_range is not None):
            axis.set_xlim(time_range);
        if(value_range is not None):
            axis.set_ylim(value_range);

        if(not show_axes):
            axis.set_axis_off()
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            #                     hspace = 0, wspace = 0)
            plt.margins(0,0)
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())
        if(margins is not None):
            plt.margins(margins[0], margins[1]);

    def plot_add_stem(self, **kwargs):
        plt.stem(self._sample_times, self.samples, **kwargs);

    def plot_stem(self, xlim=None, ylim=None, show_axes=True, margins=None, axis=None, figsize=None, **kwargs):
        assert(not((figsize is not None)and(axis is not None))), "Conflict: provided axis and figsize to {}.plot()".format(self.__class__);
        if(figsize is not None):
            figure = plt.figure(figsize=figsize)
        if(self._sampling_rate is None):
            ptlreg.apy.utils.AINFORM("This {} instance does not have a sampling rate set, so plotting with sampling rate = 1Hz".format(type(self).__name__));
            clone = self.clone(share_data=True);
            clone.sampling_rate = 1.0;
            clone.plot();
            return;

        if(len(self.shape)==1):
            if(axis is not None):
                axis.stem(self._sample_times, self.samples, **kwargs);
            else:
                plt.stem(self._sample_times, self.samples, **kwargs);
            if (self.name is not None):
                plt.title(self.name)
        if (xlim is not None):
            plt.xlim(xlim);
        if(ylim is not None):
            plt.ylim(ylim);

        if(not show_axes):
            if(axis is None):
                axis = plt.gca()
            axis.set_axis_off()
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            #                     hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if(margins is not None):
            plt.margins(margins[0], margins[1]);

    def _get_signal_plot_pixels(self, width_pixels=None, height_pixels=None, time_range=None, value_range=None, dpi = 250, vlines=None, events=None, **kwargs):
        fig = self._plot_signal_for_vis(width_pixels=width_pixels, height_pixels=height_pixels, time_range=time_range, value_range=value_range, dpi = dpi, vlines=vlines, events=events, **kwargs);
        fig.canvas.draw();
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        return data;
    # </editor-fold>


    def _show2d(self, n_ticks = 10, **kwargs):
        if (len(self.shape) == 2):
            feature_axis_labels = self.get_info('feature_axis_labels');
            plt.imshow(self.samples);
            ax = plt.gca();
            ax.set_xticks(np.linspace(0, self.n_time_samples, n_ticks));
            ax.set_xticklabels(np.around(np.linspace(0, self.duration, n_ticks), 2));
            if(feature_axis_labels is not None):
                ax.set_xticks(np.linspace(0.5, len(feature_axis_labels)+0.5, len(feature_axis_labels)))
                ax.set_yticklabels(feature_axis_labels);
            if(self.name is not None):
                plt.title(self.name)

    @classmethod
    def get_plot_pixels_for_signals(cls, signals, shape=None, time_range=None, value_range=None, dpi=250, vlines=None,
                                    events=None, show_axes=True, **kwargs):
        # fig = self._plotWSignalsOnlyForVis(signals, width_pixels=width_pixels, height_pixels=height_pixels, xlim=xlim, ylim=ylim,
        #                                  dpi=dpi, vlines=vlines, events=events, **kwargs);

        if(shape is None):
            shape=[300,500];
        fig = TimeSignalMixin._plot_signals_for_vis(signals, width_pixels=shape[1], height_pixels=shape[0], time_range=time_range, value_range=value_range,
                                                    dpi=dpi, vlines=vlines, events=events, show_axes=show_axes, **kwargs);

        fig.canvas.draw();
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        return data;
    # </editor-fold>


    def get_plot_pixels(self, time_range=None, shape=None, show_axes=True, **kwargs):
        return self.get_plot_pixels_for_signals([self], shape=shape, time_range=time_range, show_axes=show_axes, **kwargs);
    ##################\\--PlotSignalForVis--//##################

    def convolve_with(self, kernel, mode='same'):
        ksamples = kernel;
        if(isinstance(kernel, TimeSignalMixin)):
            ksamples = kernel.samples;
        self.samples = np.convolve(self.samples, ksamples, mode=mode);

    def get_convolved_with(self, kernel, mode='same'):
        t = self.clone(share_data=False);
        t.convolve_with(kernel=kernel, mode=mode);
        return t;

    def get_gaussian_blurred(self, duration=None, std_seconds=None, sym=True):
        window_size = int(np.round(duration*self.sampling_rate));
        window_std = std_seconds*self.sampling_rate;
        return self.getConvolvedWith(kernel=sp.signal.windows.gaussian(window_size, window_std, sym=sym));

    @classmethod
    def create_gaussian(cls, duration=3, sampling_rate=100, std=0.5, normalized=True, **kwargs):
        nsamples = duration*sampling_rate;
        samples = sp.signal.windows.gaussian(nsamples, std=std*sampling_rate, **kwargs);
        if(normalized):
            integral = np.sum(samples);
            scalef = np.true_divide(1.0, integral);
            samples = samples*scalef;
        rval = cls(samples=samples, sampling_rate=sampling_rate);
        return rval;

    @classmethod
    def create_kernel_gaussian(cls, duration=3, sampling_rate=100, std=0.5, **kwargs):
        nsamples = duration*sampling_rate;
        samples = sp.signal.windows.gaussian(nsamples, std=std*sampling_rate, **kwargs);
        integral = np.sum(samples);
        scalef = np.true_divide(1.0, integral);
        samples = samples*scalef;
        rval = cls(samples=samples, sampling_rate=sampling_rate);
        return rval;



class TimeSignal(SavesToJSON, TimeSignalMixin, AObject):
    def __init__(self, samples=None, sampling_rate=None, path=None, **kwargs):
        super(TimeSignal, self).__init__(samples=None, sampling_rate=None, path=None, **kwargs);
        self._init_samples(samples=samples, sampling_rate=sampling_rate);
