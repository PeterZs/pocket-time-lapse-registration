# import numpy as np
import ptlreg.apy.utils

class AOpArgUtils(object):

    @staticmethod
    def encode_dict(d):
        return ptlreg.apy.utils.jsonpickle.encode(d);

    @staticmethod
    def f_wh(obj, width=None, height=None, **kwargs):
        """
        float width height
        :param obj:
        :param width:
        :param height:
        :param kwargs:
        :return:
        """
        if(width is None):
            hratio = np.true_divide(height, obj.height);
            width = obj.width*hratio;
        elif(height is None):
            wratio = np.true_divide(width, obj.width);
            height = obj.height*wratio;
        return (width, height)

    @staticmethod
    def i_wh(obj, width=None, height=None, **kwargs):
        """
        integer width height
        :param obj:
        :param width:
        :param height:
        :param kwargs:
        :return:
        """
        if(width is None):
            hratio = np.true_divide(height, obj.height);
            width = obj.width*hratio;
        elif(height is None):
            wratio = np.true_divide(width, obj.width);
            height = obj.height*wratio;
        return (int(width), int(height))

    @staticmethod
    def f_wh_hash(obj, width=None, height=None, **kwargs):
        """
        float width height hash
        :param obj:
        :param width:
        :param height:
        :param kwargs:
        :return:
        """
        return hash((AOpArgUtils.f_wh(obj, width=width, height=height, **kwargs), AOpArgUtils.encode_dict(kwargs)));

    @staticmethod
    def i_wh_hash(obj, width=None, height=None, **kwargs):
        """
        integer width height hash
        :param obj:
        :param width:
        :param height:
        :param kwargs:
        :return:
        """
        return hash((AOpArgUtils.i_wh(obj, width=width, height=height, **kwargs), AOpArgUtils.encode_dict(kwargs)));

    @staticmethod
    def i_wh_f_sr(obj, width, height, sampling_rate, **kwargs):
        """
        integer width height, float sampling rate
        :param obj:
        :param width:
        :param height:
        :param sampling_rate:
        :param kwargs:
        :return:
        """
        if((width is None) and (height is None)):
            width = obj.width;
        if(width is None):
            hratio = np.true_divide(height, obj.height);
            width = obj.width*hratio;
        elif(height is None):
            wratio = np.true_divide(width, obj.width);
            height = obj.height*wratio;
        if(sampling_rate is None):
            sampling_rate = obj.sampling_rate;
        return (int(width), int(height), sampling_rate);

    @staticmethod
    def f_wh_f_sr(obj, width, height, sampling_rate, **kwargs):
        """
        float width height, float sampling rate
        :param obj:
        :param width:
        :param height:
        :param sampling_rate:
        :param kwargs:
        :return:
        """
        if((width is None) and (height is None)):
            width = obj.width;
        if(width is None):
            hratio = np.true_divide(height, obj.height);
            width = obj.width*hratio;
        elif(height is None):
            wratio = np.true_divide(width, obj.width);
            height = obj.height*wratio;
        if(sampling_rate is None):
            sampling_rate = obj.sampling_rate;
        return (width, height, sampling_rate);

    @staticmethod
    def remap_hash(obj, target_region=None, sampling_rate=None, time_scaling_policy=None, **kwargs):
        rm = obj._GetResolvedRegionMapTo(target_region, time_scaling_policy=time_scaling_policy);
        return hash((rm.source_region, rm.target_region, sampling_rate, AOpArgUtils.encode_dict(kwargs)));

    @staticmethod
    def i_wh_f_sr_hash(obj, width=None, height=None, sampling_rate=None, **kwargs):
        """
        integer width height, float sampling rate, hash
        :param obj:
        :param width:
        :param height:
        :param sampling_rate:
        :param kwargs:
        :return:
        """
        return hash((AOpArgUtils.i_wh_f_sr(obj, width=width, height=height, sampling_rate=sampling_rate), AOpArgUtils.encode_dict(kwargs)));
    @staticmethod
    def f_wh_f_sr_hash(obj, width=None, height=None, sampling_rate=None, **kwargs):
        """
        float width height, float sampling rate, hash
        :param obj:
        :param width:
        :param height:
        :param sampling_rate:
        :param kwargs:
        :return:
        """
        return hash((AOpArgUtils.f_wh_f_sr(obj, width=width, height=height, sampling_rate=sampling_rate), AOpArgUtils.encode_dict(kwargs)));

    @staticmethod
    def speed_factor(obj, duration=None, speed_factor=None, **kwargs):
        """
        use the speed factor as the hash/key. If duration is given, converts to speed factor for key
        :param obj:
        :param duration:
        :param speed_factor:
        :param kwargs:
        :return:
        """
        if(speed_factor is None):
            speed_factor = np.divide(obj.duration,duration);
        return speed_factor;

    @staticmethod
    def duration(obj, duration=None, speed_factor=None, **kwargs):
        """
        use the duration as the hash/key. If speed_factor is given, converts to duration for key
        :param obj:
        :param duration:
        :param speed_factor:
        :param kwargs:
        :return:
        """
        if(duration is None):
            duration = np.divide(obj.duration,speed_factor);
        return duration;

    @staticmethod
    def duration_hash(obj, duration=None, speed_factor=None, **kwargs):
        """
        hashed duration
        :param obj:
        :param duration:
        :param speed_factor:
        :param kwargs:
        :return:
        """
        return hash((AOpArgUtils.duration(obj=obj, duration=duration, speed_factor=speed_factor), AOpArgUtils.encode_dict(kwargs)));

    @staticmethod
    def speed_hash(obj, duration=None, speed_factor=None, **kwargs):
        """
        hashed speed factor
        :param obj:
        :param duration:
        :param speed_factor:
        :param kwargs:
        :return:
        """
        return AOpArgUtils.speed_factor(obj=obj, duration=duration, speed_factor=speed_factor, **kwargs);

    @staticmethod
    def media_object_hash(obj, media=None, **kwargs):
        return hash((media.get_file_and_clip_hashable()));

    @staticmethod
    def audio_object_hash(obj, audio=None, **kwargs):
        return AOpArgUtils.media_object_hash(obj, media=audio, **kwargs);

