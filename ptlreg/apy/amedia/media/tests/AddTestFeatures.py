# from ptlreg.apy.amedia.mobjects.Audio import Audio, AudioFeature, AudioMethod, AudioStaticMethod, AudioClassMethod
# from ptlreg.apy.amedia.mobjects.Video import Video, VideoFeature, VideoMethod, VideoStaticMethod, VideoClassMethod
# from ptlreg.apy.amedia.mobjects.Image import Image, ImageFeature, ImageMethod, ImageStaticMethod, ImageClassMethod
from ptlreg.apy.amedia.mobjects import *
import numpy as np

@VideoFeature('TestVideoFeature')
def TestVideoFeature(self):
    return 3.1415927;

@VideoMethod
def TestVideoMethod(self, arg):
    return arg+self.test_file_path;

@VideoStaticMethod
def TestVideoStaticMethod(arg):
    return arg;

@VideoClassMethod
def TestVideoClassMethod(cls, arg):
    return arg+cls.aobject_type_name();


@AudioFeature('TestAudioFeature')
def TestAudioFeature(self):
    return 3.1415927;

@AudioMethod
def TestAudioMethod(self, arg):
    return arg+self.test_file_path;

@AudioStaticMethod
def TestAudioStaticMethod(arg):
    return arg;

@AudioClassMethod
def TestAudioClassMethod(cls, arg):
    return arg+cls.aobject_type_name();

@ImageFeature('TestImageFeature')
def TestImageFeature(self):
    return 3.1415927;

@ImageMethod
def TestImageMethod(self, arg):
    return arg+self.test_file_path;

@ImageStaticMethod
def TestImageStaticMethod(arg):
    return arg;

@ImageClassMethod
def TestImageClassMethod(cls, arg):
    return arg+cls.aobject_type_name();
