import os
import glob
from ..MediaObject import MediaObject
from ptlreg.apy.core.filepath.FilePathList import FilePathList, _endswithany



ABEPY_MEDIA_EXAMPLEFILES_PATH = os.path.abspath(__file__)
ABEPY_MEDIA_EXAMPLEFILES_DIR = os.path.abspath(os.path.dirname(__file__));
MEDIAFILES_DIR = ABEPY_MEDIA_EXAMPLEFILES_DIR;
# MEDIAFILES_DIR = os.path.join(ABEPY_MEDIA_EXAMPLEFILES_DIR, 'static'+os.sep)


MIDI_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'midi'+os.sep);

AUDIO_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'audio'+os.sep);
AUDIO_FILES = [];
AUDIO_FILE_PATHS = {};
for filename in os.listdir(AUDIO_FILES_DIR):
    # if(reduce(lambda x,y: x or y, map(lambda ext: filename.lower().endswith(ext), Audio.MEDIA_FILE_EXTENSIONS()))):
    # if(filename[0]!='.'):
    if(_endswithany(filename, ['.wav','.mp3','.ogg'])):
        AUDIO_FILES.append(filename);
        AUDIO_FILE_PATHS[filename]=(os.path.join(AUDIO_FILES_DIR, filename));

def GetTestAudioPath(filename=None):
    if (filename is None):
        filename = "Wilhelm_Scream.ogg"
    return AUDIO_FILE_PATHS[filename];

def GetTestAudio(filename=None):
    return MediaObject.load_media_object(path=GetTestAudioPath(filename));



VIDEO_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'video'+os.sep);
VIDEO_FILES = [];
VIDEO_FILE_PATHS = {};
for filename in os.listdir(VIDEO_FILES_DIR):
    if(_endswithany(filename, ['.mp4', '.mov', '.avi', '.flv'])):
        VIDEO_FILES.append(filename);
        VIDEO_FILE_PATHS[filename]=(os.path.join(VIDEO_FILES_DIR, filename));


def GetTestVideoSynthBall(filename=None):
    if (filename is None):
        filename = "synthball.mp4"
    return MediaObject.load_media_object(VIDEO_FILE_PATHS[filename]);

def GetTestVideo(filename=None):
    if (filename is None):
        filename = "dramatic_squirrel.mp4"
    return MediaObject.load_media_object(VIDEO_FILE_PATHS[filename]);

YOUTUBE_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'youtube'+os.sep);
YOUTUBE_FILES = [];
YOUTUBE_FILE_PATHS = {};
for filename in os.listdir(YOUTUBE_FILES_DIR):
    YOUTUBE_FILES.append(filename);
    YOUTUBE_FILE_PATHS[filename]=(os.path.join(YOUTUBE_FILES_DIR, filename));
def GetTestYouTubeVideoPath(filename=None):
    if (filename is None):
        filename = "rfh4Mhp-a6U.mp4"
    return YOUTUBE_FILE_PATHS[filename];
def GetTestYouTubeVideo(filename=None):
    if (filename is None):
        filename = "rfh4Mhp-a6U.mp4"
    return MediaObject.load_media_object(YOUTUBE_FILE_PATHS[filename]);


IMAGE_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'images'+os.sep);
IMAGE_FILES = [];
IMAGE_FILE_PATHS = {};
for filename in os.listdir(IMAGE_FILES_DIR):
    if(_endswithany(filename, ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])):
        IMAGE_FILES.append(filename);
        IMAGE_FILE_PATHS[filename] = (os.path.join(IMAGE_FILES_DIR, filename));

def GetTestImagePath(filename=None):
    if(filename is None):
        filename = 'broccoli01.png';
        # filename = IMAGE_FILES[0];
    return IMAGE_FILE_PATHS[filename];

def GetTestImage(filename=None):
    return MediaObject.load_media_object(path=GetTestImagePath(filename));


ExampleImageFiles = FilePathList.from_directory(directory=IMAGE_FILES_DIR, extension_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']);

VEGGIE_FILES_DIR = os.path.join(IMAGE_FILES_DIR, 'vegetables');
VeggieFiles = FilePathList.from_directory(directory=VEGGIE_FILES_DIR, recursive=True, extension_list=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']);

ExampleAudioFiles = FilePathList.from_directory(directory=AUDIO_FILES_DIR, extension_list=['.wav', '.mp3', '.ogg']);

ExampleMIDIFiles = FilePathList.from_directory(directory=MIDI_FILES_DIR, extension_list=['.mid', '.midi']);

YouTubeKeys = dict(dramatic_squirrel='y8Kyi0WNg40',
                   breffast='6xncCLKC7gY',
                   iliketurtles='CMNry4PE93Y',
                   darksidedaughter='XF7b_MNEIAg',
                   techfear='Fc1P-AEaEp8',
                   metalscream='xpKxtTPQ1Q8',
                   catsmeowing1='UJluYVGlv5Q',
                   sandershuge='9V3-GAsMEHQ',
                   trumpno='Jopsh1n4QUY',
                   owenwilsononewow='mBr8mcLj9QY',
                   trump_bingbong1='JnidQacpioY',
                   )