import os
# import moviepy.editor as mpy
import warnings

try:
    from moviepy.audio.AudioClip import AudioArrayClip as MPYAudioArrayClip
except ImportError:
    warnings.warn("Failed to import moviepy. Install moviepy to use apy Video features")


import ptlreg.apy.utils
import sys
import math
import numpy as np
from IPython.utils import io as ipythonio
from PIL import Image as PIM

def create_video_from_video_and_audio(video=None, audio=None, video_path=None, audio_path=None,
                                      output_path=None, clip_to_video_length=True, return_vid=True, codec='libx264',
                                      bitrate=None, **kwargs):
    assert (not (video_path and video)), "provided both video path and object to CreateVideoFromVideoAndAudio"
    assert (not (audio_path and audio)), "provided both audio path and object to CreateVideoFromVideoAndAudio"
    assert (output_path), "Must provide output path for CreateVideoFromVideoAndAudio"

    import ptlreg.apy.amedia.media.Video as VideoMedia
    import ptlreg.apy.amedia.media.Audio as AudioMedia
    if (video_path):
        video = VideoMedia(video_path);
    if (audio_path):
        audio = AudioMedia(path=audio_path);

    # output_path = output_path.encode(sys.getfilesystemencoding()).strip();
    # output_path = output_path.encode('utf8').strip();
    ptlreg.apy.utils.make_sure_dir_exists(output_path);

    # audio_sig = audio_object.getSignal();
    audio_sig = audio.stereo_copy;
    audio_sampling_rate = audio.stereo_sampling_rate;
    is_stereo = True;

    if (audio_sig is None):
        is_stereo = False;
        audio_sig = audio.getSignal();
        audio_sampling_rate = audio.sampling_rate;
        n_audio_samples_sig = len(audio_sig);
    else:
        n_audio_samples_sig = audio_sig.shape[1];

    # ptlreg.apy.utils.apy.utils.AINFORM("stereo is {}".format(is_stereo));

    audio_duration = audio.duration;
    video_duration = video.duration;

    if (clip_to_video_length):
        n_audio_samples_in_vid = int(np.round(video_duration * audio_sampling_rate));

        if (n_audio_samples_in_vid < n_audio_samples_sig):
            if (is_stereo):
                audio_sig = audio_sig[:, :int(n_audio_samples_in_vid)];
            else:
                audio_sig = audio_sig[:int(n_audio_samples_in_vid)];
        else:
            if (n_audio_samples_in_vid > n_audio_samples_sig):
                nreps = math.ceil(np.true_divide(n_audio_samples_in_vid, n_audio_samples_sig));
                if (is_stereo):
                    audio_sig = np.concatenate(
                        (audio_sig, np.zeros((2, n_audio_samples_in_vid - n_audio_samples_sig))),
                        axis=1);
                else:
                    audio_sig = np.tile(audio_sig, (int(nreps)));
                    audio_sig = audio_sig[:int(n_audio_samples_in_vid)];
    if (is_stereo):
        # reshapex = np.reshape(audio_sig, (audio_sig.shape[1], audio_sig.shape[0]), order='F');
        reshapex = np.transpose(audio_sig);
        audio_clip = MPYAudioArrayClip(reshapex, fps=audio_sampling_rate);  # from a numeric arra
    else:
        reshapex = audio_sig.reshape(len(audio_sig), 1);
        reshapex = np.concatenate((reshapex, reshapex), axis=1);
        audio_clip = MPYAudioArrayClip(reshapex, fps=audio_sampling_rate);  # from a numeric arra

    video_clip = video._get_mpy_clip();
    # video_clip = video_clip.set_audio(audio_clip);

    video_clip.audio = audio_clip
    # video_clip.write_videofile(output_path,codec='libx264', write_logfile=False);

    temp_audio_file_path = video._get_temp_file_path(final_path='TEMP_' + video.file_name_base + '.m4a', temp_dir=None);
    if (bitrate is None):
        # video_clip.write_videofile(output_path, codec=codec, write_logfile=False);
        mpy_write_video_file(video_clip, output_path, temp_audio_file_path=temp_audio_file_path, codec=codec, write_logfile=False);
    else:
        mpy_write_video_file(video_clip, output_path, temp_audio_file_path=temp_audio_file_path, codec=codec, write_logfile=False, bitrate=bitrate);
        # video_clip.write_videofile(output_path, codec=codec, write_logfile=False, bitrate=bitrate);

    del video_clip
    if (return_vid):
        return VideoMedia(output_path);
    else:
        return True;


def stack_videos(video_objects=None, video_paths=None, output_path=None, audio=None, concatdim=0, force_recompute=True, **kwargs):
    from ptlreg.apy.amedia.media.Video import Video as VideoMedia
    from ptlreg.apy.amedia.media.Video import VideoWriter as VideoMediaWriter
    if(output_path is None):
        output_path = ptlreg.apy.utils.get_temp_file_path('stackvideo.mp4');
    assert output_path, "MUST PROVIDE OUTPUT PATH FOR VIDEO"

    if (not force_recompute):
        if (os.path.isfile(output_path)):
            return VideoMedia(path=output_path);
    matchdim = (concatdim + 1) % 2;
    vids = [];
    if (video_objects is not None):
        vids = video_objects;
    if (video_paths is not None):
        for vp in video_paths:
            vids.append(VideoMedia(path=vp));

    # output_path = output_path.encode(sys.getfilesystemencoding()).strip();
    ptlreg.apy.utils.make_sure_dir_exists(output_path);
    basevid = vids[0];
    # if (audio is None):
    #     audio = basevid.audio;
    sampling_rate = basevid.sampling_rate;

    if(audio is not None):
        tempfilepath = ptlreg.apy.utils.get_temp_file_path(final_file_path='temp_'+output_path, temp_dir_path=apy.utils.GetTempDir());
    else:
        tempfilepath = output_path;

    vwriter = VideoMediaWriter(output_path=tempfilepath, fps=sampling_rate);
    duration = basevid.get_duration();
    nsamples = sampling_rate * duration;
    old_frame_time = np.true_divide(1.0, sampling_rate);
    frame_start_times = np.linspace(start=0, stop=duration, num=int(np.ceil(nsamples)), endpoint=False);
    frame_index_floats = frame_start_times * sampling_rate;
    for nf in range(len(frame_index_floats)):
        frameind = frame_index_floats[nf];
        newframe = basevid.get_frame(frameind);
        for vn in range(1, len(vids)):
            addpart = vids[vn].get_frame(frameind);
            partsize = np.asarray(addpart.shape)[:];
            cumulsize = np.asarray(newframe.shape)[:];
            if (partsize[matchdim] != cumulsize[matchdim]):
                sz = partsize[:];
                sz[matchdim] = cumulsize[matchdim];
                # addpart = sp.misc.imresize(addpart, size=sz);
                addpart = np.array(PIM.fromarray(addpart).resize((int(sz[1]),int(sz[0]))));
            newframe = np.concatenate((newframe, addpart), concatdim);
        vwriter.write_frame(newframe);
    vwriter.close();

    if(audio is not None):
        rvid = create_video_from_video_and_audio(video_path=tempfilepath,
                                                 audio=audio,
                                                 output_path=output_path,
                                                 **kwargs);
        os.remove(tempfilepath);
    else:
        rvid = VideoMedia(tempfilepath);
    return rvid;


def create_video_from_video_and_audio_paths(video_path, audio_path, output_path, return_vid=True, **kwargs):
    return create_video_from_video_and_audio(video_path=video_path, audio_path=audio_path, output_path=output_path,
                                             return_vid=return_vid, **kwargs);


def create_video_from_video_and_audio_objects(video, audio, output_path, clip_to_video_length=True, return_vid=True, **kwargs):
    return create_video_from_video_and_audio(video=video, audio=audio, output_path=output_path,
                                             clip_to_video_length=clip_to_video_length, return_vid=return_vid, **kwargs);

def mpy_write_video_file(mpyclip, filename, temp_audio_file_path=None, **kwargs):
    # return mpyclip.write_videofile(filename=filename, temp_audiofile= temp_audio_file_path, audio_codec='aac', verbose=False, progress_bar=True, **kwargs);
    # print(mpyclip.fps)
    # print("Beginning of MPYWriteVideoFile")
    if (ptlreg.apy.utils.suppress_outputs()):
        ptlreg.apy.utils.AINFORM("MPY Writing {}".format(filename))
        with ipythonio.capture_output() as captured:
            mclip = mpyclip.write_videofile(filename=filename,
                                            # size=mpyclip.size,
                                            fps=mpyclip.fps,
                                            temp_audiofile=temp_audio_file_path,
                                            audio_codec='aac',
                                            # verbose=False,
                                            **kwargs);
    else:
        mclip = mpyclip.write_videofile(filename=filename,
                                        fps=mpyclip.fps,
                                        temp_audiofile=temp_audio_file_path,
                                        audio_codec='aac',
                                        # verbose=False,
                                        **kwargs);
    return mclip;

def create_noise_video(output_path=None, width=100, height=100, duration=3, audio_sampling_rate = 44000, frame_rate = 30, return_vid=True, codec='libx264', bitrate=None, **kwargs):
    assert (output_path), "Must provide output path for CreateVideoFromVideoAndAudio"
    import ptlreg.apy.amedia.media.Video as VideoMedia
    import ptlreg.apy.amedia.media.Audio as AudioMedia

    output_path = output_path.encode(sys.getfilesystemencoding()).strip();
    ptlreg.apy.utils.make_sure_dir_exists(output_path);

    audio_sig = np.random.rand(2,audio_sampling_rate*duration);
    is_stereo = True;
    n_audio_samples_sig = audio_sig.shape[1];
    audio_duration = duration;
    video_duration = duration;
    reshapex = np.transpose(audio_sig);
    audio_clip = MPYAudioArrayClip(reshapex, fps=audio_sampling_rate);  # from a numeric arra

    # #############


    video_clip = video_object._get_mpy_clip();
    video_clip = video_clip.set_audio(audio_clip);
    # video_clip.write_videofile(output_path,codec='libx264', write_logfile=False);

    temp_audio_file_path = video_object._get_temp_file_path(final_path='TEMP_' + video_object.file_name_base + '.m4a', temp_dir=None);
    if (bitrate is None):
        # video_clip.write_videofile(output_path, codec=codec, write_logfile=False);
        mpy_write_video_file(video_clip, output_path, temp_audio_file_path=temp_audio_file_path, codec=codec, write_logfile=False);
    else:
        mpy_write_video_file(video_clip, output_path, temp_audio_file_path=temp_audio_file_path, codec=codec, write_logfile=False, bitrate=bitrate);
        # video_clip.write_videofile(output_path, codec=codec, write_logfile=False, bitrate=bitrate);

    del video_clip
    if (return_vid):
        return VideoMedia(output_path);
    else:
        return True;


