import argparse
from ptlreg.apy.amedia.media.Image import *
from ptlreg.apydn.imagegraph.capture.CaptureTarget import *
from ptlreg.apydn.imagegraph.imagefilesample.ImageFileSample import *



def AlignUnstructuredDataset(
    target_name,
    output_dir,
    source_dir,
    seconds_before_new_primary=30,
    force_metadata_timestamps=False
):
    if(not os.path.exists(output_dir)):
        warnings.warn("Output dir {} does not exist".format(output_dir));
        raise ValueError("Output dir {} does not exist".format(output_dir))
    dataset_dir = os.path.join(output_dir, target_name)+os.sep;
    make_sure_dir_exists(dataset_dir)
    ct = CaptureTarget(dataset_dir, target_name)
    ct.pull_directory_to_originals(source_dir, force_metadata_timestamps=force_metadata_timestamps)
    sessions = ct.cluster_into_sessions(seconds_before_new_primary=seconds_before_new_primary).sort_by_timestamp()
    print("n samples is {}; n sessions is {}".format(len(ct.samples), len(sessions)))
    ct.save();
    ct.run_matching_on_original_samples()
    ct.calculate_aligned_undistorted_panos()
    ct.save();



def main():
    # parse the arguments from the command line
    parser = argparse.ArgumentParser(description='Align ustructured dataset for a capture target consisting of images in a common directory. The images should either have metadata timestamps or be named with strings that parse to timestamps.')
    parser.add_argument("-i",
                        help="Source directory. Path to directory holding images to register.",
                        dest="source_dir", required=True)
    parser.add_argument("-o",
                        help="Output directory. A directory matching the target name will be created here, where data will be pulled and registered.",
                        dest="output_dir", required=True)
    parser.add_argument("-n", help="Name of the capture target", dest="target_name", default="structured_capture_target")
    # parser.add_argument("-m", help="Force the use of meta data for timestamps. Otherwise, files with names that can be interpreted as timestamps will use the data in the file name (otherwise metadata will still be used). Defaults to false.", dest="force_metadata_timestamps", default=False)
    parser.add_argument("-t",
                        help="Number of seconds separating pictures that should be considered different capture sessions. Defaults to 30.",
                        dest="seconds_before_new_primary",
                        default=30)
    args = parser.parse_args()

    target_name = args.target_name;
    output_dir = args.output_dir;
    source_dir = args.source_dir;
    # force_metadata_timestamps = args.force_metadata_timestamps;
    force_metadata_timestamps = False;
    seconds_before_new_primary=args.seconds_before_new_primary;
    AlignUnstructuredDataset(
        target_name = target_name,
        output_dir = output_dir,
        source_dir = source_dir,
        seconds_before_new_primary=seconds_before_new_primary,
        force_metadata_timestamps=force_metadata_timestamps
    )
    return

if __name__ == "__main__":
    main()