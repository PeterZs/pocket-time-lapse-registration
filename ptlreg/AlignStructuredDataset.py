import argparse
from ptlreg.apy.amedia.media.Image import *
from ptlreg.apydn.imagegraph.capture.CaptureTarget import *
from ptlreg.apydn.imagegraph.imagefilesample.ImageFileSample import *



def AlignStructuredData(
    target_name,
    output_dir,
    primaries_source_dir,
    secondaries_source_dir,
    force_metadata_timestamps = False
):
    is_structured = True
    if(not os.path.exists(output_dir)):
        warnings.warn("Output dir {} does not exist".format(output_dir));
        raise ValueError("Output dir {} does not exist".format(output_dir))
    dataset_dir = os.path.join(output_dir, target_name)+os.sep;
    make_sure_dir_exists(dataset_dir)
    ct = CaptureTarget(dataset_dir, target_name)
    ct.pull_directory_to_primaries(primaries_source_dir, force_metadata_timestamps=force_metadata_timestamps)
    ct.pull_directory_to_secondaries(secondaries_source_dir, force_metadata_timestamps=force_metadata_timestamps)
    print(force_metadata_timestamps)
    print(type(force_metadata_timestamps))
    ct.samples.calc_timestamps(force_use_metadata=force_metadata_timestamps)
    ct.samples.calc_timestamps()
    sessions = ct.get_session_set()
    print("n samples is {}; n sessions is {}".format(len(ct.samples), len(sessions)))
    ct.save();
    ct.run_matching_on_original_samples()
    ct.calculate_aligned_undistorted_panos()
    ct.save()


def main():
    # parse the arguments from the command line
    parser = argparse.ArgumentParser(description='Align structured dataset for acapture target consisting of primary images and secondary images in separate directories.')
    parser.add_argument("-p",
                        help="Primary source directory. Path to directory holding primary samples.",
                        dest="primaries_source_dir", required=True)
    parser.add_argument("-s",
                        help="Secondary source directory. Path to directory holding secondary samples.",
                        dest="secondaries_source_dir", required=True)
    parser.add_argument("-o",
                        help="Output directory. A directory matching the target name will be created here, where data will be pulled and registered.",
                        dest="output_dir", required=True)
    parser.add_argument("-n", help="Name of the capture target", dest="target_name", default="structured_capture_target")
    # parser.add_argument("-m", help="Force the use of meta data for timestamps. Otherwise, files with names that can be interpreted as timestamps will use the data in the file name (otherwise metadata will still be used). Defaults to false.", dest="force_metadata_timestamps",
    #                     default=False)
    args = parser.parse_args()

    target_name = args.target_name;
    output_dir = args.output_dir;
    primaries_source_dir = args.primaries_source_dir;
    secondaries_source_dir = args.secondaries_source_dir;
    # force_metadata_timestamps = args.force_metadata_timestamps;
    force_metadata_timestamps = False;

    AlignStructuredData(
        target_name = target_name,
        output_dir = output_dir,
        primaries_source_dir=primaries_source_dir,
        secondaries_source_dir=secondaries_source_dir,
        force_metadata_timestamps=force_metadata_timestamps
    )
    return

if __name__ == "__main__":
    main()