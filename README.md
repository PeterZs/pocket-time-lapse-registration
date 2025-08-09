# Pocket Time-Lapse Registration
From the SIGGRAPH 2025 paper ["Pocket Time-Lapse"](https://pocket-timelapse.github.io/) by Eric M. Chen, Žiga Kovačič, Madhav Aggarwal, and Abe Davis

Please cite our paper if you use this code in your research.

*DISCLAIMER: The code in this repo was written by me (Abe) in my spare time over several years and is admittedly a bit messy... I will try to answer questions and requests, but apologize if I'm slow to respond.* 

## How to install:
I recommend a developer install:

```bash
pip install -e .
```

## How to use:

Images should either have accurate created timestamps in their EXIF metadata, or be named in a way that reflects their capture time (e.g.,  `2024-11-20T14:45:52.jpeg`). If the name appears to be a timestamp, that will be used by default. If not, the exif data.

There are two types of datasets you can align into panoramic time-lapse:

- **Structured Datasets**: These have a set of primary images that presumably face the same part of a scene, and a set of secondary images for each primary that cover other angles. There will be a frame of the time-lapse for each primary image.

- **Unstructured Datasets**: These are just collections of images. "primaries" will be determined based on how much time separates each image from its neighbor. Essentially, each time `t` seconds passes, the next image will be considered a new primary. Images taken within `t` seconds of a primary will be considered secondaries for that primary.


Example usage (as seen in [run_test_data.sh](./run_test_data.sh)):

```bash
mkdir ./test_results
python ./ptlreg/AlignStructuredDataset.py \
  -p ./ptlreg/test_data/structured/primary \
  -s ./ptlreg/test_data/structured/secondary \
  -o ./test_results/ \
  -n "waterfall1_test" \

python ./ptlreg/AlignUnstructuredDataset.py \
  -i ./ptlreg/test_data/unstructured \
  -o ./test_results/ -n "structured_test" \
  -n "unstructured_test" \
  -t 300
```

The above code registers the example datasets in `ptlreg/test_data/structured` and `ptlreg/test_data/unstructured`, and saves the results to `./test_results/`.

For now, I've added some of the larger datasets to the drive folder:
[https://drive.google.com/drive/folders/1ZD0GWEeg80OOKIdFzcetgRH4pTmxO1ei?usp=sharing](https://drive.google.com/drive/folders/1ZD0GWEeg80OOKIdFzcetgRH4pTmxO1ei?usp=sharing)

We are trying to find a hosting solution for the remaining datasets and will update this README when we do.

