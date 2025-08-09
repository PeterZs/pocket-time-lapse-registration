import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pltreg",
    version="0.0.1",
    author="Abe Davis",
    author_email="abedavis@cornell.edu",
    description="Pocket time-lapse panoramic time-lapse registration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pocket-timelapse.github.io/",
    project_urls={
        'Abe Davis Research Group': 'https://www.cs.cornell.edu/abe/group/',
        'Source': 'https://pocket-timelapse.github.io/',
    },
    install_requires=[
        'numpy',
        'scipy',
        'jsonpickle',
        'pandas',
        'pandera',
        'pysolar',
        'networkx',
        'bigtree',
        'matplotlib',
        'future',
        'six',
        'termcolor',
        'xxhash',
        'jupyter',
        'ipython',
        'bs4',
        'requests',
        'opencv-python',
        'palettable',
        'pypng',
        'librosa',
        'imageio',
    ],
    packages=setuptools.find_packages(exclude=['contrib', 'docs', 'tests*']),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
