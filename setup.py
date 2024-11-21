from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="RNApolis",
    version="0.4.8",
    packages=["rnapolis"],
    package_dir={"": "src"},
    author="Tomasz Zok",
    author_email="tomasz.zok@cs.put.poznan.pl",
    description="A Python library containing RNA-related bioinformatics functions and classes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tzok/rnapolis-py",
    project_urls={"Bug Tracker": "https://github.com/tzok/rnapolis-py/issues"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "annotator=rnapolis.annotator:main",
            "clashfinder=rnapolis.clashfinder:main",
            "metareader=rnapolis.metareader:main",
            "molecule-filter=rnapolis.molecule_filter:main",
            "motif-extractor=rnapolis.motif_extractor:main",
            "transformer=rnapolis.transformer:main",
            "rfam-folder=rnapolis.rfam_folder:main",
        ]
    },
    install_requires=[
        "appdirs",
        "graphviz",
        "mmcif",
        "numpy",
        "ordered-set",
        "orjson",
        "pandas",
        "pulp",
        "requests",
        "scipy",
        "viennarna",
    ],
)
