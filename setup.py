import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "cospar",
    packages = ['cospar'],
    package_dir={'': 'cospar'},
    version = '0.0.1',
    description = 'CoSpar: integrating transcriptome and clonal information for dynamic inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Shou-Wen Wang',
    author_email = 'shouwen_wang@hms.harvard.edu',
    url = 'https://github.com/allonkleinlab/cospar',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'scanpy', 'matplotlib', 'pandas', 'statsmodels','scanpy','plotnine','pot'],
    )
