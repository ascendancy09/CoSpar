import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "cospar",
    packages = ['cospar'],
    version = '0.0.2',
    description = 'CoSpar: integrating transcriptome and clonal information for dynamic inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Shou-Wen Wang',
    author_email = 'shouwen_wang@hms.harvard.edu',
    url = 'https://github.com/ShouWenWang/cospar',
    install_requires=['numpy>=1.19.4', 'scipy>=1.5.4', 'scikit-learn>=0.23.2', 'scanpy>=1.6.0', 'pandas>=1.1.4', 'statsmodels==0.12.1','plotnine>=0.7.1','matplotlib>=3.3.3','fastcluster>=1.1.26'],
    )
