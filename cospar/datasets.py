

from pathlib import Path, PurePath
import scanpy as sc
from . import settings
from . import logging as logg

url_prefix='https://kleintools.hms.harvard.edu/tools/downloads/cospar'

def synthetic_bifurcation(data_path='data_bifur',figure_path='figure_bifur',data_des='bifur'):
    """
    We re-sample clones that go through a simulated bifurcation differentiation
    process.  

    It has only two time points. There is only a single 
    round of barcoding at the beginning. 
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='bifur_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def synthetic_bifurcation_continuous_barcoding(data_path='data_bifur_conBC',figure_path='figure_bifur_conBC',data_des='bifur_conBC'):
    """
    We re-sample clones that go through a simulated bifurcation differentiation
    process.  

    It has only two time points. There is only a single 
    round of barcoding at the beginning. 
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='bifur_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def reprogramming_merge_tags(data_path='data_reprog_M',figure_path='figure_reprog_M',data_des='CellTagging'):
    """
    The reprogramming dataset from 

    Biddy, B. A. et al. `Single-cell mapping of lineage and identity in direct 
    reprogramming`. Nature 564, 219–224 (2018).

    This dataset has multiple time points for both the clones and the state measurements. 

    The cells are barcoded over 3 rounds during the entire differentiation process. 
    We combine up to 3 tags from the same cell into a single clonal label in 
    representing the X_clone matrix. In this representation, each cell has at most 
    one clonal label. 
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='CellTagging_ConcatenateClone_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def reprogramming_no_merge_tags(data_path='data_reprog_noM',figure_path='figure_reprog_noM',data_des='CellTagging_NoConcat'):
    """
    The reprogramming dataset from 

    Biddy, B. A. et al. `Single-cell mapping of lineage and identity in direct 
    reprogramming`. Nature 564, 219–224 (2018).

    This dataset has multiple time points for both the clones and the state measurements. 

    The cells are barcoded over 3 rounds during the entire differentiation process. We treat
    barcode tags from each round as independent clonal label here. In this representation, 
    each cell can have multiple clonal labels.   
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='CellTagging_NoConcat_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def lung(data_path='data_lung',figure_path='figure_lung',data_des='Lung'):
    """
    The direct lung differentiation dataset from 

    Hurley, K. et al. `Reconstructed Single-Cell Fate Trajectories Define Lineage 
    Plasticity Windows during Differentiation of Human PSC-Derived Distal Lung Progenitors`. 
    Cell Stem Cell (2020) doi:10.1016/j.stem.2019.12.009.

    This dataset has multiple time points for the state manifold, but only one time point
    for the clonal observation on day 27. 
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='Lung_pos17_21_D27_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def hematopoiesis_all(data_path='data_blood_all',figure_path='figure_blood_all',data_des='LARRY'):
    """
    All of the hematopoiesis data set from 

    Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. 
    `Lineage tracing on transcriptional landscapes links state to fate 
    during differentiation`. Science 367, (2020)

    .. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
    :width: 600px
    :align: middle

    This dataset has 3 time points for both the clones and the state measurements. 

    This dataset is very big. Generating the transition map for this datset 
    could take many hours when run for the first time. 
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='LARRY_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)



def hematopoiesis_15perct(data_path='data_blood_15perct',figure_path='figure_blood_15perct',data_des='LARRY_sp500_ranking1'):
    """
    Top 15% most heterogeneous clones of the hematopoiesis data set from 

    Weinreb, C., Rodriguez-Fraticelli, A., Camargo, F. D. & Klein, A. M. 
    `Lineage tracing on transcriptional landscapes links state to fate 
    during differentiation`. Science 367, (2020)

    This dataset has 3 time points for both the clones and the state measurements. 

    This sub-sampled data better illustrates the power of CoSpar in robstly 
    inferring differentiation dynamics from a noisy clonal dataset. Also, it 
    is smaller, and much faster to analyze. 
    """

    settings.data_path=data_path
    settings.figure_path=figure_path
    data_name='LARRY_sp500_ranking1_adata_preprocessed.h5ad'
    return load_data_core(data_path,figure_path,data_name,data_des)

def load_data_core(data_path,figure_path,data_name,data_des):
    url=f'{url_prefix}/{data_name}'
    path=f'{data_path}/{data_name}'
    path=Path(path)
    figure_path=Path(figure_path)

    if not path.parent.is_dir():
        logg.info(f'creating directory {path.parent}/ for saving data')
        path.parent.mkdir(parents=True)

    if not figure_path.is_dir():
        logg.info(f'creating directory {figure_path}/ for saving figures')
        figure_path.mkdir(parents=True)

    #print(url)
    status=_check_datafile_present_and_download(path,backup_url=url)
    if status:
        adata=sc.read(path)
        #adata.uns['data_path']=[str(data_path)]
        #adata.uns['figure_path']=[str(figure_path)]
        adata.uns['data_des']=[str(data_des)]
        return adata
    else:
        print("Error, files do not exist")
        return None    

def _check_datafile_present_and_download(path, backup_url=None):
    """Check whether the file is present, otherwise download."""
    path = Path(path)
    if path.is_file():
        return True
    if backup_url is None:
        return False
    logg.info(
        f'try downloading from url\n{backup_url}\n'
        '... this may take a while but only happens once'
    )
    if not path.parent.is_dir():
        logg.info(f'creating directory {path.parent}/ for saving data')
        path.parent.mkdir(parents=True)

    _download(backup_url, path)
    return True


def _download(url: str, path: Path):
    try:
        import ipywidgets
        from tqdm.auto import tqdm
    except ImportError:
        from tqdm import tqdm

    from urllib.request import urlopen, Request

    blocksize = 1024 * 8
    blocknum = 0

    try:
        with urlopen(Request(url, headers={"User-agent": "scanpy-user"})) as resp:
            total = resp.info().get("content-length", None)
            with tqdm(
                unit="B",
                unit_scale=True,
                miniters=1,
                unit_divisor=1024,
                total=total if total is None else int(total),
            ) as t, path.open("wb") as f:
                block = resp.read(blocksize)
                while block:
                    f.write(block)
                    blocknum += 1
                    t.update(len(block))
                    block = resp.read(blocksize)

    except (KeyboardInterrupt, Exception):
        # Make sure file doesn’t exist half-downloaded
        if path.is_file():
            path.unlink()
        raise