# -*- coding: utf-8 -*-

import numpy as np
import ot.bregman as otb
import os
import pandas as pd
import time
import scanpy as sc
import scipy.sparse as ssp 

import cospar.help_functions as hf
import cospar.plotting as CSpl


def initialize_adata_object(cell_by_feature_matrix,gene_names,time_info,
    X_clone=[],X_pca=[],X_emb=[],state_info=[],
    data_path='data',figure_path='figure',data_des='cospar'):
    """
    Initialized the :class:`~anndata.AnnData` object

    The principal components (`X_pca`), 2-d embedding (`X_emb`), and 
    state_info can be provided upfront, or generated in the next step.
    If the clonal information (X_clone) is not provided, 
    the transition map will be generated using only the state information.

    Parameters
    ---------- 
    cell_by_feature_matrix: `np.ndarray`, `sp.spmatrix`
        The (annotated) data matrix. Rows correspond to cells and columns to genes. 

    gene_names: `np.ndarray`
        An array of gene names.
    
    time_info: `np.ndarray`
        Time annotation for each cell in `str`,like 'Day27' or 'D27'.
        However, it can also contain other sample_info, 
        like 'GFP+_day27', and 'GFP-_day27'.
        
    X_clone: `sp.spmatrix` (also accpet `np.ndarray`)        
        The clonal data matrix, with the row in cell_id, and column in barcode_id.
        For evolvable barcoding, a cell may carry several different barcode_id. 

    X_pca: `np.ndarray`, optional (default: [])
        A matrix of the shape n_cell*n_pct. Create if not set. 

    X_emb: `np.ndarray`, optional (default: [])
        Two-dimensional matrix for embedding.  Create with UMAP if not set.
        It is used only for plotting after the transition map is created

    state_info: `np.ndarray`, optional (default: [])
        The classification and annotation for each cell state. 
        Create with leiden clustering if not set. 
        This will be used only after the map is created. Can be adjusted later
        
    data_path: `str`, optional (default:'data')
        A relative path to save data. The data_path name should be 
        unique to this dataset for saving all relevant results. 
        If the data folder does not existed before, create a new one.
    
    figure_path: `str`, optional (default:'figure')
        A relative path to save figures. The figure_path name should be 
        unique to this dataset for saving all relevant results. 
        If the figure folder does not existed before, create a new one.

    data_des: `str`, optional (default:'cospar')
        This is just a name to label/distinguish this data. 
        Will be used for saving the results. 
        It should be a unique name for a new dataset.
            
    Returns
    -------
    Generate an :class:`~anndata.AnnData` object with the following entries 
    obs: 'time_info', 'state_info'
    uns: 'data_des', 'data_path', 'figure_path', 'clonal_time_points'
    obsm: 'X_clone', 'X_pca', 'X_umap'
    """
   



    ### making folders
    try:
        os.mkdir(data_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(figure_path)
    except OSError as error:
        print(error)

    time_info=time_info.astype(str)

    #!mkdir -p $data_path
    adata=sc.AnnData(ssp.csr_matrix(cell_by_feature_matrix))
    adata.var_names=list(gene_names)

    if X_clone.shape[0]==0:
        X_clone=np.zeros((adata.shape[0],1))


    # we do not remove zero-sized clones here as in some case, we want 
    # to use this package even if we do not have clonal data.
    # Removing zero-sized clones will be handled downstream when we have to 
    adata.obsm['X_clone']=ssp.csr_matrix(X_clone)
    adata.obs['time_info']=pd.Categorical(time_info)
    adata.uns['data_des']=[data_des]
    adata.uns['data_path']=[data_path]
    adata.uns['figure_path']=[figure_path]

    # record time points with clonal information
    if ssp.issparse(X_clone):
        clone_N_per_cell=X_clone.sum(1).A.flatten()
    else:
        clone_N_per_cell=X_clone.sum(1)
        
    clonal_time_points=[]
    for xx in list(set(time_info)):
        idx=np.array(time_info)==xx
        if np.sum(clone_N_per_cell[idx])>0:
            clonal_time_points.append(xx)
    adata.uns['clonal_time_points']=clonal_time_points


    if len(X_pca)==adata.shape[0]:
        adata.obsm['X_pca']=np.array(X_pca)

    if len(state_info)==adata.shape[0]:
        adata.obs['state_info']=pd.Categorical(state_info)

    if len(X_emb)==adata.shape[0]:
        adata.obsm['X_umap']=X_emb

    print(f"All time points: {set(adata.obs['time_info'])}")
    print(f"Time points with clonal info: {set(adata.uns['clonal_time_points'])}")
            
    return adata


def update_X_pca(adata,normalized_counts_per_cell=10000,min_counts=3, 
    min_cells=3, min_gene_vscore_pctl=85,n_pca_comp=40):
    """
    Update X_pca

    We assume that data preprocessing are already done (e.g., via scanpy.pp), including
    removing low quality cells, regress out cell cycle effect, removing doublets etc. 
    It first perform count normalization, variable gene selection, and then compute PCA. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    normalized_counts_per_cell: int, optional (default: 1000)
        count matrix normalization 
    min_counts: int, optional (default: 3)  
        Minimum number of UMIs per cell to be considered for selecting highly variable genes. 
    min_cells: int, optional (default: 3)
        Minimum number of cells per gene to be considered for selecting highly variable genes. 
    min_gene_vscore_pctl: int, optional (default: 85)
        Genes wht a variability percentile higher than this threshold are marked as 
        highly variable genes for dimension reduction. Range: [0,100]
    n_pca_comp: int, optional (default: 40)
        Number of top principle components to keep

    Returns
    -------
    None, but the adata.obsm['X_pca'] is modified. 
    """
 
    print("X_pca is not provided or do not have the right cell number. Compute new X_pca from the feature count matrix!")
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=normalized_counts_per_cell)

    print('Finding highly variable genes...')
    gene_list=adata.var_names
    highvar_genes = gene_list[hf.filter_genes(
        adata.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_gene_vscore_pctl, 
        show_vscore_plot=True)]

    adata.var['highly_variable'] = False
    adata.var.loc[highvar_genes, 'highly_variable'] = True
    print(f'Keeping {len(highvar_genes)} genes')
    
    adata.obsm['X_pca'] = hf.get_pca(adata[:, highvar_genes].X, numpc=n_pca_comp,keep_sparse=False,normalize=True,random_state=0)


def update_X_umap(adata,n_neighbors=20,umap_min_dist=0.3):
    """
    Update X_umap using :func:`scanpy.tl.umap`

    We assume that X_pca is computed.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    n_neighbors: int, optional (default: 20)
        neighber number for constructing the KNN graph, using the UMAP method. 
    umap_min_dist: float, optional (default: 0.3)
        The effective minimum distance between embedded points. 

    Returns
    -------
    None, but the adata.obsm['X_umap'] is modified. 
    """

    if not ('X_pca' in adata.obsm.keys()):
        print('*X_pca* missing from adata.obsm... abort the operation')
    else:
        # Number of neighbors for KNN graph construction
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=umap_min_dist)


def update_state_info(adata,leiden_resolution=0.5):
    """
    Update `state_info` using :func:`scanpy.tl.leiden`

    We assume that X_pca is computed.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    n_neighbors: int, optional (default: 20)
        neighber number for constructing the KNN graph, using the UMAP method. 
    leiden_resolution: float, optional (default: 0.5)
        A parameter value controlling the coarseness of the clustering. 
        Higher values lead to more clusters.

    Returns
    -------
    None, but the adata.obs['state_info'] is modified. 
    """

    if not ('X_pca' in adata.obsm.keys()):
        print('*X_pca* missing from adata.obsm... abort the operation')
    else:
        # Number of neighbors for KNN graph construction
        sc.tl.leiden(adata,resolution=leiden_resolution)
        adata.obs['state_info']=adata.obs['leiden']


def check_adata_structure(adata):
    """
    Check whether the adata has the right structure. 

    """
    if not ('X_pca' in adata.obsm.keys()):
        print('*X_pca* missing from adata.obsm')

    if not ('X_umap' in adata.obsm.keys()):
        print('*X_umap* missing from adata.obsm')

    if not ('X_clone' in adata.obsm.keys()):
        print('*X_clone* missing from adata.obsm')

    if not ('time_info' in adata.obs.keys()):
        print('*time_info* missing from adata.obs')

    if not ('state_info' in adata.obs.keys()):
        print('*state_info* missing from adata.obs')


############# refine clusters for state_info

def refine_state_info_by_leiden_clustering(adata,selected_time_points=[],
    leiden_resolution=0.5,n_neighbors=5,confirm_change=False,cluster_name_prefix='S'):
    """
    Refine state info by clustering on states at given time points.

    Select states at desired time points to improve the clustering. When
    first run, set confirm_change=False. Only when you are happy with the 
    result, set confirm_change=True to update the adata.obs['state_info'].
    The original state_info will be stored at adata.obs['old_state_info'].  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_time_points: `list`, optional (default: include all)
        A list of selected time points for performing clustering,
        among adata.obs['time_info']. If set as [], use all time points.
    adata: :class:`~anndata.AnnData` object
    n_neighbors: `int`, optional (default: 20)
        neighber number for constructing the KNN graph, using the UMAP method. 
    leiden_resolution: `float`, optional (default: 0.5)
        A parameter value controlling the coarseness of the clustering. 
        Higher values lead to more clusters.
    confirm_change: `bool`, optional (default: False)
        If True, update adata.obs['state_info']
    cluster_name_prefix: `str`, optional (default: 'S')
        prefix for the new cluster name in case they overlap 
        with existing cluster names.

    Returns
    -------
    Update the adata.obs['state_info'] if confirm_change=True.
    """

    time_info=adata.obs['time_info']
    availabel_time_points=list(set(time_info))
    
    if len(selected_time_points)==0:
        selected_time_points=availabel_time_points

    if np.sum(np.in1d(selected_time_points,availabel_time_points))!=len(selected_time_points):
        print(f"Selected time points not available. Please select from {availabel_time_points}")

    else:
        sp_idx=np.zeros(adata.shape[0],dtype=bool)
        for xx in selected_time_points:
            idx=time_info==xx
            sp_idx[idx]=True
            
        adata_sp=sc.AnnData(adata.X[sp_idx]);
        #adata_sp.var_names=adata.var_names
        adata_sp.obsm['X_pca']=adata.obsm['X_pca'][sp_idx]
        adata_sp.obsm['X_umap']=adata.obsm['X_umap'][sp_idx]
        
        sc.pp.neighbors(adata_sp, n_neighbors=n_neighbors)
        sc.tl.leiden(adata_sp,resolution=leiden_resolution)

        sc.pl.umap(adata_sp,color='leiden')
        
        if confirm_change:
            print("Change state annotation at adata.obs['state_info']")
            adata.obs['old_state_info']=adata.obs['state_info']
            orig_state_annot=np.array(adata.obs['state_info'])
            temp_array=np.array(adata_sp.obs['leiden'])
            for j in range(len(temp_array)):
                temp_array[j]=cluster_name_prefix+temp_array[j]
            
            orig_state_annot[sp_idx]=temp_array
            adata.obs['state_info']=pd.Categorical(orig_state_annot)
            sc.pl.umap(adata,color='state_info')
        
        
    

def refine_state_info_by_marker_genes(adata,marker_genes,express_threshold=0.1,
    selected_time_points=[],new_cluster_name='new_cluster',confirm_change=False,add_neighbor_N=5):
    """
    Refine state info according to marker gene expression.

    A state is selected if it expressed all genes in the list of 
    marker_genes, and above the relative threshold express_threshold, 
    and satisfy the time point constraint. In addition, we also 
    include cell states neighboring to these valid states to smooth 
    the selection.
    

    When first run, set confirm_change=False. Only when you are happy with 
    the result, set confirm_change=True to update the adata.obs['state_info'].
    The original state_info will be stored at adata.obs['old_state_info'].  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    marker_genes: `list`
        List of marker genes to be used for defining cell states.
    express_threshold: `float`, optional (default: 0.1)
        Relative threshold of marker gene expression, in the range [0,1].
        A state must have an expression above this threshold for all genes
        to be included.
    selected_time_points: `list`, optional (default: include all)
        A list of selected time points for performing clustering,
        among adata.obs['time_info']. If set as [], use all time points.
    new_cluster_name: `str`, optional (default: 'new_cluster')
    confirm_change: `bool`, optional (default: False)
        If True, update adata.obs['state_info']
    add_neighbor_N: `int`, optional (default: 5)
        Add to the new cluster neighboring cells of a qualified 
        high-expressing state according to the KNN graph 
        with K=add_neighbor_N

    Returns
    -------
    Update the adata.obs['state_info'] if confirm_change=True.
    """
    
    time_info=adata.obs['time_info']
    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    availabel_time_points=list(set(time_info))
    
    if len(selected_time_points)==0:
        selected_time_points=availabel_time_points
        
    sp_idx=np.zeros(adata.shape[0],dtype=bool)
    for xx in selected_time_points:
        idx=time_info==xx
        sp_idx[idx]=True
        
        
    # add gene constraints 
    selected_states_idx=np.ones(adata.shape[0],dtype=bool)
    gene_list=list(adata.var_names)
    tot_name=''
    for marker_gene_temp in marker_genes:
        if marker_gene_temp in gene_list:
            expression=adata.obs_vector(marker_gene_temp)
            thresh=express_threshold*np.max(expression)
            idx=expression>thresh
            selected_states_idx=selected_states_idx & idx
            
            tot_name=tot_name+marker_gene_temp
            
    # add temporal constraint
    selected_states_idx[~sp_idx]=0    
    
    if np.sum(selected_states_idx)>0:
        # add neighboring cells to smooth selected cells (in case the expression is sparse)
        selected_states_idx=hf.add_neighboring_cells_to_a_map(selected_states_idx,adata,neighbor_N=add_neighbor_N)

        fig=plt.figure(figsize=(4,3));ax=plt.subplot(1,1,1)
        CSpl.plot_one_gene_SW(x_emb,y_emb,selected_states_idx,ax=ax)
        ax.set_title(f"{tot_name}; Selected #: {np.sum(selected_states_idx)}")
        #print(f"Selected cell state number: {np.sum(selected_states_idx)}")


        if confirm_change:
            adata.obs['old_state_info']=adata.obs['state_info']
            print("Change state annotation at adata.obs['state_info']")
            if new_cluster_name=='':
                new_cluster_name=marker_genes[0]

            orig_state_annot=np.array(adata.obs['state_info'])
            orig_state_annot[selected_states_idx]=np.array([new_cluster_name for j in range(np.sum(selected_states_idx))])
            adata.obs['state_info']=pd.Categorical(orig_state_annot)
            sc.pl.umap(adata,color='state_info')



