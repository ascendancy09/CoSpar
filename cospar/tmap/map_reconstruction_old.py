# -*- coding: utf-8 -*-

#import logging

import numpy as np
#import scipy.special
#import scipy.stats as scs
#import scipy.linalg as scl
#import sklearn
import ot.bregman as otb
#from ot.utils import dist
#import ot
import os
import pandas as pd
import time
import pdb
#from matplotlib import pyplot as plt
#from scipy.sparse import csr_matrix
import scanpy as sc
import scipy.sparse as ssp 

import cospar.help_functions as hf
import cospar.plotting as CSpl


# import clonal_OT_plotting as cot_plot
# import helper_functions_clonal_dynamics as hf



def initialize_input_adata_object(cell_by_feature_count_matrix,gene_names,time_info,cell_by_clone_matrix=np.zeros(0),
        X_pca=[],X_emb=[],state_annotation=[],data_path='',figure_path='',data_des='',
        normalized_counts_per_cell=10000,min_counts=3, min_cells=3, min_gene_variability_pctl=85,n_pca_comp=40,n_neighbors=5,umap_min_dist=0.3,leiden_resolution=0.5):
    '''
        Purpose: 
            Initialized the adata object needed for downstream analysis. 
            It is best to use your own X_pca principal components and associated 2-d embedding that you know are good.
            If not provided, this function also creates these objects using a simple pre-processing, include count normalization, feature selection, Z-score and computing PCA, UMAP embedding, leiden clustering
    
        Input: 
            cell_by_feature_count_matrix: 
                Feature matrix, with row in cell id, and column in feature id.
                For single-cell RNA seq data, the feature is gene expression. The feature matrix is the UMI count matrix.
                Used for selecting highly variable genes in constructing the HighVar_transition_map
                After the map is constructed, it is used for differential gene expression analysis

            gene_names: 
                gene names, for selecting highly variable genes, and DGE analysis
            
            time_info:
                time_info is just an annotation for each cell. 
                Typically, it contains the time information, in string data type, like 'Day27'
                However, it can also contain other sample_info, like 'GFP+_day27', and 'GFP-_day27'
                This is critical for map construction.
                
            cell_by_clone_matrix:       
                The clonal data matrix, with the row in cell_id, and column in clone_id
                Here, the clone_id is just the barcode_id. And the same cell may carry different barcode_id 
                that are integrated into its genome at different time points in the process of differentiation.
                In the case of CRISPR experiments, the barcode is each distinguishable mutations introduced over time

            X_pca: 
                Needed for construct KNN graph, for building the similarity matrix.
                It is involved also in OT_transition_map computation, and for pseudo-time ording 

            X_emb:
                The two-dimensional embedding. It can be created with UMAP, or other methods like force-directed layout.
                It is used only for plotting after the transition map is created

            state_annotation:
                The classification and annotation for each cell state.
                This will be used only after the map is created. So, it can be adjusted later
                
            data_path:
                A relative path to save data. If not existed before, create a new one.
            
            figure_path:
                A relative path to save figures. If not existed before, create a new one.
                
            data_des:
                This is just a name to indicate this data. Will be used for saving the results. Can be arbitrary
                
            If some of the above data not provided, create them with the following parameters:
                normalized_counts_per_cell: for count matrix normalized 
                n_pca_comp: number of principle components
                min_counts: a highly variable gene must be covered by at least min_counts UMI
                min_cells: a highly variable gene must be covered by at least min_cells
                min_gene_variability_pctl: the highly variable genes should be in the top min_gene_variability_pctl percentile
                n_neighbors: for building KNN graph
                umap_min_dist: for UMAP embedding. A larger value corresponds to more coare-grained embedding
                leiden_resolution: for clustering of cell states. A small value corresponds to less number of distinct clusters.

        Output:
            adata object 
    '''
   
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
    adata_orig=sc.AnnData(ssp.csr_matrix(cell_by_feature_count_matrix))
    adata_orig.var_names=list(gene_names)

    if cell_by_clone_matrix.shape[0]==0:
        cell_by_clone_matrix=np.zeros((adata_orig.shape[0],1))

    adata_orig.obsm['cell_by_clone_matrix']=ssp.csr_matrix(cell_by_clone_matrix)
    adata_orig.obs['time_info']=pd.Categorical(time_info)
    adata_orig.uns['data_des']=[data_des]
    adata_orig.uns['data_path']=[data_path]
    adata_orig.uns['figure_path']=[figure_path]

    # record time points with clonal information
    if ssp.issparse(cell_by_clone_matrix):
        clone_N_per_cell=cell_by_clone_matrix.sum(1).A.flatten()
    else:
        clone_N_per_cell=cell_by_clone_matrix.sum(1)
        
    clonal_time_points=[]
    for xx in list(set(time_info)):
        idx=np.array(time_info)==xx
        if np.sum(clone_N_per_cell[idx])>0:
            clonal_time_points.append(xx)
    adata_orig.uns['clonal_time_points']=clonal_time_points
    
    if len(X_pca)!=adata_orig.shape[0]: 
        print("X_pca is not provided or do not have the right cell number. Compute new X_pca from the feature count matrix!")
        sc.pp.normalize_per_cell(adata_orig, counts_per_cell_after=normalized_counts_per_cell)
    
        print('Finding highly variable genes...')
        gene_list=adata_orig.var_names
        highvar_genes = gene_list[hf.filter_genes(
            adata_orig.X, 
            min_counts=min_counts, 
            min_cells=min_cells, 
            min_vscore_pctl=min_gene_variability_pctl, 
            show_vscore_plot=True)]

        adata_orig.var['highly_variable'] = False
        adata_orig.var.loc[highvar_genes, 'highly_variable'] = True
        print(f'Keeping {len(highvar_genes)} genes')
        
        adata_orig.obsm['X_pca'] = hf.get_pca(adata_orig[:, highvar_genes].X, numpc=n_pca_comp,keep_sparse=False,normalize=True,random_state=0)
    else:
        adata_orig.obsm['X_pca']=np.array(X_pca)
        
    if (len(X_emb)!=adata_orig.shape[0]):
        print("The 2-d embedding is either not provided, or with wrong cell number. Create a new embedding using UMAP!")
        
        # Number of neighbors for KNN graph construction
        sc.pp.neighbors(adata_orig, n_neighbors=n_neighbors)
        sc.tl.umap(adata_orig, min_dist=umap_min_dist)

    else:
        adata_orig.obsm['X_umap']=X_emb
    
    if len(state_annotation)!=adata_orig.shape[0]:
        print("The state_annotation is either not provided, or with wrong cell number. Create a new annotation using Leiden clustering!")
        sc.pp.neighbors(adata_orig, n_neighbors=n_neighbors)
        sc.tl.leiden(adata_orig,resolution=leiden_resolution)
        adata_orig.obs['state_annotation']=adata_orig.obs['leiden']
    else:
        adata_orig.obs['state_annotation']=pd.Categorical(state_annotation)
        
    sc.pl.umap(adata_orig,color='time_info')
    sc.pl.umap(adata_orig,color='state_annotation')

    print(f"All time points: {set(adata_orig.obs['time_info'])}")
    print(f"Time points with clonal info: {set(adata_orig.uns['clonal_time_points'])}")
            
    return adata_orig


####################

# Constructing the similarity matrix (kernel matrix)

####################


def generate_full_kernel_matrix_single_v0(adata,file_name,round_of_smooth=10,neighbor_N=20,beta=0.1,truncation_threshold=0.001,save_subset=True,verbose=True,compute_new_Smatrix=False):
    '''
        At each iteration, truncate the similarity matrix (the kernel) using truncation_threshold. This promotes the sparsity of the matrix, thus the speed of computation.
        We set the truncation threshold to be small, to guarantee accracy. This method is used in our actual analysis


    '''

    if os.path.exists(file_name+f'_SM{round_of_smooth}.npz') and (not compute_new_Smatrix):
        if verbose:
            print("Compute similarity matrix: load existing data")
        kernel_matrix=ssp.load_npz(file_name+f'_SM{round_of_smooth}.npz')
    else: # compute now
        if verbose:
            print(f"Compute similarity matrix: computing new; beta={beta}")

        # add a step to compute PCA in case this is not computed 

        # here, we assume that adata already has pre-computed PCA
        sc.pp.neighbors(adata, n_neighbors=neighbor_N)

        ## compute the kernel matrix (smooth matrix)
        
        #nrow = adata.shape[0]
        #initial_clones = ssp.lil_matrix((nrow, nrow))
        #initial_clones.setdiag(np.ones(nrow))
        #kernel_matrix=hf.get_smooth_values_SW(initial_clones, adata_sp.uns['neighbors']['connectivities'], beta=0, n_rounds=round_of_smooth)
        #kernel_matrix=get_smooth_values_sparseMatrixForm(initial_clones, adata.uns['neighbors']['connectivities'], beta=0, n_rounds=round_of_smooth)
        # this kernel_matrix is column-normalized, our B here

        
        #adjacency_matrix=adata.uns['neighbors']['connectivities'];
        adjacency_matrix=adata.obsp['connectivities'];

        ############## The new method
        adjacency_matrix=(adjacency_matrix+adjacency_matrix.T)/2
        ############## 

        adjacency_matrix = hf.sparse_rowwise_multiply(adjacency_matrix, 1 / adjacency_matrix.sum(1).A.squeeze())
        nrow = adata.shape[0]
        kernel_matrix = ssp.lil_matrix((nrow, nrow))
        kernel_matrix.setdiag(np.ones(nrow))
        transpose_A=adjacency_matrix.T
        for iRound in range(round_of_smooth):
            SM=iRound+1
            if verbose:
                print("Smooth round:",SM)
            t=time.time()
            kernel_matrix =beta*kernel_matrix+(1-beta)*transpose_A*kernel_matrix
            #kernel_matrix =beta*kernel_matrix+(1-beta)*kernel_matrix*adjacency_matrix
            #kernel_matrix_array.append(kernel_matrix)
            if verbose:
                print("Time elapsed:",time.time()-t)

            t=time.time()
            sparsity_frac=(kernel_matrix>0).sum()/(kernel_matrix.shape[0]*kernel_matrix.shape[1])
            if sparsity_frac>=0.1:
                #kernel_matrix_truncate=kernel_matrix
                #kernel_matrix_truncate_array.append(kernel_matrix_truncate)
                if verbose:
                    print(f"Orignal sparsity={sparsity_frac}, Thresholding")
                kernel_matrix=hf.matrix_row_or_column_thresholding(kernel_matrix,truncation_threshold)
                sparsity_frac_2=(kernel_matrix>0).sum()/(kernel_matrix.shape[0]*kernel_matrix.shape[1])
                #kernel_matrix_truncate_array.append(kernel_matrix_truncate)
                if verbose:
                    print(f"Final sparsity={sparsity_frac_2}")
            if verbose:
                print(f"Kernel matrix truncated (SM={SM}): ", time.time()-t)

            #print("Save the matrix")
            #file_name=f'data/20200221_truncated_kernel_matrix_SM{round_of_smooth}_kNN{neighbor_N}_Truncate{str(truncation_threshold)[2:]}.npz'
            kernel_matrix=ssp.csr_matrix(kernel_matrix)


            ############## The new method
            #kernel_matrix=kernel_matrix.T.copy() 
            ##############


            if save_subset: 
                if SM%5==0: # save when SM=5,10,15,20,...
                    if verbose:
                        print("Save the matrix~~~")
                    ssp.save_npz(file_name+f'_SM{SM}.npz',kernel_matrix)
            else: # save all
                if verbose:
                    print("Save the matrix")
                ssp.save_npz(file_name+f'_SM{SM}.npz',kernel_matrix)
        

    return kernel_matrix




def generate_initial_kernel(kernel_matrix,initial_index_0,initial_index_1,verbose=True):
    
    # the input matrix can be either sparse matrix or numpy array
    # the output matrix is a numpy array
    # there is a good argument to do normalize, as the selected index is not the full space for diffusion
    
    t=time.time()
    initial_kernel=kernel_matrix[initial_index_0][:,initial_index_1];
    #initial_kernel=hf.sparse_column_multiply(initial_kernel,1/(resol+initial_kernel.sum(0)))
    if ssp.issparse(initial_kernel): initial_kernel=initial_kernel.A
    if verbose:
        print("Time elapsed: ", time.time()-t)
    return initial_kernel 


def generate_final_kernel(kernel_matrix,final_index_0,final_index_1,verbose=True):
    
    # the input matrix can be either sparse matrix or numpy array
    # the output matrix is a numpy array
    # there is a good argument to do normalize, as the selected index is not the full space for diffusion
    
    t=time.time()
    final_kernel=kernel_matrix.T[final_index_0][:,final_index_1];
    if ssp.issparse(final_kernel):final_kernel=final_kernel.A
    #final_kernel=hf.sparse_rowwise_multiply(final_kernel,1/(resol+final_kernel.sum(1)))
    if verbose:
        print("Time elapsed: ", time.time()-t)
    return final_kernel





def select_time_points_v0(adata_orig,time_point=['day_1','day_2'],verbose=True,use_all_cells=False):
    '''
        use_all_cells: use all cells at each time point. 
        we assume that the time_point are arranged in ascending order
    '''


    
    #x_emb_orig=adata_orig.obsm['X_umap'][:,0]
    #y_emb_orig=adata_orig.obsm['X_umap'][:,1]
    time_info_orig=np.array(adata_orig.obs['time_info'])
    clone_annot_orig=adata_orig.obsm['cell_by_clone_matrix']
    if len(time_point)==0: # use all clonally labelled cell states 
        time_point=np.sort(list(set(time_info_orig)))

    if (len(time_point)<2):
        print("Error! Must select more than 1 time point!")
    else:
        if len(time_point)==2:
            use_all_cells_at_t1=True
            use_all_cells_at_t2=True
        else:
            use_all_cells_at_t1=False
            use_all_cells_at_t2=False

        At=[]
        for j, time_0 in enumerate(time_point):
            At.append(ssp.csr_matrix(clone_annot_orig[time_info_orig==time_0]))

        ### Day t - t+1
        Clonal_cell_ID_FOR_t=[]
        for j in range(len(time_point)-1):
            idx_t=np.array((At[j]*At[j+1].T).sum(1)>0).flatten()
            time_index_t=time_info_orig==time_point[j]
            temp=np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_FOR_t.append(temp) # this index is in the original space, without sampling etc
            if verbose:
                print(f"Clonal cell fraction (day {time_point[j]}-{time_point[j+1]}):",len(temp)/np.sum(time_index_t))

        ### Day t+1 - t
        Clonal_cell_ID_BACK_t=[]
        for j in range(len(time_point)-1):
            idx_t=np.array((At[j+1]*At[j].T).sum(1)>0).flatten()
            time_index_t=time_info_orig==time_point[j+1]
            temp=np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_BACK_t.append(temp) # this index is in the original space, without sampling etc
            if verbose:
                print(f"Clonal cell fraction (day {time_point[j+1]}-{time_point[j]}):",len(temp)/np.sum(time_index_t))

        if verbose:
            for j in range(len(time_point)-1):    
                print(f"Numer of cells that are clonally related -- day {time_point[j]}: {len(Clonal_cell_ID_FOR_t[j])}  and day {time_point[j+1]}: {len(Clonal_cell_ID_BACK_t[j])}")

        proportion=np.ones(len(time_point))
        # flatten the list
        flatten_clonal_cell_ID_FOR=np.array([sub_item for item in Clonal_cell_ID_FOR_t for sub_item in item])
        flatten_clonal_cell_ID_BACK=np.array([sub_item for item in Clonal_cell_ID_BACK_t for sub_item in item])
        valid_clone_N_FOR=np.sum(clone_annot_orig[flatten_clonal_cell_ID_FOR].A.sum(0)>0)
        valid_clone_N_BACK=np.sum(clone_annot_orig[flatten_clonal_cell_ID_BACK].A.sum(0)>0)

        if verbose:
            print("Valid clone number 'FOR' post selection",valid_clone_N_FOR)
            #print("Valid clone number 'BACK' post selection",valid_clone_N_BACK)


        ###################### select initial and later cell states

        if use_all_cells:
            old_Tmap_cell_id_t1=[]
            for t_temp in time_point:
                old_Tmap_cell_id_t1=old_Tmap_cell_id_t1+list(np.nonzero(time_info_orig==t_temp)[0])

            old_Tmap_cell_id_t1=np.array(old_Tmap_cell_id_t1)
            old_Tmap_cell_id_t2=np.array(old_Tmap_cell_id_t1)

        else:
            if use_all_cells_at_t1:
                old_Tmap_cell_id_t1=np.nonzero(time_info_orig==time_point[0])[0]
            else:
                old_Tmap_cell_id_t1=flatten_clonal_cell_ID_FOR

            if use_all_cells_at_t2:
                old_Tmap_cell_id_t2=np.nonzero(time_info_orig==time_point[1])[0]
            else:
                old_Tmap_cell_id_t2=flatten_clonal_cell_ID_BACK


        old_clonal_cell_id_t1=flatten_clonal_cell_ID_FOR
        old_clonal_cell_id_t2=flatten_clonal_cell_ID_BACK
        ########################

        sp_id=np.sort(list(set(list(old_Tmap_cell_id_t1)+list(old_Tmap_cell_id_t2))))
        sp_idx=np.zeros(clone_annot_orig.shape[0],dtype=bool)
        sp_idx[sp_id]=True

        Tmap_cell_id_t1=hf.converting_id_from_fullSpace_to_subSpace(old_Tmap_cell_id_t1,sp_id)[0]
        clonal_cell_id_t1=hf.converting_id_from_fullSpace_to_subSpace(old_clonal_cell_id_t1,sp_id)[0]
        clonal_cell_id_t2=hf.converting_id_from_fullSpace_to_subSpace(old_clonal_cell_id_t2,sp_id)[0]
        Tmap_cell_id_t2=hf.converting_id_from_fullSpace_to_subSpace(old_Tmap_cell_id_t2,sp_id)[0]

        Clonal_cell_ID_FOR_t_new=[]
        for temp_id_list in Clonal_cell_ID_FOR_t:
            convert_list=hf.converting_id_from_fullSpace_to_subSpace(temp_id_list,sp_id)[0]
            Clonal_cell_ID_FOR_t_new.append(convert_list)

        Clonal_cell_ID_BACK_t_new=[]
        for temp_id_list in Clonal_cell_ID_BACK_t:
            convert_list=hf.converting_id_from_fullSpace_to_subSpace(temp_id_list,sp_id)[0]
            Clonal_cell_ID_BACK_t_new.append(convert_list)


        sp_id_0=np.sort(list(old_clonal_cell_id_t1)+list(old_clonal_cell_id_t2))
        sp_idx_0=np.zeros(clone_annot_orig.shape[0],dtype=bool)
        sp_idx_0[sp_id_0]=True

        barcode_id=np.nonzero(clone_annot_orig[sp_idx_0].A.sum(0).flatten()>0)[0]
        #sp_id=np.nonzero(sp_idx)[0]
        clone_annot=clone_annot_orig[sp_idx][:,barcode_id]

        adata=sc.AnnData(adata_orig.X[sp_idx]);
        adata.var_names=adata_orig.var_names
        adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
        adata.obs['state_annotation']=pd.Categorical(adata_orig.obs['state_annotation'][sp_idx])
        adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        adata.uns['data_path']=adata_orig.uns['data_path']
        adata.uns['figure_path']=adata_orig.uns['figure_path']

        adata.obsm['cell_by_clone_matrix']=clone_annot
        adata.uns['clonal_cell_id_t1']=clonal_cell_id_t1
        adata.uns['clonal_cell_id_t2']=clonal_cell_id_t2
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['multiTime_cell_id_t1']=Clonal_cell_ID_FOR_t_new
        adata.uns['multiTime_cell_id_t2']=Clonal_cell_ID_BACK_t_new
        adata.uns['proportion']=np.ones(len(time_point)-1)
        adata.uns['sp_idx']=sp_idx

        data_des_0=adata_orig.uns['data_des'][0]
        time_label='t'
        for x in time_point:
            time_label=time_label+f'*{x}'

        data_des=data_des_0+f'_TwoTimeClone_{time_label}'
        adata.uns['data_des']=[data_des]

        if verbose:
            N_cell,N_clone=clone_annot.shape;
            print(f"Cell number={N_cell}, Clone number={N_clone}")
            x_emb=adata.obsm['X_umap'][:,0]
            y_emb=adata.obsm['X_umap'][:,1]
            CSpl.plot_one_gene_SW(x_emb,y_emb,-x_emb)

        return adata        



def select_time_points(adata_orig,time_point=['day_1','day_2'],verbose=True,use_all_cells=False):
    '''
        use_all_cells: use all cells at each time point. 
        we assume that the time_point are arranged in ascending order
    '''


    
    #x_emb_orig=adata_orig.obsm['X_umap'][:,0]
    #y_emb_orig=adata_orig.obsm['X_umap'][:,1]
    time_info_orig=np.array(adata_orig.obs['time_info'])
    clone_annot_orig=adata_orig.obsm['cell_by_clone_matrix']
    if len(time_point)==0: # use all clonally labelled cell states 
        time_point=np.sort(list(set(time_info_orig)))

    if (len(time_point)<2):
        print("Error! Must select more than 1 time point!")
    else:

        At=[]
        for j, time_0 in enumerate(time_point):
            At.append(ssp.csr_matrix(clone_annot_orig[time_info_orig==time_0]))

        ### Day t - t+1
        Clonal_cell_ID_FOR_t=[]
        for j in range(len(time_point)-1):
            idx_t=np.array((At[j]*At[j+1].T).sum(1)>0).flatten()
            time_index_t=time_info_orig==time_point[j]
            temp=np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_FOR_t.append(temp) # this index is in the original space, without sampling etc
            if verbose:
                print(f"Clonal cell fraction (day {time_point[j]}-{time_point[j+1]}):",len(temp)/np.sum(time_index_t))

        ### Day t+1 - t
        Clonal_cell_ID_BACK_t=[]
        for j in range(len(time_point)-1):
            idx_t=np.array((At[j+1]*At[j].T).sum(1)>0).flatten()
            time_index_t=time_info_orig==time_point[j+1]
            temp=np.nonzero(time_index_t)[0][idx_t]
            Clonal_cell_ID_BACK_t.append(temp) # this index is in the original space, without sampling etc
            if verbose:
                print(f"Clonal cell fraction (day {time_point[j+1]}-{time_point[j]}):",len(temp)/np.sum(time_index_t))

        if verbose:
            for j in range(len(time_point)-1):    
                print(f"Numer of cells that are clonally related -- day {time_point[j]}: {len(Clonal_cell_ID_FOR_t[j])}  and day {time_point[j+1]}: {len(Clonal_cell_ID_BACK_t[j])}")

        proportion=np.ones(len(time_point))
        # flatten the list
        flatten_clonal_cell_ID_FOR=np.array([sub_item for item in Clonal_cell_ID_FOR_t for sub_item in item])
        flatten_clonal_cell_ID_BACK=np.array([sub_item for item in Clonal_cell_ID_BACK_t for sub_item in item])
        valid_clone_N_FOR=np.sum(clone_annot_orig[flatten_clonal_cell_ID_FOR].A.sum(0)>0)
        valid_clone_N_BACK=np.sum(clone_annot_orig[flatten_clonal_cell_ID_BACK].A.sum(0)>0)

        if verbose:
            print("Valid clone number 'FOR' post selection",valid_clone_N_FOR)
            #print("Valid clone number 'BACK' post selection",valid_clone_N_BACK)


        ###################### select initial and later cell states

        if use_all_cells:
            old_Tmap_cell_id_t1=[]
            for t_temp in time_point[:-1]:
                old_Tmap_cell_id_t1=old_Tmap_cell_id_t1+list(np.nonzero(time_info_orig==t_temp)[0])
            old_Tmap_cell_id_t1=np.array(old_Tmap_cell_id_t1)

            ########
            old_Tmap_cell_id_t2=[]
            for t_temp in time_point[1:]:
                old_Tmap_cell_id_t2=old_Tmap_cell_id_t2+list(np.nonzero(time_info_orig==t_temp)[0])
            old_Tmap_cell_id_t2=np.array(old_Tmap_cell_id_t2)

        else:
            old_Tmap_cell_id_t1=flatten_clonal_cell_ID_FOR
            old_Tmap_cell_id_t2=flatten_clonal_cell_ID_BACK


        old_clonal_cell_id_t1=flatten_clonal_cell_ID_FOR
        old_clonal_cell_id_t2=flatten_clonal_cell_ID_BACK
        ########################

        sp_id=np.sort(list(set(list(old_Tmap_cell_id_t1)+list(old_Tmap_cell_id_t2))))
        sp_idx=np.zeros(clone_annot_orig.shape[0],dtype=bool)
        sp_idx[sp_id]=True

        Tmap_cell_id_t1=hf.converting_id_from_fullSpace_to_subSpace(old_Tmap_cell_id_t1,sp_id)[0]
        clonal_cell_id_t1=hf.converting_id_from_fullSpace_to_subSpace(old_clonal_cell_id_t1,sp_id)[0]
        clonal_cell_id_t2=hf.converting_id_from_fullSpace_to_subSpace(old_clonal_cell_id_t2,sp_id)[0]
        Tmap_cell_id_t2=hf.converting_id_from_fullSpace_to_subSpace(old_Tmap_cell_id_t2,sp_id)[0]

        Clonal_cell_ID_FOR_t_new=[]
        for temp_id_list in Clonal_cell_ID_FOR_t:
            convert_list=hf.converting_id_from_fullSpace_to_subSpace(temp_id_list,sp_id)[0]
            Clonal_cell_ID_FOR_t_new.append(convert_list)

        Clonal_cell_ID_BACK_t_new=[]
        for temp_id_list in Clonal_cell_ID_BACK_t:
            convert_list=hf.converting_id_from_fullSpace_to_subSpace(temp_id_list,sp_id)[0]
            Clonal_cell_ID_BACK_t_new.append(convert_list)


        sp_id_0=np.sort(list(old_clonal_cell_id_t1)+list(old_clonal_cell_id_t2))
        sp_idx_0=np.zeros(clone_annot_orig.shape[0],dtype=bool)
        sp_idx_0[sp_id_0]=True

        barcode_id=np.nonzero(clone_annot_orig[sp_idx_0].A.sum(0).flatten()>0)[0]
        #sp_id=np.nonzero(sp_idx)[0]
        clone_annot=clone_annot_orig[sp_idx][:,barcode_id]

        adata=sc.AnnData(adata_orig.X[sp_idx]);
        adata.var_names=adata_orig.var_names
        adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
        adata.obs['state_annotation']=pd.Categorical(adata_orig.obs['state_annotation'][sp_idx])
        adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        adata.uns['data_path']=adata_orig.uns['data_path']
        adata.uns['figure_path']=adata_orig.uns['figure_path']

        adata.obsm['cell_by_clone_matrix']=clone_annot
        adata.uns['clonal_cell_id_t1']=clonal_cell_id_t1
        adata.uns['clonal_cell_id_t2']=clonal_cell_id_t2
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['multiTime_cell_id_t1']=Clonal_cell_ID_FOR_t_new
        adata.uns['multiTime_cell_id_t2']=Clonal_cell_ID_BACK_t_new
        adata.uns['proportion']=np.ones(len(time_point)-1)
        adata.uns['sp_idx']=sp_idx

        data_des_0=adata_orig.uns['data_des'][0]
        time_label='t'
        for x in time_point:
            time_label=time_label+f'*{x}'

        data_des=data_des_0+f'_TwoTimeClone_{time_label}'
        adata.uns['data_des']=[data_des]

        if verbose:
            N_cell,N_clone=clone_annot.shape;
            print(f"Cell number={N_cell}, Clone number={N_clone}")
            x_emb=adata.obsm['X_umap'][:,0]
            y_emb=adata.obsm['X_umap'][:,1]
            CSpl.plot_one_gene_SW(x_emb,y_emb,-x_emb)

        return adata        



####################

# CoSpar: two-time points

####################



def refine_clonal_map_by_integrating_clonal_info(cell_id_array_t1,cell_id_array_t2,clonal_map,cell_by_clone_matrix,initial_kernel,final_kernel,noise_threshold,row_normalize=True,verbose=True,normalization_mode=1):

    '''
        New version: allow cells to belong to different clones; Add mode (MultiTime)
    '''

    print("New version: allow cells to belong to different clones; Add mode (singleTime)")

    resol=10**(-10)
    clonal_map=hf.matrix_row_or_column_thresholding(clonal_map,noise_threshold,row_threshold=True)

    if verbose:
        if row_normalize: 
            if normalization_mode==0: print("Single-cell normalization")
            if normalization_mode==1: print("Clone normalization: N2/N1")
            if normalization_mode==2: print("Clone normalization: N2")

        else: print("No normalization")

    if ssp.issparse(cell_by_clone_matrix):
        cell_by_clone_matrix=ssp.csr_matrix(cell_by_clone_matrix)

    cell_N,clone_N=cell_by_clone_matrix.shape
    N1=cell_id_array_t1.shape[0]
    N2=cell_id_array_t2.shape[0]
    new_coupling_matrix=ssp.lil_matrix((N1,N2))
    for clone_id in range(clone_N):
        if verbose:
            if clone_id%1000==0: print("Clone id:",clone_id)
        idx1=cell_by_clone_matrix[cell_id_array_t1,clone_id].A.flatten()
        idx2=cell_by_clone_matrix[cell_id_array_t2,clone_id].A.flatten()
        if idx1.sum()>0 and idx2.sum()>0:
            #pdb.set_trace()
            id_1=np.nonzero(idx1)[0]
            id_2=np.nonzero(idx2)[0]
            prob=clonal_map[id_1][:,id_2]

            
            ## row thresholding 
            # prob=matrix_row_or_column_thresholding(prob,noise_threshold,row_threshold=True)
        
            ## column normalization within each clone
            # prob=hf.sparse_column_multiply(prob,1/(resol+np.sum(prob,0)))
            ## potential problem: if a can generate b with a very small probability, and this is the only case b is generated; Then row normalization gives a->b a weight of 1, which is un-reasonable

            ## try row normalization
            if row_normalize:
                #if clone_id==8:pdb.set_trace()
                if normalization_mode==0:
                    prob=hf.sparse_rowwise_multiply(prob,1/(resol+np.sum(prob,1))) # cell-level normalization
                elif normalization_mode==1:
                    temp_Nt2=np.sum(idx2>0)
                    temp_Nt1=np.sum(idx1>0)
                    prob=prob*temp_Nt2/(temp_Nt1*(resol+np.sum(prob))) # clone level normalization, account for proliferation
                elif normalization_mode==2:
                    temp_Nt2=np.sum(idx2>0)
                    prob=prob*temp_Nt2/(resol+np.sum(prob)) # clone level normalization, account for proliferation




            # if (verbose==1) and (np.sum(prob)>0):
            #     #pdb.set_trace()
            #     xx=prob.sum(0)
            #     yy=xx[xx>0]
            #     print("Negative log likelihood:",-np.sum(np.log(yy)))
            #     verbose=0

            ## update the new_coupling matrix
            #id_1=np.nonzero(idx1)[0]
            #id_2=np.nonzero(idx2)[0]
            #pdb.set_trace()

            weight_factor=np.sqrt(np.mean(idx1[idx1>0])*np.mean(idx2[idx2>0])) # the contribution of a particular clone can be tuned by its average entries
            if verbose and (weight_factor>1):
                print("marker gene weight",weight_factor)

            ############## New edition  
            #new_coupling_matrix[id_1[:,np.newaxis],id_2]=prob*weight_factor  
            new_coupling_matrix[id_1[:,np.newaxis],id_2]=new_coupling_matrix[id_1[:,np.newaxis],id_2]+prob*weight_factor
            

    ## rescale
    #pdb.set_trace()
    #new_coupling_matrix=new_coupling_matrix/(new_coupling_matrix.A.max())

    ## convert to sparse matrix form
    new_coupling_matrix=new_coupling_matrix.tocsr()

    if verbose:
        print("Start to smooth the refined clonal map")
        t=time.time()
    temp=new_coupling_matrix*final_kernel
    if verbose:
        print("Phase I: time elapsed -- ", time.time()-t)
    smoothed_new_clonal_map=initial_kernel.dot(temp)
    if verbose:
        print("Phase II: time elapsed -- ", time.time()-t)

    #if row_normalize:
    #    smoothed_new_clonal_map=hf.sparse_rowwise_multiply(smoothed_new_clonal_map,1/(resol+np.sum(smoothed_new_clonal_map,1)))

    # both return are numpy array
    return smoothed_new_clonal_map, new_coupling_matrix.A






def refine_clonal_map_by_integrating_clonal_info_multiTime(MultiTime_cell_id_array_t1,MultiTime_cell_id_array_t2,proportion,clonal_map,cell_by_clone_matrix,initial_kernel,final_kernel,noise_threshold,row_normalize=True,verbose=True,normalization_mode=0):

    '''
        New version: allow cells to belong to different clones; Add mode (MultiTime)
    '''

    resol=10**(-10)
    clonal_map=hf.matrix_row_or_column_thresholding(clonal_map,noise_threshold,row_threshold=True)

    if verbose:
        if row_normalize: 
            if normalization_mode==0: print("Single-cell normalization")
            if normalization_mode==1: print("Clone normalization: N2/N1")
            if normalization_mode==2: print("Clone normalization: N2")
        else: print("No normalization")

    #verbose=1;
    if ssp.issparse(cell_by_clone_matrix):
        cell_by_clone_matrix=ssp.csr_matrix(cell_by_clone_matrix)

    cell_N,clone_N=cell_by_clone_matrix.shape
    N1,N2=clonal_map.shape
    new_coupling_matrix=ssp.lil_matrix((N1,N2))

    offset_N1=0; # cell id order in the kernel matrix is obtained by concatenating the cell id list in MultiTime_cell_id_array_t1. So, we need to offset the id if we move to the next list
    offset_N2=0;
    for j in range(len(MultiTime_cell_id_array_t1)):
        if verbose:
            print("Time point pair index:",j)
        cell_id_array_t1=MultiTime_cell_id_array_t1[j]
        cell_id_array_t2=MultiTime_cell_id_array_t2[j]


        for clone_id in range(clone_N):
            #pdb.set_trace()
            if verbose:
                if clone_id%1000==0: print("Clone id:",clone_id)
            idx1=cell_by_clone_matrix[cell_id_array_t1,clone_id].A.flatten()
            idx2=cell_by_clone_matrix[cell_id_array_t2,clone_id].A.flatten()
            if idx1.sum()>0 and idx2.sum()>0:
                ## update the new_coupling matrix
                id_1=offset_N1+np.nonzero(idx1)[0]
                id_2=offset_N2+np.nonzero(idx2)[0]
                prob=clonal_map[id_1][:,id_2]
                
                ## row thresholding 
                # prob=matrix_row_or_column_thresholding(prob,noise_threshold,row_threshold=True)
            
                ##column normalization within each clone
                #prob=hf.sparse_column_multiply(prob,1/(resol+np.sum(prob,0)))
                ## potential problem: if a can generate b with a very small probability, and this is the only case b is generated; Then row normalization gives a->b a weight of 1, which is un-reasonable


                ## try row normalization
                if row_normalize:
                    if normalization_mode==0:
                        prob=hf.sparse_rowwise_multiply(prob,1/(resol+np.sum(prob,1))) # cell-level normalization
                    elif normalization_mode==1:
                        temp_Nt2=np.sum(idx2>0)
                        temp_Nt1=np.sum(idx1>0)
                        prob=prob*temp_Nt2/(temp_Nt1*(resol+np.sum(prob))) # clone level normalization, account for proliferation
                    elif normalization_mode==2:
                        temp_Nt2=np.sum(idx2>0)
                        prob=prob*temp_Nt2/(resol+np.sum(prob)) # clone level normalization, account for proliferation



                # if (verbose==1) and (np.sum(prob)>0):
                #     xx=prob.sum(0)
                #     print("Negative log likelihood:",-np.sum(np.log(xx)))
                #     verbose=0

                #pdb.set_trace()
                weight_factor=np.sqrt(np.mean(idx1[idx1>0])*np.mean(idx2[idx2>0])) # the contribution of a particular clone can be tuned by its average entries
                if verbose and (weight_factor>1):
                    print("marker gene weight",weight_factor)

                #new_coupling_matrix[id_1[:,np.newaxis],id_2]=proportion[j]*prob*weight_factor
                ############## New edition  
                #new_coupling_matrix[id_1[:,np.newaxis],id_2]=proportion[j]*prob*weight_factor 
                new_coupling_matrix[id_1[:,np.newaxis],id_2]=new_coupling_matrix[id_1[:,np.newaxis],id_2]+proportion[j]*prob*weight_factor 

        ## update offset
        offset_N1=offset_N1+len(cell_id_array_t1)
        offset_N2=offset_N2+len(cell_id_array_t2)
            

    ## rescale
    new_coupling_matrix=new_coupling_matrix/(new_coupling_matrix.A.max())

    ## convert to sparse matrix form
    new_coupling_matrix=new_coupling_matrix.tocsr()

    if verbose:
        print("Start to smooth the refined clonal map")
    t=time.time()
    temp=new_coupling_matrix*final_kernel
    if verbose:
        print("Phase I: time elapsed -- ", time.time()-t)
    smoothed_new_clonal_map=initial_kernel.dot(temp)
    if verbose:
        print("Phase II: time elapsed -- ", time.time()-t)

    # both return are numpy array
    return smoothed_new_clonal_map, new_coupling_matrix.A




def refine_clonal_map_by_integrating_clonal_info_multiTime_No_SM(MultiTime_cell_id_array_t1,MultiTime_cell_id_array_t2,proportion,clonal_map,cell_by_clone_matrix,intra_clone_threshold=0,noise_threshold=0.1,normalization_mode=0,verbose=True):
    # updated on Nov 10, 2020

    '''
        This is the same as 'refine_clonal_map_by_integrating_clonal_info_multiTime', except that there is a local thresholding controlled by 'intra_clone_threshold', and there is no smoothing afterwards

        noise_threshold: across all cell states (for transition profiles starting from a given state)
        intra_clone_threshold: clone-specific 

    '''

    if not isinstance(cell_by_clone_matrix[0,0], bool):
        cell_by_clone_matrix=cell_by_clone_matrix.astype(bool)

    resol=10**(-10)
    if verbose:
        if normalization_mode==0: print("Single-cell normalization")
        if normalization_mode==1: print("Clone normalization: N2/N1")
        if normalization_mode==2: print("Clone normalization: N2")

    clonal_map=hf.matrix_row_or_column_thresholding(clonal_map,noise_threshold,row_threshold=True)
    
    if not ssp.issparse(clonal_map): clonal_map=ssp.csr_matrix(clonal_map)
    if not ssp.issparse(cell_by_clone_matrix): cell_by_clone_matrix=ssp.csr_matrix(cell_by_clone_matrix)

    cell_N,clone_N=cell_by_clone_matrix.shape
    N1,N2=clonal_map.shape
    new_coupling_matrix=ssp.lil_matrix((N1,N2))

    offset_N1=0; # cell id order in the kernel matrix is obtained by concatenating the cell id list in MultiTime_cell_id_array_t1. So, we need to offset the id if we move to the next list
    offset_N2=0;
    for j in range(len(MultiTime_cell_id_array_t1)):
        if verbose:
            print("Time point pair index:",j)
        cell_id_array_t1=MultiTime_cell_id_array_t1[j]
        cell_id_array_t2=MultiTime_cell_id_array_t2[j]


        for clone_id in range(clone_N):
            if verbose:
                if clone_id%1000==0: print("Clone id:",clone_id)
            idx1=cell_by_clone_matrix[cell_id_array_t1,clone_id].A.flatten()
            idx2=cell_by_clone_matrix[cell_id_array_t2,clone_id].A.flatten()
            if idx1.sum()>0 and idx2.sum()>0:
                ## update the new_coupling matrix
                id_1=offset_N1+np.nonzero(idx1)[0]
                id_2=offset_N2+np.nonzero(idx2)[0]
                prob=clonal_map[id_1][:,id_2].A
                
                ## row thresholding 
                # prob=matrix_row_or_column_thresholding(prob,noise_threshold,row_threshold=True)
            
                ##column normalization within each clone
                #prob=hf.sparse_column_multiply(prob,1/(resol+np.sum(prob,0)))
                ## potential problem: if a can generate b with a very small probability, and this is the only case b is generated; Then row normalization gives a->b a weight of 1, which is un-reasonable

                ## try row normalization
                #prob=hf.sparse_rowwise_multiply(prob,1/(resol+np.sum(prob,1)))
                # temp_Nt2=np.sum(idx2>0)
                # temp_Nt1=np.sum(idx1>0)
                # prob=prob*temp_Nt2/(temp_Nt1*(resol+np.sum(prob))) # clone level normalization, account for proliferation


                 ## try row normalization
                if normalization_mode==0:
                    prob=hf.sparse_rowwise_multiply(prob,1/(resol+np.sum(prob,1))) # cell-level normalization
                elif normalization_mode==1:
                    temp_Nt2=np.sum(idx2>0)
                    temp_Nt1=np.sum(idx1>0)
                    prob=prob*temp_Nt2/(temp_Nt1*(resol+np.sum(prob))) # clone level normalization, account for proliferation
                elif normalization_mode==2:
                    temp_Nt2=np.sum(idx2>0)
                    prob=prob*temp_Nt2/(resol+np.sum(prob)) # clone level normalization, account for proliferation


                ## local thresholding
                threshold_value=intra_clone_threshold*np.max(prob)
                idx=(prob<threshold_value)
                prob[idx]=0


                weight_factor=np.sqrt(np.mean(idx1[idx1>0])*np.mean(idx2[idx2>0])) # the contribution of a particular clone can be tuned by its average entries
                if verbose and (weight_factor>1):
                    print("marker gene weight",weight_factor)
                #print("marker gene weight",weight_factor)
                ############## New edition  
                #new_coupling_matrix[id_1[:,np.newaxis],id_2]=proportion[j]*prob
                new_coupling_matrix[id_1[:,np.newaxis],id_2]=new_coupling_matrix[id_1[:,np.newaxis],id_2]+proportion[j]*prob*weight_factor 

        ## update offset
        offset_N1=offset_N1+len(cell_id_array_t1)
        offset_N2=offset_N2+len(cell_id_array_t2)
            

    ## convert to sparse matrix form
    new_coupling_matrix=new_coupling_matrix.tocsr()
    #

    return  new_coupling_matrix


###############

def CoSpar_TwoTimeClones(adata_orig,selected_clonal_time_points=['1','2'],SM_array=[15,10,5],CoSpar_KNN=20,noise_threshold=0.1,demulti_threshold=0.05,normalization_mode=0,use_all_cells=False,save_subset=True,use_full_kernel=False,verbose=True,trunca_threshold=0.001,compute_new_Smatrix=False):
    '''
        Purpose: 
            Compute transition map for re-sampled clonal data with both stte and lineage information. We assume that the lineage information spans at least two time points.

        Input:
            selected_clonal_time_points: the time point to be included for analysis. It should be in ascending order: 'day_1','day_2'.... We assume that it is string

            The adata object should encode the following information
                
                cell_by_clone_matrix: clone-by-cell matrix at adata.obsm['cell_by_clone_matrix']
                
                Day: a time information at adata_orig.obs['time_info']
                
                X_pca: for knn graph construction, at adata_orig.obsm['X_pca']
                
                state_annotation: state annotation (e.g., in terms of clusters), at adata_orig.obs['state_annotation'], as categorical variables
                
                X_umap: an embedding, does not have to be from UMAP. But the embedding should be saved to this place by adata_rig.obsm['X_umap']=XXX
                
                data_des: a description of the data, for saving downstream results, at adata_orig.uns['data_des'][0], i.e., it should be a list, or adata_orig.uns['data_des']=[XXX]. This is a compromise because once we save adata object, it becomes a np.array. 
                
                data_path: a path string for saving data, accessed as adata_orig.uns['data_path'][0], i.e., it should be a list
                
                figure_path: a path string for saving figures, adata_orig.uns['figure_path'][0]


            normalization_mode: 0, the default, treat T as the transition probability matrix, row normalized;  
                                1, add modulation to the transition probability according to proliferation
            

            SM_array: the length of this list determines the total number of iteration in CoSpar, and the corresponding entry determines the rounds of graph diffusion in generating the corresponding similarity matrix used for that iteration
            
            CoSpar_KNN: the number of neighbors for KNN graph used for computing the similarity matrix
            
            use_full_kernel: True, we use all available cell states to generate the similarity matrix. We then sub-sample cell states that are relevant for downstream analysis from this full matrix. This tends to be more accurate.
                             False, we only use cell states for the selected time points to generate the similarity matrix. This is faster, yet could be less accurate.

            use_all_cells: True, infer the transition map among all cell states in the selected time points (t1+t2)-(t1+t2) matrix.  This can be very slow for large datasets. 
                           False, infer the transition map for cell states that are clonally labeled (except that you only select two time points.)

            verbose: True, print information of the analysis; False, do not print.
            
            trunca_threshold: this value is only for reducing the computed matrix size for saving. We set entries to zero in the similarity matrix that are smaller than this threshold. This threshld should be small, but not too small. 
            
            save_subset: True, save only similarity matrix corresponds to 5*n rounds of graph diffusion, where n is an integer; 
                          False, save all similarity matrix with any rounds generated in the process


        return:
            adata object that includes the transition map at adata.uns['transition_map'].
            This adata is different from the original one: adata_orig.

    '''

    for xx in selected_clonal_time_points:
        if xx not in adata_orig.uns['clonal_time_points']:
            print(f"Error! 'selected_clonal_time_points' contain time points without clonal information. Please set clonal_time_point to be at least two of {adata_orig.uns['clonal_time_points']}. If there is only one clonal time point, plesae run ----cospar.tmap.CoSpar_OneTimeClones----")
            return adata_orig


    if verbose:
        print("-------Step 1: Select time points---------")
    data_path=adata_orig.uns['data_path'][0]
    adata=select_time_points(adata_orig,time_point=selected_clonal_time_points,verbose=verbose,use_all_cells=use_all_cells)

    if verbose:
        print("-------Step 2: Compute the full Similarity matrix if necessary---------")

    if use_full_kernel: # prepare the kernel matrix with all state info, all subsequent kernel will be down-sampled from this one.

        temp_str='0'+str(trunca_threshold)[2:]
        round_of_smooth=np.max(SM_array)

        kernel_file_name=f'{data_path}/Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}_v0_fullkernel{use_full_kernel}'
        if not (os.path.exists(kernel_file_name+f'_SM{round_of_smooth}.npz') and (not compute_new_Smatrix)):
            kernel_matrix_full=generate_full_kernel_matrix_single_v0(adata_orig,kernel_file_name,round_of_smooth=round_of_smooth,
                        neighbor_N=CoSpar_KNN,truncation_threshold=trunca_threshold,save_subset=True,verbose=verbose,compute_new_Smatrix=compute_new_Smatrix)
    if verbose:
        print("-------Step 3: Optimize the transition map recursively---------")
    CoSpar_TwoTimeClones_private(adata,SM_array=SM_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,demulti_threshold=demulti_threshold,normalization_mode=normalization_mode,
            save_subset=save_subset,use_full_kernel=use_full_kernel,verbose=verbose,trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix)


    return adata
    

def CoSpar_TwoTimeClones_private(adata,SM_array=[15,10,5],neighbor_N=20,noise_threshold=0.1,demulti_threshold=0.05,normalization_mode=0,save_subset=True,use_full_kernel=False,verbose=True,trunca_threshold=0.001,compute_new_Smatrix=False):
    '''
        This function is for internal use. 

        Input:
            The adata object should encode the following information
                cell_by_clone_matrix: clone-by-cell matrix at adata.obsm['cell_by_clone_matrix']
                Day: a time information at adata_orig.obs['time_info']
                X_pca: for knn graph construction, at adata_orig.obsm['X_pca']
                state_annotation: state annotation (e.g., in terms of clusters), at adata_orig.obs['state_annotation'], as categorical variables
                X_umap: an embedding, does not have to be from UMAP. But the embedding should be saved to this place by adata_rig.obsm['X_umap']=XXX
                data_des: a description of the data, for saving downstream results, at adata_orig.uns['data_des'][0], i.e., it should be a list, or adata_orig.uns['data_des']=[XXX]. This is a compromise because once we save adata object, it becomes a np.array. 
                data_path: a path string for saving data, accessed as adata_orig.uns['data_path'][0], i.e., it should be a list
                figure_path: a path string for saving figures, adata_orig.uns['figure_path'][0]


            normalization_mode: 0,1 
                                0, the default, treat T as the transition probability matrix, row normalized;  
                                1, add modulation to the transition probability according to proliferation)
            

            SM_array: the length of this list determines the total number of iteration in CoSpar, and the corresponding entry determines the rounds of graph diffusion in generating the corresponding similarity matrix used for that iteration
            neighbor_N: the number of neighbors for KNN graph used for computing the similarity matrix
            use_full_kernel: True, we use all available cell states to generate the similarity matrix. We then sub-sample cell states that are relevant for downstream analysis from this full matrix. This tends to be more accurate.
                             False, we only use cell states for the selected time points to generate the similarity matrix. This is faster, yet could be less accurate.


            verbose: True, print information of the analysis; False, do not print.
            trunca_threshold: this value is only for reducing the computed matrix size for saving. We set entries to zero in the similarity matrix that are smaller than this threshold. This threshld should be small, but not too small. 
            save_subset: True, save only similarity matrix corresponds to 5*n rounds of graph diffusion, where n is an integer; 
                          False, save all similarity matrix with any rounds generated in the process


        return:
            all information will be attached to the same adata object as input

    '''


    ########## extract data
    clone_annot=adata.obsm['cell_by_clone_matrix']
    clonal_cell_id_t1=adata.uns['clonal_cell_id_t1']
    clonal_cell_id_t2=adata.uns['clonal_cell_id_t2']
    Tmap_cell_id_t1=adata.uns['Tmap_cell_id_t1']
    Tmap_cell_id_t2=adata.uns['Tmap_cell_id_t2']
    sp_idx=adata.uns['sp_idx']
    data_des=adata.uns['data_des'][0]
    multiTime_cell_id_t1=adata.uns['multiTime_cell_id_t1']
    multiTime_cell_id_t2=adata.uns['multiTime_cell_id_t2']
    proportion=adata.uns['proportion']
    data_path=adata.uns['data_path'][0]

    #########



    intra_clone_threshold=0
    
    ########################### Compute the transition map 
    if verbose:
        print("---------Compute the transition map-----------")

    #trunca_threshold=0.001 # this value is only for reducing the computed matrix size for saving
    temp_str='0'+str(trunca_threshold)[2:]

    if use_full_kernel:
        kernel_file_name=f'{data_path}/Similarity_matrix_with_all_cell_states_kNN{neighbor_N}_Truncate{temp_str}_v0_fullkernelTrue'
        for round_of_smooth in SM_array:
            if not os.path.exists(kernel_file_name+f'_SM{round_of_smooth}.npz'):
                print(f"Error! Similarity matrix at given parameters have not been computed before! Name: {kernel_file_name}")     
                return   

    else:
        kernel_file_name=f'{data_path}/{data_des}_Similarity_matrix_with_states_kNN{neighbor_N}_Truncate{temp_str}_v0_fullkernelFalse'

    initial_kernel_array=[]
    final_kernel_array=[]
    initial_kernel_array_ext=[]
    final_kernel_array_ext=[]

    for round_of_smooth in SM_array:
        # we cannot force it to compute new at this time. Otherwise, if we use_full_kernel, the resulting kernel is actually from adata, thus not full kernel. 

        re_compute=(not use_full_kernel) and (compute_new_Smatrix) # re-compute only when not using full kernel 
        kernel_matrix_full=generate_full_kernel_matrix_single_v0(adata,kernel_file_name,round_of_smooth=round_of_smooth,
                    neighbor_N=neighbor_N,truncation_threshold=trunca_threshold,save_subset=save_subset,verbose=verbose,compute_new_Smatrix=re_compute)

        if use_full_kernel:
            #pdb.set_trace()
            kernel_matrix_full_sp=kernel_matrix_full[sp_idx][:,sp_idx]

            #pdb.set_trace()
            ### extended similarity matrix
            initial_kernel_ext=generate_initial_kernel(kernel_matrix_full_sp,Tmap_cell_id_t1,clonal_cell_id_t1,verbose=verbose)
            final_kernel_ext=generate_final_kernel(kernel_matrix_full_sp,clonal_cell_id_t2,Tmap_cell_id_t2,verbose=verbose)
            
            ### minimum similarity matrix that only involves the multi-time clones
            initial_kernel=generate_initial_kernel(kernel_matrix_full_sp,clonal_cell_id_t1,clonal_cell_id_t1,verbose=verbose)
            final_kernel=generate_final_kernel(kernel_matrix_full_sp,clonal_cell_id_t2,clonal_cell_id_t2,verbose=verbose)
        else:
            initial_kernel_ext=generate_initial_kernel(kernel_matrix_full,Tmap_cell_id_t1,clonal_cell_id_t1,verbose=verbose)
            final_kernel_ext=generate_final_kernel(kernel_matrix_full,clonal_cell_id_t2,Tmap_cell_id_t2,verbose=verbose)
            initial_kernel=generate_initial_kernel(kernel_matrix_full,clonal_cell_id_t1,clonal_cell_id_t1,verbose=verbose)
            final_kernel=generate_final_kernel(kernel_matrix_full,clonal_cell_id_t2,clonal_cell_id_t2,verbose=verbose)


        initial_kernel_array.append(initial_kernel)
        final_kernel_array.append(final_kernel)
        initial_kernel_array_ext.append(initial_kernel_ext)
        final_kernel_array_ext.append(final_kernel_ext)


    #### Compute the core of the transition map that involve multi-time clones, then extend to other cell states
    clonal_coupling_v1=np.ones((len(clonal_cell_id_t1),len(clonal_cell_id_t2)))
    clonal_map_array=[clonal_coupling_v1]



    cell_by_clone_matrix=clone_annot.copy()
    if not ssp.issparse(cell_by_clone_matrix):
        cell_by_clone_matrix=ssp.csr_matrix(cell_by_clone_matrix)

    CoSpar_iter_N=len(SM_array)
    for j in range(CoSpar_iter_N):
        if verbose:
            print("Current iteration:",j)
        clonal_map=clonal_map_array[j]
        if j<len(SM_array):
            if verbose:
                print(f"Use SM={SM_array[j]}")
            used_initial_kernel=initial_kernel_array[j]
            used_final_kernel=final_kernel_array[j]
        else:
            if verbose:
                print(f"Use SM={SM_array[-1]}")
            used_initial_kernel=initial_kernel_array[-1]
            used_final_kernel=final_kernel_array[-1]

        # clonal_coupling, unSM_sc_coupling=refine_clonal_map_by_integrating_clonal_info(clonal_cell_id_t1,clonal_cell_id_t2,
        #        clonal_map,cell_by_clone_matrix,used_initial_kernel,used_final_kernel,noise_threshold,row_normalize=True,normalization_mode=normalization_mode)

        
        clonal_coupling, unSM_sc_coupling=refine_clonal_map_by_integrating_clonal_info_multiTime(multiTime_cell_id_t1,multiTime_cell_id_t2,
            proportion,clonal_map,cell_by_clone_matrix,used_initial_kernel,used_final_kernel,noise_threshold,row_normalize=True,verbose=verbose,normalization_mode=normalization_mode)


        clonal_map_array.append(clonal_coupling)



    ### expand the map to other cell states
    ratio_t1=np.sum(np.in1d(Tmap_cell_id_t1,clonal_cell_id_t1))/len(Tmap_cell_id_t1)
    ratio_t2=np.sum(np.in1d(Tmap_cell_id_t2,clonal_cell_id_t2))/len(Tmap_cell_id_t2)
    if (ratio_t1==1) and (ratio_t2==1): # no need to SM the map
        if verbose:
            print("No need for Final Smooth")
            adata.uns['transition_map']=ssp.csr_matrix(clonal_coupling)
    else:
        if verbose:
            print("Final round of SM")

        if j<len(SM_array):
            used_initial_kernel_ext=initial_kernel_array_ext[j]
            used_final_kernel_ext=final_kernel_array_ext[j]
        else:
            used_initial_kernel_ext=initial_kernel_array_ext[-1]
            used_final_kernel_ext=final_kernel_array_ext[-1]

        unSM_sc_coupling=ssp.csr_matrix(unSM_sc_coupling)
        t=time.time()
        temp=unSM_sc_coupling*used_final_kernel_ext
        if verbose:
            print("Phase I: time elapsed -- ", time.time()-t)
        clonal_map_1=used_initial_kernel_ext.dot(temp)
        if verbose:
            print("Phase II: time elapsed -- ", time.time()-t)


        adata.uns['transition_map']=ssp.csr_matrix(clonal_map_1)
        #adata.uns['transition_map_unExtended']=ssp.csr_matrix(clonal_coupling)


    if verbose:
        print("----Demultiplexed transition map----")

    #pdb.set_trace()
    demultiplexed_map_0=refine_clonal_map_by_integrating_clonal_info_multiTime_No_SM(multiTime_cell_id_t1,multiTime_cell_id_t2,proportion,clonal_coupling,
        cell_by_clone_matrix,intra_clone_threshold=0,noise_threshold=demulti_threshold,normalization_mode=normalization_mode)

    idx_t1=hf.converting_id_from_fullSpace_to_subSpace(clonal_cell_id_t1,Tmap_cell_id_t1)[0]
    idx_t2=hf.converting_id_from_fullSpace_to_subSpace(clonal_cell_id_t2,Tmap_cell_id_t2)[0]
    demultiplexed_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
    demultiplexed_map[idx_t1[:,np.newaxis],idx_t2]=demultiplexed_map_0.A
    adata.uns['demulti_transition_map']=ssp.csr_matrix(demultiplexed_map)



def CoSpar_TwoTimeClones_demultiplexing(adata,intra_clone_threshold=0,demulti_threshold=0.05,normalization_mode=0):
    '''
        Infer transition amplitude of cell states towards clonally-related cell states

        If the transition map is further smoothed to expand the cell states to cover all possible states, due to normalization, the probability at individual cell states now gets diluted. You will need to lower the demulti_threshold. 
    '''
    ########## extract data
    if 'transition_map' not in adata.uns.keys():
        print("Please run ---- CS.tmap.CoSpar_TwoTimeClones ---- first")

    else:

        clone_annot=adata.obsm['cell_by_clone_matrix']

        multiTime_cell_id_t1=[adata.uns['Tmap_cell_id_t1']]
        multiTime_cell_id_t2=[adata.uns['Tmap_cell_id_t2']]
        proportion=adata.uns['proportion']
        #data_path=adata.uns['data_path']
        transition_map=adata.uns['transition_map']

        cell_by_clone_matrix=clone_annot.copy()
        if not ssp.issparse(cell_by_clone_matrix):
            cell_by_clone_matrix=ssp.csr_matrix(cell_by_clone_matrix)

        demultiplexed_map=refine_clonal_map_by_integrating_clonal_info_multiTime_No_SM(multiTime_cell_id_t1,multiTime_cell_id_t2,proportion,transition_map,
            cell_by_clone_matrix,intra_clone_threshold=intra_clone_threshold,noise_threshold=demulti_threshold,normalization_mode=normalization_mode)

        adata.uns['demulti_transition_map']=ssp.csr_matrix(demultiplexed_map)





def Transition_map_from_highly_variable_genes(adata,min_counts=3,min_cells=3,min_vscore_pctl=85,noise_threshold=0.2,neighbor_N=20,normalization_mode=0,
    use_full_kernel=False,SM_array=[15,10,5],verbose=True,trunca_threshold=0.001,use_all_cells=False,compute_new_Smatrix=True):
    '''
        Input:
            adata: assumed to be preprocessed, only has two time points.

            ### selecting highly variable genes
            min_counts: minium number of UMI count per gene
            min_cells: must be shared by at least min_cells
            min_vscore_pctl: must be larger than this percentile, where the genes are ranked, with the most highly variable genes in the 100 percentile.

            
            ### All the rest parameters are related to applying CoSpar to infer the transition map. See CoSpar_TwoTimeClones for details.

        return: 
            adata (since a new adata object is generated in CoSpar, it might be necessary that we return the object)

    '''

    weight=1 # wehight of each gene. 

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    real_clone_annot=adata.obsm['cell_by_clone_matrix']

    time_info=np.array(adata.obs['time_info'])
    selected_time_points=[time_info[cell_id_array_t1][0],time_info[cell_id_array_t2][0]]


    if verbose:
        print("----------------")
        print('Step a: find the commonly shared highly variable genes')
    adata_t1=sc.AnnData(adata.X[cell_id_array_t1]);
    adata_t2=sc.AnnData(adata.X[cell_id_array_t2]);

    ## use marker genes
    gene_list=adata.var_names

    highvar_genes_t1 = gene_list[hf.filter_genes(
        adata_t1.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_vscore_pctl, 
        show_vscore_plot=verbose)]

    highvar_genes_t2 = gene_list[hf.filter_genes(
        adata_t2.X, 
        min_counts=min_counts, 
        min_cells=min_cells, 
        min_vscore_pctl=min_vscore_pctl, 
        show_vscore_plot=verbose)]

    common_gene=list(set(highvar_genes_t1).intersection(highvar_genes_t2))
    if verbose:
        print(f"Highly varable gene number at t1 is {len(highvar_genes_t1)}, Highly varable gene number at t2 is {len(highvar_genes_t2)}")
        print(f"Common gene set is {len(common_gene)}")

        print("----------------")
        print('Step b: convert the shared highly variable genes into clonal info')

    sel_marker_gene_list=common_gene.copy()
    clone_annot_gene=np.zeros((adata.shape[0],len(sel_marker_gene_list)))
    N_t1=len(cell_id_array_t1)
    N_t2=len(cell_id_array_t2)
    cumu_sel_idx_t1=np.zeros(N_t1,dtype=bool)
    cumu_sel_idx_t2=np.zeros(N_t2,dtype=bool)
    cell_fraction_per_gene=1/len(sel_marker_gene_list) # fraction of cells as clonally related by this gene
    for j,gene_id in enumerate(sel_marker_gene_list): 
        temp_t1=adata.obs_vector(gene_id)[cell_id_array_t1]
        temp_t1[cumu_sel_idx_t1]=0 # set selected cell id to have zero expression
        cutoff_t1=int(np.ceil(len(cell_id_array_t1)*cell_fraction_per_gene))
        sel_id_t1=np.argsort(temp_t1)[::-1][:cutoff_t1]
        clone_annot_gene[cell_id_array_t1[sel_id_t1],j]=weight
        cumu_sel_idx_t1[sel_id_t1]=True 
        #print(f"Gene id {gene_id}, cell number at t1 is {sel_id_t1.shape[0]}, fraction at t1: {sel_id_t1.shape[0]/len(cell_id_array_t1)}")

        temp_t2=adata.obs_vector(gene_id)[cell_id_array_t2]
        temp_t2[cumu_sel_idx_t2]=0 # set selected cell id to have zero expression
        cutoff_t2=int(np.ceil(len(cell_id_array_t2)*cell_fraction_per_gene))
        sel_id_t2=np.argsort(temp_t2)[::-1][:cutoff_t2]
        clone_annot_gene[cell_id_array_t2[sel_id_t2],j]=weight
        cumu_sel_idx_t2[sel_id_t2]=True 
        #print(f"Gene id {gene_id}, cell number at t2 is {sel_id_t2.shape[0]}, fraction at t2: {sel_id_t2.shape[0]/len(cell_id_array_t2)}")
        
        if (np.sum(~cumu_sel_idx_t1)==0) or (np.sum(~cumu_sel_idx_t2)==0):
            print(f'No cells left for assignment, total used genes={j}')
            break

    #print(f"Selected cell fraction: t1 -- {np.sum(cumu_sel_idx_t1)/len(cell_id_array_t1)}; t2 -- {np.sum(cumu_sel_idx_t2)/len(cell_id_array_t2)}")


    if verbose:
        print("----------------")
        print("Step c: compute the transition map based on clonal info from highly variable genes")
    
    adata.obsm['cell_by_clone_matrix']=ssp.csr_matrix(clone_annot_gene)
    adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1]
    adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2]
    adata.uns['proportion']=[1]
    data_des_0=adata.uns['data_des'][0]
    data_des_1=data_des_0+'_HighVar0' # to distinguish Similarity matrix for this step and the next step of CoSpar (use _HighVar0, instead of _HighVar1)
    adata.uns['data_des'][0]=[data_des_1]

    CoSpar_TwoTimeClones_private(adata,SM_array=SM_array,neighbor_N=neighbor_N,noise_threshold=noise_threshold,
        normalization_mode=normalization_mode,save_subset=True,use_full_kernel=use_full_kernel,
        verbose=verbose,trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix)

    adata.uns['HighVar_transition_map']=adata.uns['transition_map']
    adata.obsm['cell_by_clone_matrix']=real_clone_annot # This entry has been changed previously. Note correct the clonal matrix
    data_des_1=data_des_0+'_HighVar1' # to record which initialization is used
    adata.uns['data_des']=[data_des_1]


def Compute_custom_OT_transition_map(adata,OT_epsilon=0.02,OT_max_iter=1000,OT_stopThr=1e-09,OT_dis_KNN=5,verbose=True):

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=adata.uns['data_path'][0]


    ############ Compute shorted-path distance
    # use sklearn KNN graph construction method and select the connectivity option, not related to UMAP
    # use the mode 'distance' to obtain the shortest-path *distance*, rather than 'connectivity'
    SPD_file_name=f'{data_path}/{data_des}_ShortestPathDistanceMatrix_t0t1_KNN{OT_dis_KNN}.npy'
    if os.path.exists(SPD_file_name):
        if verbose:
            print("Load pre-computed shortest path distance matrix")
        ShortPath_dis_t0t1=np.load(SPD_file_name)

    else:
        if verbose:
            print("Compute new shortest path distance matrix")
        t=time.time()       
        data_matrix=adata.obsm['X_pca']
        ShortPath_dis=hf.compute_shortest_path_distance_from_raw_matrix(data_matrix,num_neighbors_target=OT_dis_KNN,mode='distance')

        idx0=cell_id_array_t1
        idx1=cell_id_array_t2
        ShortPath_dis_t0t1=ShortPath_dis[idx0[:,np.newaxis],idx1]; ShortPath_dis_t0t1=ShortPath_dis_t0t1/ShortPath_dis_t0t1.max()
        ShortPath_dis_t0=ShortPath_dis[idx0[:,np.newaxis],idx0]; ShortPath_dis_t0=ShortPath_dis_t0/ShortPath_dis_t0.max()
        ShortPath_dis_t1=ShortPath_dis[idx1[:,np.newaxis],idx1]; ShortPath_dis_t1=ShortPath_dis_t1/ShortPath_dis_t1.max()

        np.save(SPD_file_name,ShortPath_dis_t0t1)

        if verbose:
            print(f"Finishing computing shortest-path distance, used time {time.time()-t}")

    # if verbose:
    #     target_cell_ID=2;
    #     distance=ShortPath_dis_t0t1[target_cell_ID,:]
    #     cot_plot.plot_graph_distance(x_emb[cell_id_array_t2],y_emb[cell_id_array_t2],distance,delta=0.05)


    ######## apply optimal transport
    CustomOT_file_name=f'{data_path}/{data_des}_CustomOT_map_epsilon{OT_epsilon}_IterN{OT_max_iter}_stopThre{OT_stopThr}_KNN{OT_dis_KNN}.npy'
    if os.path.exists(CustomOT_file_name):
        if verbose:
            print("Load pre-computed custon OT matrix")
        OT_transition_map=np.load(CustomOT_file_name)

    else:
        if verbose:
            print("Compute new custon OT matrix")

        t=time.time()
        mu1=np.ones(len(cell_id_array_t1));
        nu1=np.ones(len(cell_id_array_t2));
        input_mu=mu1 # initial distribution
        input_nu=nu1 # final distribution
        OT_transition_map=otb.sinkhorn_stabilized(input_mu,input_nu,ShortPath_dis_t0t1,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

        np.save(CustomOT_file_name,OT_transition_map)

        if verbose:
            print(f"Finishing computing optial transport map, used time {time.time()-t}")

    adata.uns['OT_transition_map']=ssp.csr_matrix(OT_transition_map)
    data_des_0=adata.uns['data_des'][0]
    data_des_1=data_des_0+'_OT' # to record which initialization is used
    adata.uns['data_des']=[data_des_1]


def label_early_cells_per_clone( current_clone_id,clonal_cell_id_t2_subspace,available_cell_id_t1_subspace,cell_N_to_extract,cell_id_array_t1,cell_id_array_t2,clone_annot_new,transition_map):
    '''
        Information related to this specific clone:
            current_clone_id
            clonal_cell_id_t2_subspace
            available_cell_id_t1_subspace
            cell_N_to_extract

        General background information:
            cell_id_array_t1
            cell_id_array_t2
            clone_annot_new
            transition_map
    '''

    # add cell states on t2 for this clone
    temp_idx=np.ones(len(cell_id_array_t2),dtype=bool)
    clone_annot_new[cell_id_array_t2,current_clone_id]=temp_idx
    
    # infer the earlier clonal states for each clone
    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end
    sorted_id_array=np.argsort(transition_map[available_cell_id_t1_subspace][:,clonal_cell_id_t2_subspace].sum(1).A.flatten())[::-1]
    #available_cell_id=np.nonzero(available_cell_idx_t1_subspace)[0]

    #pdb.set_trace()
    if len(sorted_id_array)>cell_N_to_extract:
        sel_id_t1=available_cell_id_t1_subspace[sorted_id_array][:cell_N_to_extract]
    else:
        sel_id_t1=available_cell_id_t1_subspace

    # add cell states on t1 for this clone
    clone_annot_new[cell_id_array_t1[sel_id_t1],current_clone_id]=np.ones(len(sel_id_t1),dtype=bool)

    return clone_annot_new,sel_id_t1


def round_number_probabilistically(x):
    y0=int(x) # round the number directly
    x1=x-y0 # between 0 and 1
    if np.random.rand()<x1:
        y=y0+1
    else:
        y=y0
    return int(y)


def CoSpar_OneTimeClone_JointOptimization_Private(adata,initialized_map,Clone_update_iter_N=1,normalization_mode=0,
    noise_threshold=0.2,CoSpar_KNN=20,use_full_kernel=False,SM_array=[15,10,5],verbose=True,trunca_threshold=0.001,use_all_cells=False,compute_new_Smatrix=True):
    '''
        the adata structure must be prepared by preprocessing steps

    '''
    print("Consider possibility of clonal overlap")

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=adata.uns['data_path'][0]
    clone_annot=adata.obsm['cell_by_clone_matrix']
    if not ssp.issparse(clone_annot): clone_annot=ssp.csr_matrix(clone_annot) 

    time_info=np.array(adata.obs['time_info'])
    time_index_t1=time_info==(time_info[cell_id_array_t1[0]])
    time_index_t2=time_info==(time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp=ssp.csr_matrix(initialized_map)
    else:
        map_temp=initialized_map

    clone_annot_temp=clone_annot.copy()
    clone_N1=clone_annot.shape[1]


    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    # we sort clones according to their sizes. The order of cells are not affected. So, it should not affect downstream analysis
    # small clones tend to be the ones that are barcoded/mutated later, while large clones tend to be early mutations...
    clone_size_t2_temp=clone_annot_temp.sum(0).A.flatten()
    sort_clone_id=np.argsort(clone_size_t2_temp)
    clone_size_t2=clone_size_t2_temp[sort_clone_id]
    clone_annot_sort=clone_annot_temp[:,sort_clone_id]

    for x0 in range(Clone_update_iter_N):

        ########## begin: update clones
        remaining_ids_t1=list(np.arange(len(cell_id_array_t1),dtype=int))

        clone_annot_new=np.zeros(clone_annot_sort.shape,dtype=bool)
        for j in range(clone_N1):
            if (j%50==0) and verbose:
                #pdb.set_trace()
                print(f"Inferring early clonal states: current clone id {j}")


            # identify overlapped clones at t2
            overlap_cell_N_per_clone=(clone_annot_sort[cell_id_array_t2,j].T*clone_annot_sort[cell_id_array_t2,:j]).A.flatten()
            overlap_id=np.nonzero(overlap_cell_N_per_clone>0)[0]
            if len(overlap_id)>0:
                for current_clone_id in overlap_id:
                    available_cell_id_t1_subspace=np.nonzero(clone_annot_new[cell_id_array_t1,current_clone_id].flatten()>0)[0]
                    #available_cell_idx_t1_subspace=cell_id_array_t1[temp_idx]
                    overlapped_idx_t2_subspace=(clone_annot_sort[cell_id_array_t2,current_clone_id].A.flatten()>0) & (clone_annot_sort[cell_id_array_t2,j].A.flatten()>0)
                    overlapped_id_t2_subspace=np.nonzero(overlapped_idx_t2_subspace)[0]
                    cell_N_to_extract_0=len(overlapped_cell_id_t2_subspace)*len(cell_id_array_t1)/len(cell_id_array_t2)
                    cell_N_to_extract=round_number_probabilistically(cell_N_to_extract_0)

                    clone_annot_new,sel_cell_id_t1=label_early_cells_per_clone(current_clone_id,overlapped_id_t2_subspace,available_cell_id_t1_subspace,cell_N_to_extract,cell_id_array_t1,cell_id_array_t2,clone_annot_new,map_temp)
            
            else: # if there is no overlap 
                current_clone_id=j
                overlapped_cell_id_t2_subspace=np.nonzero(clone_annot_sort[cell_id_array_t2][:,j].A.flatten()>0)[0]
                #overlapped_cell_id_t2=cell_id_array_t2[overlapped_idx_t2]
                cell_N_to_extract_0=len(overlapped_cell_id_t2_subspace)*len(cell_id_array_t1)/len(cell_id_array_t2)
                cell_N_to_extract=round_number_probabilistically(cell_N_to_extract_0)
                available_cell_id_t1_subspace=np.array(remaining_ids_t1)
                clone_annot_new,sel_cell_id_t1=label_early_cells_per_clone(current_clone_id,overlapped_cell_id_t2_subspace,available_cell_id_t1_subspace,cell_N_to_extract,cell_id_array_t1,cell_id_array_t2,clone_annot_new,map_temp)

                # remove selected id from the list
                for kk in sel_cell_id_t1:
                    remaining_ids_t1.remove(kk)

            
            if len(remaining_ids_t1)==0: 
                print('Warning: early break (OK if initial cell number less than clone number)')
                break
        ########### end: update clones

        cell_id_array_t1_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t1))[0]
        cell_id_array_t2_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t2))[0]


        #clone_annot_new=np.zeros(clone_annot_sort.shape,dtype=bool)
        #clone_annot_new[:,sort_clone_id]=clone_annot_new

        clone_annot_new=ssp.csr_matrix(clone_annot_new)
        adata.obsm['cell_by_clone_matrix']=clone_annot_new
        adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1_new] # For CoSpar, clonally-related states
        adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2_new]
        adata.uns['clonal_cell_id_t1']=cell_id_array_t1_new # for prepare the similarity matrix with same cell states
        adata.uns['clonal_cell_id_t2']=cell_id_array_t2_new
        adata.uns['proportion']=[1]

        #pdb.set_trace()
        CoSpar_TwoTimeClones_private(adata,SM_array=SM_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,
            normalization_mode=normalization_mode,save_subset=True,use_full_kernel=use_full_kernel,verbose=verbose,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix)

        ## update, for the next iteration
        map_temp=adata.uns['transition_map']
        clone_annot_sort=clone_annot_new.copy()





def CoSpar_OneTimeClone_JointOptimization_Private_v0(adata,initialized_map,Clone_update_iter_N=1,normalization_mode=0,
    noise_threshold=0.2,CoSpar_KNN=20,use_full_kernel=False,SM_array=[15,10,5],verbose=True,trunca_threshold=0.001,use_all_cells=False,compute_new_Smatrix=True):
    '''
        the adata structure must be prepared by preprocessing steps

    '''

    cell_id_array_t1=adata.uns['Tmap_cell_id_t1']
    cell_id_array_t2=adata.uns['Tmap_cell_id_t2']
    data_des=adata.uns['data_des'][0]
    data_path=adata.uns['data_path'][0]
    clone_annot=adata.obsm['cell_by_clone_matrix']
    if not ssp.issparse(clone_annot): clone_annot=ssp.csr_matrix(clone_annot) 

    time_info=np.array(adata.obs['time_info'])
    time_index_t1=time_info==(time_info[cell_id_array_t1[0]])
    time_index_t2=time_info==(time_info[cell_id_array_t2[0]])

    if not ssp.issparse(initialized_map):
        map_temp=ssp.csr_matrix(initialized_map)
    else:
        map_temp=initialized_map

    cell_id_array_t1_temp=cell_id_array_t1.copy()
    cell_id_array_t2_temp=cell_id_array_t2.copy()
    clone_annot_temp=clone_annot.copy()
    clone_N1=clone_annot.shape[1]


    ### select the early states using the grouped distribution of a clone
    ### clones are not overlapping, and all early states should be attached to clones at the end

    ave_clone_size_t1=int(np.ceil(len(cell_id_array_t1)/clone_N1));

    for x0 in range(Clone_update_iter_N):

        ########## begin: update clones
        remaining_ids_t1=list(np.arange(len(cell_id_array_t1),dtype=int))

        clone_annot_new=np.zeros(clone_annot_temp.shape,dtype=bool)
        for j in range(clone_N1):
            if (j%50==0) and verbose:
                #pdb.set_trace()
                print(f"Inferring early clonal states: current clone id {j}")

            # add back the known clonal states at t2
            #pdb.set_trace()
            temp_t2_idx=clone_annot_temp[cell_id_array_t2_temp][:,j].A.flatten()>0
            clone_annot_new[cell_id_array_t2_temp,j]=temp_t2_idx
            
            # infer the earlier clonal states for each clone
            ### select the early states using the grouped distribution of a clone
            ### clones are not overlapping, and all early states should be attached to clones at the end
            sorted_id_array=np.argsort(map_temp[remaining_ids_t1][:,temp_t2_idx].sum(1).A.flatten())[::-1]
            sel_id_t1=sorted_id_array[:ave_clone_size_t1]
            temp_t1_idx=np.zeros(len(cell_id_array_t1_temp),dtype=bool)
            temp_t1_idx[np.array(remaining_ids_t1)[sel_id_t1]]=True
            clone_annot_new[cell_id_array_t1_temp,j]=temp_t1_idx
            for kk in np.array(remaining_ids_t1)[sel_id_t1]:
                remaining_ids_t1.remove(kk)
            
            if len(remaining_ids_t1)==0: 
                print('Warning: early break (OK if initial cell number less than clone number)')
                break
        ########### end: update clones

        cell_id_array_t1_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t1))[0]
        cell_id_array_t2_new=np.nonzero((clone_annot_new.sum(1)>0) & (time_index_t2))[0]


        clone_annot_new=ssp.csr_matrix(clone_annot_new)
        adata.obsm['cell_by_clone_matrix']=clone_annot_new
        adata.uns['multiTime_cell_id_t1']=[cell_id_array_t1_new] # For CoSpar, clonally-related states
        adata.uns['multiTime_cell_id_t2']=[cell_id_array_t2_new]
        adata.uns['clonal_cell_id_t1']=cell_id_array_t1_new # for prepare the similarity matrix with same cell states
        adata.uns['clonal_cell_id_t2']=cell_id_array_t2_new
        adata.uns['proportion']=[1]

        CoSpar_TwoTimeClones_private(adata,SM_array=SM_array,neighbor_N=CoSpar_KNN,noise_threshold=noise_threshold,
            normalization_mode=normalization_mode,save_subset=True,use_full_kernel=use_full_kernel,verbose=verbose,
            trunca_threshold=trunca_threshold,compute_new_Smatrix=compute_new_Smatrix)

        ## update, for the next iteration
        map_temp=adata.uns['transition_map']
        clone_annot_temp=clone_annot_new.copy()

def CoSpar_OneTimeClones(adata_orig,initial_time_points=['1'],clonal_time_point='2',initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_max_iter=1000,OT_stopThr=1e-09,HighVar_gene_pctl=85,
    Clone_update_iter_N=1,normalization_mode=0,noise_threshold=0.2,CoSpar_KNN=20,use_full_kernel=False,SM_array=[15,10,5],
    verbose=True,trunca_threshold=0.001,use_all_cells=False,compute_new_Smatrix=False):


    '''
        Purpose:
            Infer transition map from scRNAseq data where cells at one time point are clonally labeled. After initializing the map by either OT method or HighVar method, We jointly infer the likely clonal ancestors and the transition map between cell states in these two time points. 

        Input:
            selected_two_time_points: the time point to be included for analysis. Only two. The second time point should be the states with clonal information.

            The adata object should encode the following information
                
                cell_by_clone_matrix: clone-by-cell matrix at adata.obsm['cell_by_clone_matrix']
                
                Day: a time information at adata_orig.obs['time_info']
                
                X_pca: for knn graph construction, at adata_orig.obsm['X_pca']
                
                state_annotation: state annotation (e.g., in terms of clusters), at adata_orig.obs['state_annotation'], as categorical variables
                
                X_umap: an embedding, does not have to be from UMAP. But the embedding should be saved to this place by adata_rig.obsm['X_umap']=XXX
                
                data_des: a description of the data, for saving downstream results, at adata_orig.uns['data_des'][0], i.e., it should be a list, or adata_orig.uns['data_des']=[XXX]. This is a compromise because once we save adata object, it becomes a np.array. 
                
                data_path: a path string for saving data, accessed as adata_orig.uns['data_path'][0], i.e., it should be a list
                
                figure_path: a path string for saving figures, adata_orig.uns['figure_path'][0]


            Initialization related:
                initialize_method: 'OT' for the optimal transport method, or 'HighVar' that relies on converting highly variable genes into artificial clones.
                'OT' related:
                    
                    OT_epsilon: regulation parameter, related to the entropy or smoothness of the map. A larger epsilon means better smoothness, but less informative
                    
                    OT_dis_KNN: number of nearest neighbor for the KNN graph construction 
                    
                    OT_max_iter: maximum number of iteration for running optimal transport algorithm
                    
                    OT_stopThr: the convergence threshold for running optimal transport. A small value means the ieration has a small difference from previous run, thus convergent.
                'HighVar' related:
                    
                    HighVar_gene_pctl: percentile of the most highly variable genes to be used for constructing the pseudo-clone

            CoSpar related:
                normalization_mode: 0, the default, treat T as the transition probability matrix, row normalized;
                                    1, add modulation to the transition probability according to proliferation.
                
                noise_threshold: we set to zero entries of the computed transition map below this threshold, to promote sparsity of the map
                
                SM_array: the length of this list determines the total number of iteration in CoSpar, and the corresponding entry determines the rounds of graph diffusion in generating the corresponding similarity matrix used for that iteration
                
                CoSpar_KNN: the number of neighbors for KNN graph used for computing the similarity matrix
                
                use_full_kernel: True, we use all available cell states to generate the similarity matrix. We then sub-sample cell states that are relevant for downstream analysis from this full matrix. This tends to be more accurate.
                                 False, we only use cell states for the selected time points to generate the similarity matrix. This is faster, yet could be less accurate.

                use_all_cells: True, infer the transition map among all cell states in the selected time points (t1+t2)-(t1+t2) matrix.  This can be very slow for large datasets. 
                               False, infer the transition map for cell states that are clonally labeled (except that you only select two time points.)

                
                verbose: True, print information of the analysis; False, do not print.
                
                trunca_threshold: this value is only for reducing the computed matrix size for saving. We set entries to zero in the similarity matrix that are smaller than this threshold. This threshld should be small, but not too small. 
                
                save_subset: True, save only similarity matrix corresponds to 5*n rounds of graph diffusion, where n is an integer; 
                              False, save all similarity matrix with any rounds generated in the process


        return:
            adata object that includes the transition map at adata.uns['transition_map']. 
            Depending the initialization method, the initialized map is stored at either adata.uns['HighVar_transition_map'] or adata.uns['OT_transition_map']
            This adata is different from the original one: adata_orig.

        Summary:
            Parameters relevant for cell state selection:  initial_time_points, clonal_time_point, use_full_kernel, use_all_cells, 

            Choose the initialization method, and set the corresponding parameters. 
            
            1) 'HighVar':  is faster and robust to batch effect
                  related parameters: HighVar_gene_pctl
            
            2) 'OT': tend to be more accurate, but very slow and not reliable under batch effect
                  related parameters: OT_epsilon, OT_dis_KNN, ,OT_max_iter

            Parameters relevant for CoSpar itself:   SM_array,normalization_mode,CoSpar_KNN,noise_threshold, Clone_update_iter_N
    '''


    for xx in initial_time_points:
        if xx not in list(set(adata_orig.obs['time_info'])):
            print(f"Error! the 'initial_time_points' are not valid. Please select from {list(set(adata_orig.obs['time_info']))}")
            return adata_orig

    with_clonal_info=(clonal_time_point in adata_orig.uns['clonal_time_points'])
    if not with_clonal_info:
        print(f"'clonal_time_point' do not contain clonal information. Please set clonal_time_point to be one of {adata_orig.uns['clonal_time_points']}")
        #print("Consider run ----cs.tmap.CoSpar_NoClonalInfo------")
        print("Keep running but without clonal information")
        #return adata_orig

    sp_idx=np.zeros(adata_orig.shape[0],dtype=bool)
    time_info_orig=np.array(adata_orig.obs['time_info'])
    all_time_points=initial_time_points+[clonal_time_point]
    label='t'
    for xx in all_time_points:
        id_array=np.nonzero(time_info_orig==xx)[0]
        sp_idx[id_array]=True
        label=label+'*'+str(xx)

    adata=sc.AnnData(adata_orig.X[sp_idx]);
    adata.var_names=adata_orig.var_names
    adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
    adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
    adata.obs['state_annotation']=pd.Categorical(adata_orig.obs['state_annotation'][sp_idx])
    adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
    adata.uns['data_path']=adata_orig.uns['data_path']
    data_des_0=adata_orig.uns['data_des'][0]
    data_des=data_des_0+f'_OneTimeClone_{label}'
    adata.uns['data_des']=[data_des]
    adata.uns['figure_path']=adata_orig.uns['figure_path']


    clone_annot_orig=adata_orig.obsm['cell_by_clone_matrix']        
    clone_annot=clone_annot_orig[sp_idx]
    adata.obsm['cell_by_clone_matrix']=clone_annot

    time_info=np.array(adata.obs['time_info'])
    time_index_t2=time_info==clonal_time_point
    time_index_t1=~time_index_t2

    #### used for kernel matrix generation
    Tmap_cell_id_t1=np.nonzero(time_index_t1)[0]
    Tmap_cell_id_t2=np.nonzero(time_index_t2)[0]
    adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
    adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
    adata.uns['clonal_cell_id_t1']=Tmap_cell_id_t1
    adata.uns['clonal_cell_id_t2']=Tmap_cell_id_t2
    adata.uns['sp_idx']=sp_idx
    data_path=adata_orig.uns['data_path'][0]

    transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))
    ini_transition_map=np.zeros((len(Tmap_cell_id_t1),len(Tmap_cell_id_t2)))


    for yy in initial_time_points:
        if verbose:
            print("-------------------------------New Start--------------------------------------------------")
            print(f"Current time point: {yy}")

        adata_temp=CoSpar_OneTimeClones_TwoTimePoints(adata_orig,selected_two_time_points=[yy,clonal_time_point],initialize_method=initialize_method,OT_epsilon=OT_epsilon,OT_dis_KNN=OT_dis_KNN,
            OT_max_iter=OT_max_iter,OT_stopThr=OT_stopThr,HighVar_gene_pctl=HighVar_gene_pctl,Clone_update_iter_N=Clone_update_iter_N,normalization_mode=normalization_mode,
            noise_threshold=noise_threshold,CoSpar_KNN=CoSpar_KNN,use_full_kernel=use_full_kernel,SM_array=SM_array,
            verbose=verbose,trunca_threshold=trunca_threshold,use_all_cells=use_all_cells,compute_new_Smatrix=compute_new_Smatrix)

        temp_id_t1=np.nonzero(time_info==yy)[0]
        sp_id_t1=hf.converting_id_from_fullSpace_to_subSpace(temp_id_t1,Tmap_cell_id_t1)[0]
        
        if with_clonal_info:
            transition_map_temp=adata_temp.uns['transition_map'].A
            transition_map[sp_id_t1,:]=transition_map_temp

        if initialize_method=='OT':
            transition_map_ini_temp=adata_temp.uns['OT_transition_map']
        else:
            transition_map_ini_temp=adata_temp.uns['HighVar_transition_map']

        ini_transition_map[sp_id_t1,:]=transition_map_ini_temp.A

    if with_clonal_info:
        adata.uns['transition_map']=ssp.csr_matrix(transition_map)
    
    if initialize_method=='OT':
        adata.uns['OT_transition_map']=ssp.csr_matrix(ini_transition_map)
    else:
        adata.uns['HighVar_transition_map']=ssp.csr_matrix(ini_transition_map)


    return adata



def CoSpar_OneTimeClones_TwoTimePoints(adata_orig,selected_two_time_points=['1','2'],initialize_method='OT',OT_epsilon=0.02,OT_dis_KNN=5,OT_max_iter=1000,OT_stopThr=1e-09,HighVar_gene_pctl=80,
    Clone_update_iter_N=1,normalization_mode=0,noise_threshold=0.2,CoSpar_KNN=20,use_full_kernel=False,SM_array=[15,10,5],
    verbose=True,trunca_threshold=0.001,use_all_cells=False,compute_new_Smatrix=True):

    '''
        Purpose:
            Infer transition map from scRNAseq data where cells at one time point are clonally labeled. After initializing the map by either OT method or HighVar method, We jointly infer the likely clonal ancestors and the transition map between cell states in these two time points. 

        Input:
            selected_two_time_points: the time point to be included for analysis. Only two. The second time point should be the states with clonal information.

            The adata object should encode the following information
                cell_by_clone_matrix: clone-by-cell matrix at adata.obsm['cell_by_clone_matrix']
                Day: a time information at adata_orig.obs['time_info']
                X_pca: for knn graph construction, at adata_orig.obsm['X_pca']
                state_annotation: state annotation (e.g., in terms of clusters), at adata_orig.obs['state_annotation'], as categorical variables
                X_umap: an embedding, does not have to be from UMAP. But the embedding should be saved to this place by adata_rig.obsm['X_umap']=XXX
                data_des: a description of the data, for saving downstream results, at adata_orig.uns['data_des'][0], i.e., it should be a list, or adata_orig.uns['data_des']=[XXX]. This is a compromise because once we save adata object, it becomes a np.array. 
                data_path: a path string for saving data, accessed as adata_orig.uns['data_path'][0], i.e., it should be a list
                figure_path: a path string for saving figures, adata_orig.uns['figure_path'][0]


            Initialization related:
                initialize_method: 'OT' for the optimal transport method, or 'HighVar' that relies on converting highly variable genes into artificial clones.
                OT_epsilon: regulation parameter, related to the entropy or smoothness of the map. A larger epsilon means better smoothness, but less informative
                OT_dis_KNN: number of nearest neighbor for the KNN graph construction 
                OT_max_iter: maximum number of iteration for running optimal transport algorithm
                OT_stopThr: the convergence threshold for running optimal transport. A small value means the ieration has a small difference from previous run, thus convergent.
                HighVar_gene_pctl: percentile of the most highly variable genes to be used for constructing the pseudo-clone, relevant for the 'HighVar' mode

            CoSpar related:
                normalization_mode: 0,1 
                                    0, the default, treat T as the transition probability matrix, row normalized;  
                                    1, add modulation to the transition probability according to proliferation)
                

                SM_array: the length of this list determines the total number of iteration in CoSpar, and the corresponding entry determines the rounds of graph diffusion in generating the corresponding similarity matrix used for that iteration
                neighbor_N: the number of neighbors for KNN graph used for computing the similarity matrix
                use_full_kernel: True, we use all available cell states to generate the similarity matrix. We then sub-sample cell states that are relevant for downstream analysis from this full matrix. This tends to be more accurate.
                                 False, we only use cell states for the selected time points to generate the similarity matrix. This is faster, yet could be less accurate.

                use_all_cells: True, infer the transition map among all cell states in the selected time points (t1+t2)-(t1+t2) matrix.  This can be very slow for large datasets. 
                               False, infer the transition map for cell states that are clonally labeled (except that you only select two time points.)

                verbose: True, print information of the analysis; False, do not print.
                trunca_threshold: this value is only for reducing the computed matrix size for saving. We set entries to zero in the similarity matrix that are smaller than this threshold. This threshld should be small, but not too small. 
                save_subset: True, save only similarity matrix corresponds to 5*n rounds of graph diffusion, where n is an integer; 
                              False, save all similarity matrix with any rounds generated in the process


        return:
            adata object that includes the transition map at adata.uns['transition_map']. 
            Depending the initialization method, the initialized map is stored at either adata.uns['HighVar_transition_map'] or adata.uns['OT_transition_map']
            This adata is different from the original one: adata_orig.

    '''

    time_info_orig=np.array(adata_orig.obs['time_info'])
    sort_time_point=np.sort(list(set(time_info_orig)))
    N_valid_time=np.sum(np.in1d(sort_time_point,selected_two_time_points))
    if (N_valid_time!=2): 
        print(f"Error! Must select only two time points among the list {sort_time_point}")
        #The second time point in this list (not necessarily later time point) is assumed to have clonal data.")
    else:
        ####################################
        if verbose:
            print("-----------Pre-processing and sub-sampling cells------------")
        # select cells from the two time points, and sub-sampling, create the new adata object with these cell states
        sp_idx=(time_info_orig==selected_two_time_points[0]) | (time_info_orig==selected_two_time_points[1])
  
        adata=sc.AnnData(adata_orig.X[sp_idx]);
        adata.var_names=adata_orig.var_names
        adata.obsm['X_pca']=adata_orig.obsm['X_pca'][sp_idx]
        adata.obsm['X_umap']=adata_orig.obsm['X_umap'][sp_idx]
        adata.obs['state_annotation']=pd.Categorical(adata_orig.obs['state_annotation'][sp_idx])
        adata.obs['time_info']=pd.Categorical(adata_orig.obs['time_info'][sp_idx])
        adata.uns['data_path']=adata_orig.uns['data_path']
        data_des_0=adata_orig.uns['data_des'][0]
        data_des=data_des_0+f'_OneTimeClone_t*{selected_two_time_points[0]}*{selected_two_time_points[1]}'
        adata.uns['data_des']=[data_des]
        adata.uns['figure_path']=adata_orig.uns['figure_path']


        clone_annot_orig=adata_orig.obsm['cell_by_clone_matrix']        
        barcode_id=np.nonzero(clone_annot_orig[sp_idx].A.sum(0).flatten()>0)[0]
        clone_annot=clone_annot_orig[sp_idx][:,barcode_id]
        adata.obsm['cell_by_clone_matrix']=clone_annot

        time_info=np.array(adata.obs['time_info'])
        time_index_t1=time_info==selected_two_time_points[0]
        time_index_t2=time_info==selected_two_time_points[1]

        #### used for kernel matrix generation
        Tmap_cell_id_t1=np.nonzero(time_index_t1)[0]
        Tmap_cell_id_t2=np.nonzero(time_index_t2)[0]
        adata.uns['Tmap_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['Tmap_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['clonal_cell_id_t1']=Tmap_cell_id_t1
        adata.uns['clonal_cell_id_t2']=Tmap_cell_id_t2
        adata.uns['sp_idx']=sp_idx
        data_path=adata_orig.uns['data_path'][0]


        cell_id_array_t1=Tmap_cell_id_t1
        cell_id_array_t2=Tmap_cell_id_t2

        ###############################
        # prepare the kernel matrix with all state info, all subsequent kernel will be down-sampled from this one.
        if use_full_kernel: 

            temp_str='0'+str(trunca_threshold)[2:]
            round_of_smooth=np.max(SM_array)

            kernel_file_name=f'{data_path}/Similarity_matrix_with_all_cell_states_kNN{CoSpar_KNN}_Truncate{temp_str}_v0_fullkernel{use_full_kernel}'
            if not (os.path.exists(kernel_file_name+f'_SM{round_of_smooth}.npz') and (not compute_new_Smatrix)):
                kernel_matrix_full=generate_full_kernel_matrix_single_v0(adata_orig,kernel_file_name,round_of_smooth=round_of_smooth,
                            neighbor_N=CoSpar_KNN,truncation_threshold=trunca_threshold,save_subset=True,verbose=verbose,compute_new_Smatrix=compute_new_Smatrix)

        

        if initialize_method=='OT':
            if verbose:
                print("----------------")
                print("Step 1: Use OT method for initialization")

            Compute_custom_OT_transition_map(adata,OT_epsilon=OT_epsilon,OT_max_iter=OT_max_iter,OT_stopThr=OT_stopThr,OT_dis_KNN=OT_dis_KNN,verbose=verbose)
            OT_transition_map=adata.uns['OT_transition_map']
            initialized_map=OT_transition_map

            
        else:
            if verbose:
                print("----------------")
                print("Step 1: Use highly variable genes to construct pseudo-clones, and apply CoSpar to generate initialized map!")

            t=time.time()
            Transition_map_from_highly_variable_genes(adata,min_counts=3,min_cells=3,min_vscore_pctl=HighVar_gene_pctl,noise_threshold=noise_threshold,neighbor_N=CoSpar_KNN,
                normalization_mode=normalization_mode,use_full_kernel=use_full_kernel,SM_array=SM_array,verbose=verbose,trunca_threshold=trunca_threshold,use_all_cells=use_all_cells,
                compute_new_Smatrix=compute_new_Smatrix)

            HighVar_transition_map=adata.uns['HighVar_transition_map']
            initialized_map=HighVar_transition_map

            if verbose:
                print(f"Finishing computing transport map from highly variable genes, used time {time.time()-t}")


        ########### Jointly optimize the transition map and the initial clonal states
        if selected_two_time_points[1] in adata_orig.uns['clonal_time_points']:
            if verbose:
                print("----------------")
                print("Step 2: Jointly optimize the transition map and the initial clonal states!")

            t=time.time()
            CoSpar_OneTimeClone_JointOptimization_Private(adata,initialized_map,Clone_update_iter_N=Clone_update_iter_N,normalization_mode=normalization_mode,noise_threshold=noise_threshold,
                CoSpar_KNN=CoSpar_KNN,use_full_kernel=use_full_kernel,SM_array=SM_array,verbose=verbose,trunca_threshold=trunca_threshold,use_all_cells=use_all_cells,
                compute_new_Smatrix=compute_new_Smatrix)

            if verbose:
                print(f"Finishing computing transport map from CoSpar using inferred clonal data, used time {time.time()-t}")
        else:
            print("No clonal information available. Skip the joint optimization of clone and scRNAseq data")


        return adata

def compute_naive_map(adata,include_undiff=True):
    '''
        Assume uniform transition within the same clone
    '''

    cell_id_t2=adata.uns['Tmap_cell_id_t2']
    cell_id_t1=adata.uns['Tmap_cell_id_t1']
    clone_annot=adata.obsm['cell_by_clone_matrix']
    state_annote=adata.obs['state_annotation']


    weinreb_map=clone_annot[cell_id_t1]*clone_annot[cell_id_t2].T
    weinreb_map=weinreb_map.astype(int)
    adata.uns['naive_transition_map']=ssp.csr_matrix(weinreb_map)

def compute_weinreb_map(adata,include_undiff=True):
    '''
        This is not supposed to be used outsides
    '''

    print("This method works when there are only time points, and only for LARRY dataset")
    cell_id_t2=adata.uns['Tmap_cell_id_t2']
    cell_id_t1=adata.uns['Tmap_cell_id_t1']
    clone_annot=adata.obsm['cell_by_clone_matrix']
    state_annote=adata.obs['state_annotation']

    if include_undiff:
        fate_array=['Ccr7_DC',  'Mast', 'Meg', 'pDC', 'Eos', 'Lymphoid', 'Erythroid', 'Baso', 'Neutrophil', 'Monocyte','undiff']
    else:
        fate_array=['Ccr7_DC',  'Mast', 'Meg', 'pDC', 'Eos', 'Lymphoid', 'Erythroid', 'Baso', 'Neutrophil', 'Monocyte']
    potential_vector_clone, fate_entropy_clone=hf.compute_state_potential(clone_annot[cell_id_t2].T,state_annote[cell_id_t2],fate_array,fate_count=True)


    sel_unipotent_clone_id=np.array(list(set(np.nonzero(fate_entropy_clone==1)[0])))
    #sel_unipotent_clone_id=core.converting_id_from_fullSpace_to_subSpace(Clone_id_with_single_fate,barcode_id)[0]
    #sel_unipotent_clone_id=np.array(list(Clone_id_with_single_fate_on_day6.intersection(barcode_id)))
    clone_annot_unipotent=clone_annot[:,sel_unipotent_clone_id]
    weinreb_map=clone_annot_unipotent[cell_id_t1]*clone_annot_unipotent[cell_id_t2].T
    weinreb_map=weinreb_map.astype(int)
    print(f"Used clone fraction {len(sel_unipotent_clone_id)/clone_annot.shape[1]}")
    adata.uns['weinreb_transition_map']=ssp.csr_matrix(weinreb_map)

def save_map(adata):
    data_des=adata.uns['data_des'][0]
    data_path=adata.uns['data_path'][0]
    adata.uns['state_trajectory']={} # need to set to empty, otherwise, it does not work
    adata.uns['fate_trajectory']={} # need to set to empty, otherwise, it does not work
    adata.uns['multiTime_cell_id_t1']={} 
    adata.uns['multiTime_cell_id_t2']={} 
    adata.uns['fate_map']={}
    adata.uns['fate_map']={}
    file_name=f'{data_path}/{data_des}_adata_with_transition_map.h5ad'
    adata.write_h5ad(file_name, compression='gzip')
    print(f"Saved file: data_des='{data_des}'")

    