import numpy as np
import scipy
import scipy.stats
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
#import time
#import os
#import json
#from datetime import datetime
#import matplotlib.pyplot as plt
import ot.bregman as otb
import scipy.sparse as ssp
import scanpy as sc
import pandas as pd
import statsmodels.sandbox.stats.multicomp
#import scipy.stats

def get_dge_SW(ad, mask1, mask2, min_frac_expr=0.05, pseudocount=1):

    
    gene_mask = ((ad.X[mask1,:]>0).sum(0).A.squeeze()/mask1.sum() > min_frac_expr) | ((ad.X[mask2,:]>0).sum(0).A.squeeze()/mask2.sum() > min_frac_expr)
    #print(gene_mask.sum())
    E1 = ad.X[mask1,:][:,gene_mask].toarray()
    E2 = ad.X[mask2,:][:,gene_mask].toarray()
    
    m1 = E1.mean(0) + pseudocount
    m2 = E2.mean(0) + pseudocount
    r = np.log2(m1 / m2)
    
    pv = np.zeros(gene_mask.sum())
    for ii,iG in enumerate(np.nonzero(gene_mask)[0]):
        pv[ii] = scipy.stats.ranksums(E1[:,ii], E2[:,ii])[1]
    pv = statsmodels.sandbox.stats.multicomp.multipletests(pv, alpha=0.05, method='fdr_bh',)[1]
    sort_idx=np.argsort(pv)
    
    df = pd.DataFrame({
        'gene': ad.var_names.values.astype(str)[gene_mask][sort_idx],
        'pv': pv[sort_idx],
        'mean_1': m1[sort_idx] - pseudocount, 
        'mean_2': m2[sort_idx] - pseudocount, 
        'ratio': r[sort_idx]
    })
    
    return df


########## USEFUL SPARSE FUNCTIONS

def sparse_var(E, axis=0):
    ''' calculate variance across the specified axis of a sparse matrix'''

    mean_gene = E.mean(axis=axis).A.squeeze()
    tmp = E.copy()
    tmp.data **= 2
    return tmp.mean(axis=axis).A.squeeze() - mean_gene ** 2

def mean_center(E, column_means=None):
    ''' mean-center columns of a sparse matrix '''

    if column_means is None:
        column_means = E.mean(axis=0)
    return E - column_means

def normalize_variance(E, column_stdevs=None):
    ''' variance-normalize columns of a sparse matrix '''

    if column_stdevs is None:
        column_stdevs = np.sqrt(sparse_var(E, axis=0))
    return sparse_rowwise_multiply(E.T, 1 / column_stdevs).T

def sparse_zscore(E, gene_mean=None, gene_stdev=None):
    ''' z-score normalize each column of a sparse matrix '''
    if gene_mean is None:
        gene_mean = E.mean(0)
    if gene_stdev is None:
        gene_stdev = np.sqrt(sparse_var(E))
    return sparse_rowwise_multiply((E - gene_mean).T, 1/gene_stdev).T

def sparse_rowwise_multiply(E, a):
    ''' multiply each row of sparse matrix by a scalar '''
    nrow = E.shape[0]
    if nrow!=a.shape[0]:
        print("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((nrow, nrow))
        w.setdiag(a)
        return w * E

def sparse_column_multiply(E, a):
    ''' multiply each row of sparse matrix by a scalar '''
    ncol = E.shape[1]
    if ncol!=a.shape[0]:
        print("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((ncol, ncol))
        w.setdiag(a)
        return (ssp.csr_matrix(E)*w)




def matrix_row_or_column_thresholding(input_matrix,threshold=0.1,row_threshold=True,verbose=0):
    # threshold: threshold magnitude with respect to the maximum value in the corresponding direction;
    # row_threshold: direction of thresholding,  0 for col, and 1 for row; (default is row threshold)
    # if we want to infer the fate probability right, then use row normalization; 
    # column threshold is used only when linking individual cells with its progenies 

    # the input matrix is supposed to be a numpy array
    if ssp.issparse(input_matrix): input_matrix=input_matrix.A

    output_matrix=input_matrix.copy()
    max_vector=np.max(input_matrix,int(row_threshold))
    for j in range(len(max_vector)):
        if verbose:
            if j%2000==0: print(j)
        if row_threshold:
            idx=input_matrix[j,:]<threshold*max_vector[j]
            output_matrix[j,idx]=0
        else:
            idx=input_matrix[:,j]<threshold*max_vector[j]
            output_matrix[idx,j]=0

    return output_matrix


def get_pca(E, base_ix=[], numpc=50, keep_sparse=False, normalize=True, random_state=0):
    '''
    Note that there is Zscore transformation before PCA. this may not be what you want
    Run PCA on the counts matrix E, gene-level normalizing if desired
    Return PCA coordinates
    '''
    # If keep_sparse is True, gene-level normalization maintains sparsity
    #     (no centering) and TruncatedSVD is used instead of normal PCA.

    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    if keep_sparse:
        if normalize: # normalize variance
            zstd = np.sqrt(sparse_var(E[base_ix,:]))
            Z = sparse_rowwise_multiply(E.T, 1 / zstd).T
        else:
            Z = E
        pca = TruncatedSVD(n_components=numpc, random_state=random_state)

    else:
        if normalize:
            zmean = E[base_ix,:].mean(0)
            zstd = np.sqrt(sparse_var(E[base_ix,:]))
            Z = sparse_rowwise_multiply((E - zmean).T, 1/zstd).T
        else:
            Z = E
        pca = PCA(n_components=numpc, random_state=random_state)

    pca.fit(Z[base_ix,:])
    return pca.transform(Z)



########## GENE FILTERING

def runningquantile(x, y, p, nBins):
    ''' calculate the quantile of y in bins of x '''

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
            else:
                yOut[i] = np.nan

    return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    '''
    Calculate v-score (above-Poisson noise statistic) for genes in the input sparse counts matrix
    Return v-scores and other stats
    '''

    ncell = E.shape[0]

    mu_gene = E.mean(axis=0).A.squeeze()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:,gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    gLog = lambda input: np.log(input[1] * np.exp(-input[0]) + input[2])
    h,b = np.histogram(np.log(FF_gene[mu_gene>0]), bins=200)
    b = b[:-1] + np.diff(b)/2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))
    errFun = lambda b2: np.sum(abs(gLog([x,c,b2])-y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func = errFun, x0=[b0], disp=False)
    a = c / (1+b) - 1


    v_scores = FF_gene / ((1+a)*(1+b) + b * mu_gene);
    CV_eff = np.sqrt((1+a)*(1+b) - 1);
    CV_input = np.sqrt(b);

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b

def filter_genes(E, base_ix = [], min_vscore_pctl = 85, min_counts = 3, min_cells = 3, show_vscore_plot = False, sample_name = ''):
    ''' 
    Filter genes by expression level and variability
    Return list of filtered gene indices
    '''

    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E[base_ix, :])
    ix2 = Vscores>0
    Vscores = Vscores[ix2]
    gene_ix = gene_ix[ix2]
    mu_gene = mu_gene[ix2]
    FF_gene = FF_gene[ix2]
    min_vscore = np.percentile(Vscores, min_vscore_pctl)
    ix = (((E[:,gene_ix] >= min_counts).sum(0).A.squeeze() >= min_cells) & (Vscores >= min_vscore))
    
    if show_vscore_plot:
        import matplotlib.pyplot as plt
        x_min = 0.5*np.min(mu_gene)
        x_max = 2*np.max(mu_gene)
        xTh = x_min * np.exp(np.log(x_max/x_min)*np.linspace(0,1,100))
        yTh = (1 + a)*(1+b) + b * xTh
        plt.figure(figsize=(4, 3));
        plt.scatter(np.log10(mu_gene), np.log10(FF_gene), c = [[.8,.8,.8]], alpha = 0.3, s = 3);
        plt.scatter(np.log10(mu_gene)[ix], np.log10(FF_gene)[ix], c = [[0,0,0]], alpha = 0.3,  s= 3);
        plt.plot(np.log10(xTh),np.log10(yTh));
        plt.title(sample_name)
        plt.xlabel('log10(mean)');
        plt.ylabel('log10(Fano factor)');
        plt.show()

    return gene_ix[ix]

def remove_corr_genes(E, gene_list, exclude_corr_genes_list, test_gene_idx, min_corr = 0.1):
    ''' remove signature-correlated genes from a list of test genes 
    Arguments:
    E: ssp.csc_matrix, shape (n_cells, n_genes)
        - full counts matrix
    gene_list: numpy array, shape (n_genes,)
        - full gene list
    exclude_corr_genes_list: list of list(s)
        - Each sublist is used to build a signature. Test genes correlated
          with this signature will be removed
    test_gene_idx: 1-D numpy array
        - indices of genes to test for correlation with the 
          gene signatures from exclude_corr_genes_list
    min_corr: float (default=0.1)
        - Test genes with a Pearson correlation of min_corr or higher 
          with any of the gene sets from exclude_corr_genes_list will
          be excluded

    Returns:
        numpy array of gene indices (subset of test_gene_idx) that 
        are not correlated with any of the gene signatures
    '''
    seed_ix_list = []
    for l in exclude_corr_genes_list:
        seed_ix_list.append(np.array([i for i in range(len(gene_list)) if gene_list[i] in l], dtype=int))

    exclude_ix = []
    for iSet in range(len(seed_ix_list)):
        seed_ix = seed_ix_list[iSet][E[:,seed_ix_list[iSet]].sum(axis=0).A.squeeze() > 0]
        if type(seed_ix) is int:
            seed_ix = np.array([seed_ix], dtype=int)
        elif type(seed_ix[0]) is not int:
            seed_ix = seed_ix[0]
        indat = E[:, seed_ix]
        tmp = sparse_zscore(indat)
        tmp = tmp.sum(1).A.squeeze()

        c = np.zeros(len(test_gene_idx))
        for iG in range(len(c)):
            c[iG],_ = scipy.stats.pearsonr(tmp, E[:,test_gene_idx[iG]].A.squeeze())

        exclude_ix.extend([test_gene_idx[i] for i in range(len(test_gene_idx)) if (c[i]) >= min_corr])
    exclude_ix = np.array(exclude_ix)

    return np.array([g for g in test_gene_idx if g not in exclude_ix], dtype=int)




#################################################################

# check if a given id is in the list L2 (day 24 or 46), or L4 (day26)
# a conversion algorithm 
def converting_id_from_fullSpace_to_subSpace(query_id_array_fullSpace,subSpace_id_array_inFull):
    id_sub=np.array(subSpace_id_array_inFull);
    query_id_inSub=[]
    query_success=np.zeros(len(query_id_array_fullSpace),dtype=bool)
    # check one by one
    for j,id_full in enumerate(query_id_array_fullSpace):
        temp=np.nonzero(id_sub==id_full)[0]
        if len(temp)>0:
            query_success[j]=True
            query_id_inSub.append(temp[0])
            
    return np.array(query_id_inSub), query_success
        


def converting_id_from_subSpace_to_fullSpace(query_id_array_subSpace,subSpace_id_array_inFull):
    return np.array(subSpace_id_array_inFull)[query_id_array_subSpace]




def compute_state_potential(input_matrix,state_annote,fate_array,fate_count=False,map_backwards=True):

    '''
        input_matrix: transition map 
        backward_map: True, Compute probability of ancester states to enter given fate clusters;
                      False, Compute probability of later states to originate from given state clusters;

        fate_array: targeted clusters
        
        The potential vector is normalized to be 1 due to the initial step of normalization for input matrix
    '''
    
    if not ssp.issparse(input_matrix): input_matrix=ssp.csr_matrix(input_matrix).copy()
    resol=10**(-10)
    input_matrix=sparse_rowwise_multiply(input_matrix,1/(resol+np.sum(input_matrix,1).A.flatten()))
    fate_N=len(fate_array)
    N1,N2=input_matrix.shape

    if map_backwards:
        idx_array=np.zeros((N2,fate_N),dtype=bool)
        for k in range(fate_N):
            idx_array[:,k]=(state_annote==fate_array[k])

        potential_vector=np.zeros((N1,fate_N))
        fate_entropy=np.zeros(N1)

        for k in range(fate_N):
            potential_vector[:,k]=np.sum(input_matrix[:,idx_array[:,k]],1).A.flatten()

        for j in range(N1):
                ### compute the "fate-entropy" for each state
            if fate_count:
                p0=potential_vector[j,:]
                fate_entropy[j]=np.sum(p0>0)
            else:
                p0=potential_vector[j,:]
                p0=p0/(resol+np.sum(p0))+resol
                for k in range(fate_N):
                    fate_entropy[j]=fate_entropy[j]-p0[k]*np.log(p0[k])

    ### forward map
    else:
        idx_array=np.zeros((N1,fate_N),dtype=bool)
        for k in range(fate_N):
            idx_array[:,k]=(state_annote==fate_array[k])

        potential_vector=np.zeros((N2,fate_N))
        fate_entropy=np.zeros(N2)

        for k in range(fate_N):
            potential_vector[:,k]=np.sum(input_matrix[idx_array[:,k],:],0).A.flatten()


        for j in range(N1):
                
                ### compute the "fate-entropy" for each state
            if fate_count:
                p0=potential_vector[j,:]
                fate_entropy[j]=np.sum(p0>0)
            else:
                p0=potential_vector[j,:]
                p0=p0/(resol+np.sum(p0))+resol
                for k in range(fate_N):
                    fate_entropy[j]=fate_entropy[j]-p0[k]*np.log(p0[k])

    return potential_vector, fate_entropy



def compute_fate_probability_map(adata,fate_array=[],used_map_name='transition_map',map_backwards=True):
    '''
        fate_array: targeted fate clusters. If not provided, use all fates in the annotation list. 
        use_transition_map: True, use transitino map; False, use demultiplexed map
    '''
    
    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    state_annote_0=adata.obs['state_annotation']
    if map_backwards:
        cell_id_t1=adata.uns['Tmap_cell_id_t1']
        cell_id_t2=adata.uns['Tmap_cell_id_t2']

    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']
        cell_id_t1=adata.uns['Tmap_cell_id_t2']

    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    data_des=adata.uns['data_des'][0]
    
    if len(fate_array)==0: fate_array=list(set(state_annote_0))
    

    state_annote_BW=state_annote_0[cell_id_t2]
    
    if used_map_name in adata.uns.keys():
        used_map=adata.uns[used_map_name]

        potential_vector, fate_entropy=compute_state_potential(used_map,state_annote_BW,fate_array,fate_count=True,map_backwards=map_backwards)

        adata.uns['fate_map']={'fate_array':fate_array,'fate_map':potential_vector,'fate_entropy':fate_entropy}

    else:
        print(f"Error, used_map_name should be among adata.uns.keys(), with _transition_map as suffix")

        
def compute_fate_map_and_bias(adata,selected_fates=[],used_map_name='transition_map',map_backwards=True):

    state_annote=adata.obs['state_annotation']
    valid_state_annot=list(set(np.array(state_annote)))
    if map_backwards:
        cell_id_t2=adata.uns['Tmap_cell_id_t2']
    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']

    if len(selected_fates)==0: selected_fates=list(set(state_annote))

    fate_array_flat=[] # a flatten list of cluster names
    fate_list_array=[] # a list of cluster lists, each cluster list is a macro cluster
    fate_list_descrip=[] # a list of string description for the macro cluster
    for xx in selected_fates:
        if type(xx) is list:
            fate_list_array.append(xx)
            des_temp=''
            for zz in xx:
                if zz in valid_state_annot:
                    fate_array_flat.append(zz)
                    des_temp=des_temp+str(zz)+'_'
                else:
                    print(f'{zz} is not a valid cluster name. Please select from: {valid_state_annot}')
            fate_list_descrip.append(des_temp)
        else:
            if xx in valid_state_annot:
                fate_list_array.append([xx])

                fate_array_flat.append(xx)
                fate_list_descrip.append(str(xx))
            else:
                print(f'{xx} is not a valid cluster name. Please select from: {valid_state_annot}')
                fate_list_descrip.append('')

    compute_fate_probability_map(adata,fate_array=fate_array_flat,used_map_name=used_map_name,map_backwards=map_backwards)
    fate_map_0=adata.uns['fate_map']['fate_map']

    N_macro=len(fate_list_array)
    fate_map=np.zeros((fate_map_0.shape[0],N_macro))
    extent_of_bias=np.zeros((fate_map_0.shape[0],N_macro))
    expected_bias=np.zeros(N_macro)
    for jj in range(N_macro):
        idx=np.in1d(fate_array_flat,fate_list_array[jj])
        fate_map[:,jj]=fate_map_0[:,idx].sum(1)

        for yy in fate_list_array[jj]:
            expected_bias[jj]=expected_bias[jj]+np.sum(state_annote[cell_id_t2]==yy)/len(cell_id_t2)

        # transformation
        temp_idx=fate_map[:,jj]<expected_bias[jj]
        temp_diff=fate_map[:,jj]-expected_bias[jj]
        extent_of_bias[temp_idx,jj]=temp_diff[temp_idx]/expected_bias[jj]
        extent_of_bias[~temp_idx,jj]=temp_diff[~temp_idx]/(1-expected_bias[jj])

        extent_of_bias[:,jj]=(extent_of_bias[:,jj]+1)/2 # rescale to the range [0,1]

    return fate_map,fate_list_descrip,extent_of_bias,expected_bias,fate_list_array
    

    

def mapout_trajectories(input_map,expre_0r,threshold=0.1,cell_id_t1=[],cell_id_t2=[]):
    '''
        input_map: a transition matrix (dimension: N1*N2) that connects subspace t1 (cell states in t1, total number N1) and subspace t2 (cell states in t2, total number N2) 
                    We assume that the input_map has been properly normalized. 
        The map is row-normalized first, before downstream application. This enhandce the robustness of the results.

        expre_0r: a continuous-valued vector (dimension N2) that defines the final cell states to be mapped to. 
                  The vector providues the weight of each cell state 

        cell_id_t1: the id array for cell states at t1
        cell_id_t2: the id array for cell states at t2


        Returns the fate probability of each cell state to enter a given cluster, as defined by the continuous-value vector expre_0r, at the next time point

        The fate probability vector is relatively thresholded

    '''

    ########## We assume that the input_map has been properly normalized.    
    # if not ssp.issparse(input_map): input_map=ssp.csr_matrix(input_map).copy()
    # resol=10**(-10)
    # input_map=sparse_rowwise_multiply(input_map,1/(resol+np.sum(input_map,1).A.flatten()))

    if ssp.issparse(input_map): input_map=input_map.A

    N1,N2=input_map.shape
    if len(cell_id_t1)==0 and N1==N2: # two idx at t1 and t2 are the same
        expre_1r=input_map.dot(expre_0r)
        expre_1r_idx=expre_1r>threshold*np.max(expre_1r)
        expre_1r_id=np.nonzero(expre_1r_idx)[0]
        expre_1r_truc=np.zeros(len(expre_1r))
        expre_1r_truc[expre_1r_id]=expre_1r[expre_1r_id]
    else:
        # both cell_id_t1 and cell_id_t2 are id's in the full space
        # selected_cell_id is also in the full space
        cell_id_t1=np.array(cell_id_t1)
        cell_id_t2=np.array(cell_id_t2)
        expre_0r_subspace=expre_0r[cell_id_t2]

        expre_1r=input_map.dot(expre_0r_subspace)
        expre_1r_idx=expre_1r>threshold*np.max(expre_1r)
        expre_1r_id=np.nonzero(expre_1r_idx)[0] # id in t1 subspace
        expre_1r_truc=expre_1r[expre_1r_id]
        expre_1r_truc=np.zeros(len(expre_1r))
        expre_1r_truc[expre_1r_id]=expre_1r[expre_1r_id]

    return expre_1r_truc

# def mapout_backward_trajectories_v1(input_map,expre_0r,threshold=0.1,cell_id_t1=[],cell_id_t2=[],row_norm=True):
#     '''
#         input_map: a transition matrix (dimension: N1*N2) that connects subspace t1 (cell states in t1, total number N1) and subspace t2 (cell states in t2, total number N2) 

#         The map is row-normalized first, before downstream application. This enhandce the robustness of the results.

#         expre_0r: a continuous-valued vector (dimension N2) that defines the final cell states to be mapped to. 
#                   The vector providues the weight of each cell state 

#         cell_id_t1: the id array for cell states at t1
#         cell_id_t2: the id array for cell states at t2


#         Returns the fate probability of each cell state to enter a given cluster, as defined by the continuous-value vector expre_0r, at the next time point

#         The fate probability vector is relatively thresholded

#     '''
    
#     if row_norm:
#         if not ssp.issparse(input_map): input_map=ssp.csr_matrix(input_map).copy()
#         resol=10**(-10)
#         input_map=sparse_rowwise_multiply(input_map,1/(resol+np.sum(input_map,1).A.flatten()))

#     if ssp.issparse(input_map): input_map=input_map.A

#     N1,N2=input_map.shape
#     if len(cell_id_t1)==0 and N1==N2: # two idx at t1 and t2 are the same
#         expre_1r=input_map.dot(expre_0r)
#         expre_1r_idx=expre_1r>threshold*np.max(expre_1r)
#         expre_1r_id=np.nonzero(expre_1r_idx)[0]
#         expre_1r_truc=np.zeros(len(expre_1r))
#         expre_1r_truc[expre_1r_id]=expre_1r[expre_1r_id]
#     else:
#         # both cell_id_t1 and cell_id_t2 are id's in the full space
#         # selected_cell_id is also in the full space
#         cell_id_t1=np.array(cell_id_t1)
#         cell_id_t2=np.array(cell_id_t2)
#         expre_0r_subspace=expre_0r[cell_id_t2]

#         expre_1r=input_map.dot(expre_0r_subspace)
#         expre_1r_idx=expre_1r>threshold*np.max(expre_1r)
#         expre_1r_id=np.nonzero(expre_1r_idx)[0] # id in t1 subspace
#         expre_1r_truc=expre_1r[expre_1r_id]
#         expre_1r_truc=np.zeros(len(expre_1r))
#         expre_1r_truc[expre_1r_id]=expre_1r[expre_1r_id]

#     return expre_1r_truc




# def mapout_forward_trajectories_v1(input_map,expre_0r,threshold=0.1,cell_id_t1=[],cell_id_t2=[],column_norm=True):
#     '''
#         Returns the ancestor states two-days/four-days backwards
#     '''

#     if column_norm:
#         if not ssp.issparse(input_map): input_map=ssp.csr_matrix(input_map).copy()
#         resol=10**(-10)
#         input_map=sparse_column_multiply(input_map,1/(resol+np.sum(input_map,0).A.flatten()))

#     if ssp.issparse(input_map): input_map=input_map.A

#     N1,N2=input_map.shape
#     if len(cell_id_t1)==0 and N1==N2: # two idx at t1 and t2 are the same
#         expre_1r=expre_0r.dot(input_map)
#         expre_1r_idx=expre_1r>threshold*np.max(expre_1r)
#         expre_1r_id=np.nonzero(expre_1r_idx)[0]
#         expre_1r_truc=np.zeros(len(expre_1r))
#         expre_1r_truc[expre_1r_id]=expre_1r[expre_1r_id]
#     else:
#         # both cell_id_t1 and cell_id_t2 are id's in the full space
#         # selected_cell_id is also in the full space
#         cell_id_t1=np.array(cell_id_t1)
#         cell_id_t2=np.array(cell_id_t2)
#         expre_0r_subspace=expre_0r[cell_id_t1]

#         expre_1r=expre_0r_subspace.dot(input_map)
#         expre_1r_idx=expre_1r>threshold*np.max(expre_1r)
#         expre_1r_id=np.nonzero(expre_1r_idx)[0] # id in t1 subspace
#         expre_1r_truc=expre_1r[expre_1r_id]
#         expre_1r_truc=np.zeros(len(expre_1r))
#         expre_1r_truc[expre_1r_id]=expre_1r[expre_1r_id]

#     return expre_1r_truc


def compute_shortest_path_distance_from_raw_matrix(data_matrix,num_neighbors_target=5,mode='connectivity',limit=np.inf):
    '''
        Data matrix is cell by 'gene' (gene can be different principle components)
        mode can be 'distance' or 'connectivity'
    '''
    adj_matrix = kneighbors_graph(data_matrix, num_neighbors_target, mode=mode, include_self=True)
    ShortPath_dis = dijkstra(csgraph = ssp.csr_matrix(adj_matrix), directed = False,return_predecessors = False,limit=limit)
    ShortPath_dis_max = np.nanmax(ShortPath_dis[ShortPath_dis != np.inf])
    ShortPath_dis[ShortPath_dis > ShortPath_dis_max] = ShortPath_dis_max #set threshold for shortest paths

    # Set normalized cost matrices based on shortest paths matrices at target and source spaces
    return ShortPath_dis / ShortPath_dis.max()





            
def add_neighboring_cells_to_a_map(input_map,adata,neighbor_N=5):
    input_map=input_map>0
    #print(f"Initial: {np.sum(input_map)}")
#     if (np.sum(input_map)<size_thresh) & (np.sum(input_map)>0):
#         #n0=np.round(size_thresh/np.sum(input_map))
#         #sc.pp.neighbors(adata, n_neighbors=int(n0)) #,method='gauss')
#         output_idx=adata.uns['neighbors']['connectivities'][input_map].sum(0).A.flatten()>0
#         input_map=input_map | output_idx

    sc.pp.neighbors(adata, n_neighbors=neighbor_N) #,method='gauss')
    output_idx=adata.uns['neighbors']['connectivities'][input_map].sum(0).A.flatten()>0
    out_map=input_map | output_idx
    #print(f"Final: {np.sum(out_map)}")

    return out_map


def compute_symmetric_Wasserstein_distance(sp_id_target,sp_id_ref,full_cost_matrix,target_value=[], ref_value=[],OT_epsilon=0.05,OT_stopThr=10**(-8),OT_max_iter=1000):
    import ot.bregman as otb
    # normalized distribution
    if len(target_value)==0:
        target_value=np.ones(len(sp_id_target))
    if len(ref_value)==0:
        ref_value=np.ones(len(sp_id_ref))
    
    input_mu=target_value/np.sum(target_value);
    input_nu=ref_value/np.sum(ref_value);

    full_cost_matrix_1=full_cost_matrix[sp_id_target][:,sp_id_ref]
    OT_transition_map_1=otb.sinkhorn_stabilized(input_mu,input_nu,full_cost_matrix_1,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

    full_cost_matrix_2=full_cost_matrix[sp_id_ref][:,sp_id_target]
    OT_transition_map_2=otb.sinkhorn_stabilized(input_nu,input_mu,full_cost_matrix_2,OT_epsilon,numItermax=OT_max_iter,stopThr=OT_stopThr)

    for_Wass_dis=np.sum(OT_transition_map_1*full_cost_matrix_1)
    back_Wass_dis=np.sum(OT_transition_map_2*full_cost_matrix_2)
    return [for_Wass_dis, back_Wass_dis, (for_Wass_dis+back_Wass_dis)/2]


    
def get_normalized_covariance(data,method='Caleb'):
    if method=='Caleb':
        cc = np.cov(data.T)
        mm = np.mean(data,axis=0) + .0001
        X,Y = np.meshgrid(mm,mm)
        cc = cc / X / Y
        return cc#/np.max(cc)
    else:
        resol=10**(-10)
        
        # No normalization performs better.  Not all cell states contribute equally to lineage coupling
        # Some cell states are in the progenitor regime, most ambiguous. They have a larger probability to remain in the progenitor regime, rather than differentiate.
        # Normalization would force these cells to make early choices, which could add noise to the result. 
        # data=core.sparse_rowwise_multiply(data,1/(resol+np.sum(data,1)))
        
        X=data.T.dot(data)
        diag_temp=np.sqrt(np.diag(X))
        for j in range(len(diag_temp)):
            for k in range(len(diag_temp)):
                X[j,k]=X[j,k]/(diag_temp[j]*diag_temp[k])
        return X#/np.max(X)
    


def above_the_line(x_array,x1,x2):
    return (x_array[:,1]-x1[1])>((x2[1]-x1[1])/(x2[0]-x1[0]))*(x_array[:,0]-x1[0])
