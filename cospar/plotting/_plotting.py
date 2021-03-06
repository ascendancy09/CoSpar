import numpy as np
import time
from plotnine import *  
from sklearn import manifold
import pandas as pd
import scanpy as sc
import pdb
import os
import scipy.sparse as ssp
from .. import help_functions as hf
from matplotlib import pyplot as plt
from .. import settings
from .. import logging as logg

####################

## General

####################
    

def darken_cmap(cmap, scale_factor):
    """
    Generate a gradient color map for plotting.
    """

    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii, 0] = curcol[0] * scale_factor
        cdat[ii, 1] = curcol[1] * scale_factor
        cdat[ii, 2] = curcol[2] * scale_factor
        cdat[ii, 3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap

def start_subplot_figure(n_subplots, n_columns=5, fig_width=14, row_height=3):
    """
    Generate a figure object with given subplot specification
    """

    n_rows = int(np.ceil(n_subplots / float(n_columns)))
    fig = plt.figure(figsize = (fig_width, n_rows * row_height))
    return fig, n_rows, n_columns

def plot_one_gene(E, gene_list, gene_to_plot, x, y, normalize=False, 
    ax=None, order_points=True, col_range=(0,100), buffer_pct=0.03, 
    point_size=1, color_map=None, smooth_operator=None):
    """
    Plot the expression of a list of genes on an embedding

    Parameters
    ----------
    E: `sp.sparse`
        Cell-by-gene expression matrix
    gene_list: `list`
        Full list of gene names corresponding to E
    gene_to_plot, `list`
        List of genes to be plotted. 
    x: `np.array`
        x coordinate of the embedding
    y: `np.array`
        y coordinate of the embedding
    color_map: {plt.cm.Reds,plt.cm.Blues,...}, (default: None)
    ax: `axis`, optional (default: None)
        An external ax object can be passed here.
    order_points: `bool`, optional (default: True)
        Order points to plot by the gene expression 
    col_range: `tuple`, optional (default: (0,100))
        The color range to plot. The range should be within [0,100]
    buffer_pct: `float`, optional (default: 0.03)
        Extra space for the plot box frame
    point_size: `int`, optional (default: 1)
        Size of the data point
    smooth_operator: `np.array`, optional (default: None)
        A smooth matrix to be applied to the subsect of gene expression matrix. 

    Returns
    -------
    pp: the figure object
    """

    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds,.9)
    if ax is None:
        fig,ax=plt.subplots()
        
    if normalize:
        E = tot_counts_norm(E, target_mean=1e6)[0]
    
    k = list(gene_list).index(gene_to_plot)
    coldat = E[:,k].A
    
    if smooth_operator is None:
        coldat = coldat.squeeze()
    else:
        coldat = np.dot(smooth_operator, coldat).squeeze()
    
    if order_points:
        o = np.argsort(coldat)
    else:
        o = np.arange(len(coldat))
        
    vmin = np.percentile(coldat, col_range[0])
    vmax = np.percentile(coldat, col_range[1])
    if vmax==vmin:
        vmax = coldat.max()
        
    pp = ax.scatter(x[o], y[o], c=coldat[o], s=point_size, cmap=color_map,
               vmin=vmin, vmax=vmax)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min()-x.ptp()*buffer_pct, x.max()+x.ptp()*buffer_pct)
    ax.set_ylim(y.min()-y.ptp()*buffer_pct, y.max()+y.ptp()*buffer_pct)
    
    return pp


def embedding(adata,basis='X_emb',color=None):
    """
    Scatter plot for user specified embedding basis.

    We imported scanpy.pl.embedding for this purpose.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    basis: `str`, optional (default: 'X_emb')
        The embedding to use for the plot.
    color: `str, list of str, or None` (default: None)
        Keys for annotations of observations/cells or variables/genes, 
        e.g., 'state_info', 'time_info',['Gata1','Gata2']
    """

    sc.pl.embedding(adata,basis=basis,color=color)


def customized_embedding(x, y, vector, normalize=False, title=None, ax=None, 
    order_points=True, set_ticks=False, col_range=(0, 100), buffer_pct=0.03, point_size=1, 
    color_map=None, smooth_operator=None,set_lim=True,
    vmax=np.nan,vmin=np.nan,color_bar=False):
    """
    Plot a vector on an embedding

    Parameters
    ----------
    x: `np.array`
        x coordinate of the embedding
    y: `np.array`
        y coordinate of the embedding
    vector: `np.array`
        A vector to be plotted.
    color_map: {plt.cm.Reds,plt.cm.Blues,...}, (default: None)
    ax: `axis`, optional (default: None)
        An external ax object can be passed here.
    order_points: `bool`, optional (default: True)
        Order points to plot by the gene expression 
    col_range: `tuple`, optional (default: (0,100))
        The color range to plot. The range should be within [0,100]
    buffer_pct: `float`, optional (default: 0.03)
        Extra space for the plot box frame
    point_size: `int`, optional (default: 1)
        Size of the data point
    smooth_operator: `np.array`, optional (default: None)
        A smooth matrix to be applied to the subsect of gene expression matrix. 
    set_lim: `bool`, optional (default: True)
        Set the plot range (x_limit, and y_limit) automatically.
    vmax: `float`, optional (default: np.nan)
        Maximum color range (saturation). 
        All values above this will be set as vmax.
    vmin: `float`, optional (default: np.nan)
        The minimum color range, all values below this will be set to be vmin.
    color_bar: `bool`, optional (default, False)
        If True, plot the color bar. 
    set_ticks: `bool`, optional (default, False)
        If False, remove figure ticks.   
    """

    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds, .9)
    if ax is None:
        fig, ax = plt.subplots()

    coldat = vector

    if smooth_operator is None:
        coldat = coldat.squeeze()
    else:
        coldat = np.dot(smooth_operator, coldat).squeeze()

    if order_points:
        o = np.argsort(coldat)
    else:
        o = np.arange(len(coldat))
    if np.isnan(vmin):
        vmin = np.percentile(coldat, col_range[0])
    if np.isnan(vmax):
        vmax = np.percentile(coldat, col_range[1])
    if vmax == vmin:
        vmax = coldat.max()

    pp = ax.scatter(x[o], y[o], c=coldat[o], s=point_size, cmap=color_map,
                    vmin=vmin, vmax=vmax)

    if not set_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    if set_lim==True:
        ax.set_xlim(x.min() - x.ptp() * buffer_pct, x.max() + x.ptp() * buffer_pct)
        ax.set_ylim(y.min() - y.ptp() * buffer_pct, y.max() + y.ptp() * buffer_pct)

    if title is not None:
        ax.set_title(title)

    if color_bar:
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax)

    # if savefig:
    #     fig.savefig(f'figure/customized_embedding_fig_{int(np.round(np.random.rand()*100))}.{settings.file_format_figs}')


def gene_expression_on_manifold(adata,selected_genes,savefig=False,plot_time_points=[],color_bar=False):
    """
    Plot gene expression on the state manifold

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_genes: `list` or 'str'
        List of genes to plot.
    savefig: `bool`, optional (default: False)
        Save the figure.
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    color_bar: `bool`, (default: False)
        If True, plot the color bar. 
    """


    #hf.check_available_map(adata)
    #set_up_plotting()
        
    if type(selected_genes)==str:
        selected_genes=[selected_genes]

    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    figure_path=settings.figure_path

    time_info=np.array(adata.obs['time_info'])
    if (len(plot_time_points)>0):
        sp_idx=np.zeros(adata.shape[0],dtype=bool)
        for xx in plot_time_points:
            sp_id_temp=np.nonzero(time_info==xx)[0]
            sp_idx[sp_id_temp]=True
    else:
        sp_idx=np.ones(adata.shape[0],dtype=bool)
            
    for j in range(len(selected_genes)):
        genes_plot=[selected_genes[j]]

        gene_list=adata.var_names

        col=1
        row=1
        fig = plt.figure(figsize=(4 * col, 3 * row))

        # Plot each gene's expression in a different subplot
        for iG,g in enumerate(genes_plot):
            ax = plt.subplot(row, col, iG+1)
            plot_one_gene(adata.X[sp_idx], gene_list, g, x_emb[sp_idx]-200, y_emb[sp_idx], ax=ax, col_range=(0, 99.8), point_size=2)
            ax.set_title(f'{g}')
            ax.axis('off')
    
        if color_bar:
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax)

        fig.tight_layout()

        if savefig:
            plt.savefig(f'{figure_path}/lung_marker_genes_{selected_genes[j]}.{settings.file_format_figs}')


####################

## Fate bias analysis

####################

def single_cell_transition(adata,selected_state_id_list,used_map_name='transition_map',map_backwards=True,savefig=False,point_size=3):
    """
    Plot transition probability from given initial cell states

    If `map_backwards=True`, plot future state probability starting from given initial;
    else, plot probability of source states where the current cell state come from.  

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_state_id_list: `list`
        List of cell id's. Like [0,1,2].
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    point_size: `int`, optional (default: 2)
        Size of the data point.
    save_fig: `bool`, optional (default: False)
        If true, save figure to defined directory at settings.figure_path
    """

    hf.check_available_map(adata)
    #set_up_plotting()
    state_annote=adata.obs['state_info']
    cell_id_t2=adata.uns['Tmap_cell_id_t2']
    cell_id_t1=adata.uns['Tmap_cell_id_t1']
    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    figure_path=settings.figure_path

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:
        prediction_array=[adata.uns[used_map_name]]
        method_descrip=[used_map_name]


        for k, matrix in enumerate(prediction_array):
                #pdb.set_trace()
            if ssp.issparse(matrix): prediction_array[k]=prediction_array[k].A

                
        disp_name = 1
        row = len(selected_state_id_list)
        col = len(prediction_array)
        fig = plt.figure(figsize=(4 * col, 3 * row))

        for j, target_cell_ID in enumerate(selected_state_id_list):
            if j > 0:
                disp_name = 0
            ax0 = plt.subplot(row, col, col * j + 1)


            for k, matrix in enumerate(prediction_array):
                if not map_backwards:
                    prob_vec=np.zeros(len(x_emb))
                    prob_vec[cell_id_t2]=matrix[target_cell_ID, :]
                    customized_embedding(x_emb, y_emb, prob_vec, point_size=point_size, ax=ax0)
                    
                    ax0.plot(x_emb[cell_id_t1][target_cell_ID],y_emb[cell_id_t1][target_cell_ID],'*b',markersize=3*point_size)
                    if disp_name:
                        ax0.set_title(f"t1 state (blue star) ({cell_id_t1[target_cell_ID]})")
                    else:
                        ax0.set_title(f"ID: {cell_id_t1[target_cell_ID]}")

                else:
                    prob_vec=np.zeros(len(x_emb))
                    prob_vec[cell_id_t1]=matrix[:,target_cell_ID]
                    customized_embedding(x_emb, y_emb,prob_vec, point_size=point_size, ax=ax0)
                    
                    ax0.plot(x_emb[cell_id_t2][target_cell_ID],y_emb[cell_id_t2][target_cell_ID],'*b',markersize=3*point_size)
                    if disp_name:
                        ax0.set_title(f"t2 fate (blue star) ({cell_id_t2[target_cell_ID]})")
                    else:
                        ax0.set_title(f"ID: {cell_id_t2[target_cell_ID]}")


        Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Probability')


        plt.tight_layout()
        if savefig:
            fig.savefig(f"{figure_path}/plotting_transition_map_probability_{map_backwards}.{settings.file_format_figs}")


def fate_map(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,plot_time_points=[],plot_background=True,
    plot_target_state=True,normalize=False,auto_color_scale=True,plot_color_bar=True,
    point_size=2,alpha_target=0.2,figure_index='',plot_horizontal=False):
    """
    Plot transition probability to given fate/ancestor clusters

    If `map_backwards=True`, plot transition probability of early 
    states to given fate clusters (fate map); else, plot transition 
    probability of later states from given ancestor clusters (ancestor map).
    Figures are saved at defined directory at settings.figure_path.

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey color as the background. 
    plot_target_state: `bool`, optional (default: True)
        If true, color the target cluster as defined in selected_fates in cyan.
    normalize: `bool`, optional (default: False)
        If true, normalize the fate map towards selected clusters. This
        seems to be redundant as the computation of fate_map requires 
        normalization at the whole Tmap level.
    plot_color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    point_size: `int`, optional (default: 2)
        Size of the data point.
    alpha_target: `float`, optional (default: 0.2)
        It controls the transpancy of the plotted target cell states, 
        for visual effect. Range: [0,1].
    figure_index: `str`, optional (default, '')
        A string to label different figures when saving.
    plot_horizontal: `bool`, optional (default: False)
        If true, plot the figure panels horizontally; else, vertically.


    Returns
    -------
    Store a dictionary of results {"fate_map","relative_bias","expected_prob"} at adata.uns['fate_map_output']. 
    """

    hf.check_available_map(adata)
    #set_up_plotting()

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        if (len(plot_time_points)>0):
            sp_idx=np.zeros(len(cell_id_t1),dtype=bool)
            for xx in plot_time_points:
                sp_id_temp=np.nonzero(time_info[cell_id_t1]==xx)[0]
                sp_idx[sp_id_temp]=True
        else:
            sp_idx=np.ones(len(cell_id_t1),dtype=bool)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path



        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

        if '' not in mega_cluster_list:
            # normalize the map to enhance the fate choice difference among selected clusters
            if normalize and (fate_map.shape[1]>1):
                resol=10**-10 
                fate_map=hf.sparse_rowwise_multiply(fate_map,1/(resol+np.sum(fate_map,1)))
                #fate_entropy_temp=fate_entropy_array[x0]


            ################### plot fate probability
            vector_array=[vector for vector in list(fate_map.T)]
            description=[fate for fate in mega_cluster_list]
            if plot_horizontal:
                row =1; col =len(vector_array)
            else:
                row =len(vector_array); col =1

            fig = plt.figure(figsize=(4.5 * col, 3.5 * row))
            for j in range(len(vector_array)):
                ax0 = plt.subplot(row, col, j + 1)
                
                if plot_background:
                    customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0,title=description[j])            
                    if plot_target_state:
                        for zz in valid_fate_list[j]:
                            idx_2=state_annote==zz
                            ax0.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=alpha_target)
                else:
                    customized_embedding(x_emb[cell_id_t1],y_emb[cell_id_t1],np.zeros(len(y_emb[cell_id_t1])),point_size=point_size,ax=ax0,title=description[j])

                if auto_color_scale:
                    customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],point_size=point_size,ax=ax0,title=description[j],set_lim=False)
                else:
                    customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],point_size=point_size,ax=ax0,title=description[j],set_lim=False,vmax=1,vmin=0)
            
            if plot_color_bar:
                fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Fate probability')
          
            #yy=int(np.random.rand()*100)
            fig.savefig(f'{figure_path}/{data_des}_fate_map_overview_{figure_index}.{settings.file_format_figs}')

            ## save data to adata
            adata.uns['fate_map_output']={"fate_map":fate_map[sp_idx,:],"relative_bias":relative_bias[sp_idx,:],"expected_prob":expected_prob}
        else:
            logg.error(f"Selected fates are not valid.")


def relative_fate_bias(adata,selected_fates=[],used_map_name='transition_map',
    method='intrinsic',map_backwards=True,plot_time_points=[],point_size=1,
    sum_fate_prob_thresh=0,plot_background=True,plot_target_state=False,
    plot_color_bar=True,plot_horizontal=True,alpha_target=0.2):
    """
    Plot the relative fate bias towards given clusters.

    The relative fate bias is defined in two ways: 

    * if method='intrinsic',
    we calculate the fate bias as predicted_fate_prob/expected_prob, where
    the expected probability of a cluster is given by the relative size. This
    definition is per a cluster. 

    * If method='competition', we assume two clusters {A,B} are selected, and the
    bias for a state is defined as preference of fate A over B. 

    If `map_backwards=True`, plot fate bias of early 
    states to given fate clusters; else, plot fate bias of later states 
    from given ancestor clusters. Figures are saved at defined 
    directory at settings.figure_path.

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`, optional (default: transition_map)
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    method: `str`, optional (default: intrinsic)
        Method used to define the relative fate bias. See above. 
        Choices: {'intrinsic, competition'}.
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    sum_fate_prob_thresh: `float`, optional (default: 0)
        The fate bias of a state is plotted only when it has a cumulative fate 
        probability to the combined cluster (A+B) larger than this threshold,
        i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh. Only activated 
        when method='competition'.
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_target_state: `bool`, optional (default: True)
        If true, color the target clusters as defined in selected_fates in cyan.
    plot_background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey color as the background. 
    plot_color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    point_size: `int`, optional (default: 2)
        Size of the data point.
    plot_horizontal: `bool`, optional (default: True)
        Arrange the subplots horizontally. 
    alpha_target: `float`, optional (default: 0.2)
        It controls the transpancy of the plotted target cell states, 
        for visual effect. Range: [0,1].
    """

    if method=='intrinsic':
        fate_bias_intrinsic(adata,selected_fates=selected_fates,used_map_name=used_map_name,
        map_backwards=map_backwards,plot_time_points=plot_time_points,point_size=point_size,
        plot_target_state=plot_target_state,plot_color_bar=plot_color_bar,
        plot_horizontal=plot_horizontal,alpha_target=alpha_target)
    else:
        if len(selected_fates)!=2:
            logg.error("When using method=competition, please provide exactly 2 fates!") 
        else:
            fate_bias_from_binary_competition(adata,selected_fates=selected_fates,
            used_map_name=used_map_name,map_backwards=map_backwards,
            sum_fate_prob_thresh=sum_fate_prob_thresh,plot_time_points=plot_time_points,
            point_size=point_size,plot_target_state=plot_target_state,
            plot_color_bar=plot_color_bar,alpha_target=alpha_target)



def fate_bias_intrinsic(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,plot_time_points=[],point_size=1,plot_background=True,
    plot_target_state=False,plot_color_bar=True,plot_horizontal=True,alpha_target=0.2):
    """
    Plot the relative fate bias (predicted/expected) towards a cluster.

    Compared with :func:`fate_map`, we normalized the fate probability by the 
    relative proportion of the targeted cluster, as the plotted relative fate bias. 

    If `map_backwards=True`, plot fate bias of early 
    states to given fate clusters; else, plot fate bias of later states 
    from given ancestor clusters. Figures are saved at defined 
    directory at settings.figure_path.

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_target_state: `bool`, optional (default: True)
        If true, color the target clusters as defined in selected_fates in cyan.
    plot_background: `bool`, optional (default: True)
        If true, plot all cell states (t1+t2) in grey color as the background. 
    plot_color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    point_size: `int`, optional (default: 2)
        Size of the data point.
    plot_horizontal: `bool`, optional (default: True)
        Arrange the subplots horizontally. 
    alpha_target: `float`, optional (default: 0.2)
        It controls the transpancy of the plotted target cell states, 
        for visual effect. Range: [0,1].

    Returns
    -------
    Store a dictionary of results {"fate_map","relative_bias","expected_prob"} at adata.uns['fate_map_output']. 
    """

    hf.check_available_map(adata)
    #set_up_plotting()

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        if (len(plot_time_points)>0):
            sp_idx=np.zeros(len(cell_id_t1),dtype=bool)
            for xx in plot_time_points:
                sp_id_temp=np.nonzero(time_info[cell_id_t1]==xx)[0]
                sp_idx[sp_id_temp]=True
        else:
            sp_idx=np.ones(len(cell_id_t1),dtype=bool)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path



        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

        if '' not in mega_cluster_list:
            ################# plot the extent of fate bias as compared with null hypothesis: random transitions
            vector_array=[vector for vector in list(relative_bias.T)]
            description=[fate for fate in mega_cluster_list]

            if plot_horizontal:
                row =1; col =len(vector_array)
            else:
                row =len(vector_array); col =1

            fig = plt.figure(figsize=(4.5 * col, 3.5 * row))
            for j in range(len(vector_array)):
                ax0 = plt.subplot(row, col, j + 1)
                # customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0,title=description[j])
                # if plot_target_state:
                #     for zz in valid_fate_list[j]:
                #         idx_2=state_annote==zz
                #         ax0.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*2,alpha=alpha_target)

                if plot_background:
                    customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0,title=description[j])            
                    if plot_target_state:
                        for zz in valid_fate_list[j]:
                            idx_2=state_annote==zz
                            ax0.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=alpha_target)

                temp_array=vector_array[j][sp_idx]
                new_idx=np.argsort(abs(temp_array-0.5))
                customized_embedding(x_emb[cell_id_t1][sp_idx][new_idx],y_emb[cell_id_t1][sp_idx][new_idx],temp_array[new_idx],point_size=point_size,ax=ax0,title=description[j],color_map=plt.cm.bwr,set_lim=False,vmax=1,vmin=0)

            if plot_color_bar:
                Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), ax=ax0,label='Predicted/expected bias')
                Clb.set_ticks([])
                #Clb.ax.set_title(f'description[j]')

            fig.savefig(f'{figure_path}/{data_des}_intrinsic_fate_bias_BW{map_backwards}.{settings.file_format_figs}')


            fig = plt.figure(figsize=(4.5 * col, 3.5 * row))
            for j in range(len(vector_array)):
                ax = plt.subplot(row, col, j + 1)

                temp_array=vector_array[j][sp_idx]
                new_idx=np.argsort(abs(temp_array-0.5))
                xxx=temp_array[new_idx]
                ax = plt.subplot(row, col, j + 1)
                ax.hist(xxx,50,color='#2ca02c')
                ax.set_xlim([0,1])
                ax.set_xlabel('Relative fate bias')
                ax.set_ylabel('Histogram')
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_title(f'{description[j]}, Ave.: {int(np.mean(xxx)*100)/100}',fontsize=16)
                fig.savefig(f'{figure_path}/{data_des}_intrinsic_fate_bias_BW{map_backwards}_histogram.{settings.file_format_figs}')
            
            ## save data to adata
            adata.uns['fate_map_output']={"fate_map":fate_map[sp_idx,:],"relative_bias":relative_bias[sp_idx,:],"expected_prob":expected_prob}

        else:
            logg.error(f"Selected fates are not valid.")





def fate_bias_from_binary_competition(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,sum_fate_prob_thresh=0,plot_time_points=[],point_size=1,
    plot_target_state=False,plot_color_bar=True,alpha_target=0.2):
    """
    Plot fate bias to given two fate/ancestor clusters (A, B).

    If `map_backwards=True`, plot fate bias of early 
    states to given fate clusters; else, plot fate bias of later states 
    from given ancestor clusters. Figures are saved at defined 
    directory at settings.figure_path.

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    sum_fate_prob_thresh: `float`, optional (default: 0)
        The fate bias of a state is plotted only when it has a cumulative fate 
        probability to the combined cluster (A+B) larger than this threshold,
        i.e., P(i->A)+P(i+>B) >  sum_fate_prob_thresh. 
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_target_state: `bool`, optional (default: True)
        If true, color the target clusters as defined in selected_fates in cyan.
    plot_color_bar: `bool`, optional (default: True)
        plot the color bar if True.
    point_size: `int`, optional (default: 2)
        Size of the data point.
    alpha_target: `float`, optional (default: 0.2)
        It controls the transpancy of the plotted target cell states, 
        for visual effect. Range: [0,1].

    Returns
    -------
    The results are stored at adata.uns['relative_fate_bias']
    """

    hf.check_available_map(adata)
    #set_up_plotting()
    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        figure_path=settings.figure_path


        ## select time points
        time_info=np.array(adata.obs['time_info'])
        if (len(plot_time_points)>0):
            sp_idx=np.zeros(len(cell_id_t1),dtype=bool)
            for xx in plot_time_points:
                sp_id_temp=np.nonzero(time_info[cell_id_t1]==xx)[0]
                sp_idx[sp_id_temp]=True
        else:
            sp_idx=np.ones(len(cell_id_t1),dtype=bool)

        cell_id_t1_sp=cell_id_t1[sp_idx]
            

        if len(selected_fates)!=2: 
            logg.error(f"Must have only two fates")
        else:
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)


            if '' not in mega_cluster_list:
                resol=10**(-10)

                fig=plt.figure(figsize=(5,4))
                ax=plt.subplot(1,1,1)
                #logg.error(fate_map.shape)
                potential_vector_temp=fate_map[sp_idx,:]


                #potential_vector_temp=hf.sparse_rowwise_multiply(potential_vector_temp,1/(resol+np.sum(potential_vector_temp,1)))
                potential_vector_temp=potential_vector_temp+resol
                diff=potential_vector_temp[:,0]#-potential_vector_temp[:,1]
                tot=potential_vector_temp.sum(1)

                valid_idx=tot>sum_fate_prob_thresh # default 0.5
                vector_array=np.zeros(np.sum(valid_idx))
                vector_array=diff[valid_idx]/(tot[valid_idx])
                #vector_array=2*potential_vector_temp[valid_idx,8]/tot[valid_idx]-1
                #vector_array=potential_vector_temp[:,8]/potential_vector_temp[:,9]

                #customized_embedding(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],np.zeros(len(y_emb[cell_id_t1][sp_idx])),point_size=point_size,ax=ax)
                if plot_target_state:
                    customized_embedding(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax)
         
                    for zz in valid_fate_list[0]:
                        idx_2=state_annote[cell_id_t2]==zz
                        ax.plot(x_emb[cell_id_t2[idx_2]],y_emb[cell_id_t2[idx_2]],'.',color='red',markersize=point_size*2,alpha=alpha_target)
                    for zz in valid_fate_list[1]:
                        idx_2=state_annote[cell_id_t2]==zz
                        ax.plot(x_emb[cell_id_t2[idx_2]],y_emb[cell_id_t2[idx_2]],'.',color='blue',markersize=point_size*2,alpha=alpha_target)

                        
                else:
                    customized_embedding(x_emb[cell_id_t1_sp],y_emb[cell_id_t1_sp],np.zeros(len(y_emb[cell_id_t1_sp])),point_size=point_size,ax=ax)
                #customized_embedding(x_emb[cell_id_t2],y_emb[cell_id_t2],np.zeros(len(y_emb[cell_id_t2])),point_size=point_size,ax=ax)

                new_idx=np.argsort(abs(vector_array-0.5))
                customized_embedding(x_emb[cell_id_t1_sp][valid_idx][new_idx],y_emb[cell_id_t1_sp][valid_idx][new_idx],
                                    vector_array[new_idx],vmax=1,vmin=0,
                                    point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

        #         # remove un-wanted time points
        #         if len(cell_id_t1[~sp_idx])>0:
        #             customized_embedding(x_emb[cell_id_t1[~sp_idx]],y_emb[cell_id_t1[~sp_idx]],np.zeros(len(y_emb[cell_id_t1[~sp_idx]])),
        #                         point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

                if plot_color_bar:
                    Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), ax=ax,label='Fate bias')
                    Clb.ax.set_title(f'{mega_cluster_list[0]}')

                fig.savefig(f'{figure_path}/{data_des}_binary_fate_bias_BW{map_backwards}.{settings.file_format_figs}')


                adata.uns['relative_fate_bias']=vector_array[new_idx]
                xxx=adata.uns['relative_fate_bias']
                fig=plt.figure(figsize=(4,3.5));ax=plt.subplot(1,1,1)
                ax.hist(xxx,50,color='#2ca02c')
                ax.set_xlim([0,1])
                ax.set_xlabel('Binary fate bias')
                ax.set_ylabel('Histogram')
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_title(f'Average: {int(np.mean(xxx)*100)/100}',fontsize=16)
                fig.savefig(f'{figure_path}/{data_des}_binary_fate_bias_BW{map_backwards}_histogram.{settings.file_format_figs}')
            else:
                logg.error(f"Selected fates are not valid.")


def fate_coupling_from_Tmap(adata,selected_fates=[],used_map_name='transition_map',plot_time_points=[],normalize_fate_map=False,color_bar=True,coupling_normalization='SW',rename_selected_fates=[]):

    """
    Plot fate coupling determined by Tmap.

    We use the fate map of cell states at t1 to compute the fate coupling.

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    normalize_fate_map: `bool`, optional (default: False)
        If true, normalize fate map before computing the fate coupling. 
    color_bar: `bool`, optional (default: True)
        Plot the color bar.
    coupling_normalization: `str`, optional (default: 'SW')
        Method to normalize the coupling matrix: {'SW','Weinreb'}
    rename_selected_fates: `list`, optional (default: [])
        Provide new names in substitution of names in selected_fates.
    """

    hf.check_available_map(adata)
    #set_up_plotting()
    
    map_backwards=True
    
    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_info']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']


        time_info=np.array(adata.obs['time_info'])
        if (len(plot_time_points)>0):
            sp_idx=np.zeros(len(cell_id_t1),dtype=bool)
            for xx in plot_time_points:
                sp_id_temp=np.nonzero(time_info[cell_id_t1]==xx)[0]
                sp_idx[sp_id_temp]=True
        else:
            sp_idx=np.ones(len(cell_id_t1),dtype=bool)


        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        data_des=adata.uns['data_des'][-1]
        data_des=f'{data_des}_Tmap_fate_coupling'
        figure_path=settings.figure_path



        fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

        if '' not in mega_cluster_list:
            # normalize the map to enhance the fate choice difference among selected clusters
            if normalize_fate_map and (fate_map.shape[1]>1):
                resol=10**-10 
                fate_map=hf.sparse_rowwise_multiply(fate_map,1/(resol+np.sum(fate_map,1)))
                #fate_entropy_temp=fate_entropy_array[x0]
            
    
            if len(rename_selected_fates)!=len(mega_cluster_list):
                rename_selected_fates=mega_cluster_list

            X_ICSLAM = hf.get_normalized_covariance(fate_map[sp_idx],method=coupling_normalization)
            heatmap(figure_path, X_ICSLAM, rename_selected_fates,color_bar_label='Coupling',color_bar=color_bar,data_des=data_des)


####################

## DDE analysis

####################


def dynamic_trajectory_from_intrinsic_bias(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,plot_time_points=[],bias_threshold_A=0.5,bias_threshold_B=0.5,
    plot_ancestor=True,point_size=2,savefig=False,plot_target_state=True,alpha_target=0.2,
    avoid_target_states=False):
    """
    Identify trajectory towards/from two given clusters.

    If `map_backwards=True`, use the fate probability towards given clusters (A,B)
    to define the ancestor population for A and B. The ancestor population + target fates 
    are the dynamic trajectory. 

    Fate bias at each state is a `vector`: (`relative_bias_A,relative_bias_B`). 
    For a given cluster, is defined by comparing predicted fate probability with 
    expected one, same as :func:`.fate_map`. It sales from [0,1], with 0.5 as the 
    neutral prediction that agrees with null hypothesis. Selected ancestors satisfy: 

        * For A: `relative_bias_A` > `bias_threshold_A`

        * For B: `relative_bias_B` > `bias_threshold_B`

    If `map_backwards=False`, use the probability from given clusters (A,B)
    to define the fate population, and perform DGE analysis. This is not useful. 

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    bias_threshold_A: `float`, optional (default: 0.5)
        Relative fate bias threshold to select ancestor population for cluster A.
        The 0.5 is the neural bias point. Range: [0,1]. 
        Closer to 1 means more stringent selection. 
    bias_threshold_B: `float`, optional (default: 0.5)
        Relative fate bias threshold to select ancestor population for cluster B.
        Closer to 1 means more stringent selection. 
    savefig: `bool`, optional (default: False)
        Save all plots.
    point_size: `int`, optional (default: 2)
        Size of the data point.
    avoid_target_states: `bool`, optional (default: False)
        If true, avoid selecting cells at the target cluster (A, or B) as 
        ancestor popolation.
    plot_target_state: `bool`, optional (default: True)
        Plot the target states in the state manifold. 
    alpha_target: `float`, optional (default: 0.2)
        Transparency parameter for plotting. 

    Returns
    -------
    Store the inferred ancestor states in adata.uns['cell_group_A'] and adata.uns['cell_group_B']

    Combine ancestor states and target states into adata.uns['dynamic_trajectory'] for each fate. 
    """

    diff_gene_A=[]
    diff_gene_B=[]
    hf.check_available_map(adata)
    #set_up_plotting()

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")


    else:
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        figure_path=settings.figure_path
        state_annote_t1=np.array(adata.obs['state_info'][cell_id_t1])

        if (len(selected_fates)!=1) and (len(selected_fates)!=2):
            logg.error(f" Must provide one or two fates.")

        else:
            ## select time points
            time_info=np.array(adata.obs['time_info'])
            if (len(plot_time_points)>0):
                sp_idx=np.zeros(len(cell_id_t1),dtype=bool)
                for xx in plot_time_points:
                    sp_id_temp=np.nonzero(time_info[cell_id_t1]==xx)[0]
                    sp_idx[sp_id_temp]=True
            else:
                sp_idx=np.ones(len(cell_id_t1),dtype=bool)

            #if 'fate_map' not in adata.uns.keys():
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

            if '' not in mega_cluster_list:

                if len(selected_fates)==1:
                    zz=relative_bias[:,0]
                    idx_for_group_A=zz>bias_threshold_A
                    idx_for_group_B=~idx_for_group_A

                    ### remove states already exist in the selected fate cluster 
                    if avoid_target_states:
                        for zz in valid_fate_list[0]:
                            id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                            idx_for_group_A[id_A_t1]=False


                else:
                    zz=relative_bias[:,0]
                    idx_for_group_A=zz>bias_threshold_A
                    kk=relative_bias[:,1]
                    idx_for_group_B=kk>bias_threshold_B

                    ### remove states already exist in the selected fate cluster 
                    if avoid_target_states:
                        for zz in valid_fate_list[0]:
                            id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                            idx_for_group_A[id_A_t1]=False

                        for zz in valid_fate_list[1]:
                            id_B_t1=np.nonzero(state_annote_t1==zz)[0]
                            idx_for_group_B[id_B_t1]=False

                
                group_A_idx_full=np.zeros(adata.shape[0],dtype=bool)
                group_A_idx_full[cell_id_t1[sp_idx]]=idx_for_group_A[sp_idx]
                group_B_idx_full=np.zeros(adata.shape[0],dtype=bool)
                group_B_idx_full[cell_id_t1[sp_idx]]=idx_for_group_B[sp_idx]
                adata.obs['cell_group_A']=group_A_idx_full
                adata.obs['cell_group_B']=group_B_idx_full             


                if plot_ancestor:
                    x_emb=adata.obsm['X_emb'][:,0]
                    y_emb=adata.obsm['X_emb'][:,1]
                    state_annote=adata.obs['state_info']

                    fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=8)
                    ax = plt.subplot(nrow, ncol, 1)
                    customized_embedding(x_emb,y_emb,group_A_idx_full,ax=ax,point_size=point_size)
                    if plot_target_state:
                        for zz in valid_fate_list[0]:
                            idx_2=state_annote==zz
                            ax.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=alpha_target)
                    ax.set_title(f'Group A')
                    ax.axis('off')
                    
                    ax = plt.subplot(nrow, ncol, 2)
                    customized_embedding(x_emb,y_emb,group_B_idx_full,ax=ax,point_size=point_size)
                    if plot_target_state:
                        for zz in valid_fate_list[1]:
                            idx_2=state_annote==zz
                            ax.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=alpha_target)
                    ax.set_title(f'Group B')
                    ax.axis('off')
                    
                    plt.tight_layout()
                    if savefig:
                        fig.savefig(f'{figure_path}/ancestor_state_groups.{settings.file_format_figs}')

                #diff_gene_A,diff_gene_B=differential_genes(adata,plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig,point_size=point_size)
        
                if 'dynamic_trajectory' not in adata.uns.keys():
                    adata.uns['dynamic_trajectory']={}

                # store the trajectory
                temp_list=[group_A_idx_full,group_B_idx_full]
                for j, selected_fate in enumerate(selected_fates):
                    fate_name,selected_idx=flexible_selecting_cells(adata,selected_fate)
                    combined_prob=temp_list[j].astype(int)+selected_idx.astype(int)
                    
                    if map_backwards:
                        adata.uns['dynamic_trajectory'][fate_name]={'map_backwards':combined_prob} # include both the targeted fate cluster and the inferred earlier states
                    else:
                        adata.uns['dynamic_trajectory'][fate_name]={'map_forward':combined_prob} 




def dynamic_trajectory_from_competition_bias(adata,selected_fates=[],used_map_name='transition_map',
    map_backwards=True,plot_time_points=[],bias_threshold=0.1,sum_fate_prob_thresh=0,
    avoid_target_states=False,plot_ancestor=True,point_size=2,savefig=False,
    plot_target_state=True,alpha_target=0.2):
    """
    Identify trajectory towards/from two given clusters.

    If `map_backwards=True`, use the fate probability towards given clusters (A,B)
    to define the ancestor population for A and B. The ancestor population + target fates 
    are the dynamic trajectory. 

    Fate bias is a `scalar` between (0,1) at each state, defined as competition between 
    two fate clusters, as in :func:`.fate_bias_from_binary_competition`. Selected ancestor population satisfies:

       * Prob(A)+Prob(B)>sum_fate_prob_thresh; 

       * for A: Bias>0.5+bias_threshold

       * for B: bias<0.5+bias_threshold


    If `map_backwards=False`, use the probability from given clusters (A,B)
    to define the fate population, and perform DGE analysis. This is not useful. 

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    used_map_name: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    bias_threshold: `float`, optional (default: 0), range: (0,0.5)
        Threshold for ancestor population selection. 
    sum_fate_prob_thresh: `float`, optional (default: 0), range: (0,1)
        Minimum cumulative probability towards joint cluster (A,B) 
        to qualify for ancestor selection.
    savefig: `bool`, optional (default: False)
        Save all plots.
    point_size: `int`, optional (default: 2)
        Size of the data point.
    avoid_target_states: `bool`, optional (default: False)
        If true, avoid selecting cells at the target cluster (A, or B) as 
        ancestor popolation.
    plot_target_state: `bool`, optional (default: True)
        Plot the target states in the state manifold. 
    alpha_target: `float`, optional (default: 0.2)
        Transparency parameter for plotting. 

    Returns
    -------
    Store the inferred ancestor states in adata.uns['cell_group_A'] and adata.uns['cell_group_B']

    Combine ancestor states and target states into adata.uns['dynamic_trajectory'] for each fate. 
    """

    diff_gene_A=[]
    diff_gene_B=[]
    hf.check_available_map(adata)
    #set_up_plotting()

    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")


    else:
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        figure_path=settings.figure_path
        state_annote_t1=np.array(adata.obs['state_info'][cell_id_t1])

        if (len(selected_fates)!=1) and (len(selected_fates)!=2):
            logg.error(f" Must provide one or two fates.")

        else:
            ## select time points
            time_info=np.array(adata.obs['time_info'])
            if (len(plot_time_points)>0):
                sp_idx=np.zeros(len(cell_id_t1),dtype=bool)
                for xx in plot_time_points:
                    sp_id_temp=np.nonzero(time_info[cell_id_t1]==xx)[0]
                    sp_idx[sp_id_temp]=True
            else:
                sp_idx=np.ones(len(cell_id_t1),dtype=bool)

            #if 'fate_map' not in adata.uns.keys():
            fate_map,mega_cluster_list,relative_bias,expected_prob,valid_fate_list=hf.compute_fate_map_and_intrinsic_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

            if '' not in mega_cluster_list:

                if len(selected_fates)==1:
                    idx_for_group_A=fate_map[:,0]>bias_threshold
                    idx_for_group_B=~idx_for_group_A
                    #valid_idx=np.ones(len(cell_id_t1),dtype=bool)

                    ### remove states already exist in the selected fate cluster 
                    if avoid_target_states:
                        for zz in valid_fate_list[0]:
                            id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                            idx_for_group_A[id_A_t1]=False


                else:
                    resol=10**(-10)
                    potential_vector_temp=fate_map+resol
                    diff=potential_vector_temp[:,0]#-potential_vector_temp[:,1]
                    tot=potential_vector_temp.sum(1)

                    valid_idx=tot>sum_fate_prob_thresh # default 0
                    valid_id=np.nonzero(valid_idx)[0]
                    vector_array=np.zeros(np.sum(valid_idx))
                    vector_array=diff[valid_idx]/(tot[valid_idx])

                    idx_for_group_A=np.zeros(len(tot),dtype=bool)
                    idx_for_group_B=np.zeros(len(tot),dtype=bool)
                    idx_for_group_A[valid_id]=vector_array>(0.5+bias_threshold)
                    idx_for_group_B[valid_id]=vector_array<(0.5-bias_threshold)


                    ### remove states already exist in the selected fate cluster 
                    if avoid_target_states:
                        for zz in valid_fate_list[0]:
                            id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                            idx_for_group_A[id_A_t1]=False

                        for zz in valid_fate_list[1]:
                            id_B_t1=np.nonzero(state_annote_t1==zz)[0]
                            idx_for_group_B[id_B_t1]=False


                group_A_idx_full=np.zeros(adata.shape[0],dtype=bool)
                group_A_idx_full[cell_id_t1[sp_idx]]=idx_for_group_A[sp_idx]
                group_B_idx_full=np.zeros(adata.shape[0],dtype=bool)
                group_B_idx_full[cell_id_t1[sp_idx]]=idx_for_group_B[sp_idx]
                adata.obs['cell_group_A']=group_A_idx_full
                adata.obs['cell_group_B']=group_B_idx_full                


                if plot_ancestor:
                    x_emb=adata.obsm['X_emb'][:,0]
                    y_emb=adata.obsm['X_emb'][:,1]
                    state_annote=adata.obs['state_info']

                    fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=8)
                    ax = plt.subplot(nrow, ncol, 1)
                    customized_embedding(x_emb,y_emb,group_A_idx_full,ax=ax,point_size=point_size)
                    if plot_target_state:
                        for zz in valid_fate_list[0]:
                            idx_2=state_annote==zz
                            ax.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=alpha_target)
                    ax.set_title(f'Group A')
                    ax.axis('off')
                    
                    ax = plt.subplot(nrow, ncol, 2)
                    customized_embedding(x_emb,y_emb,group_B_idx_full,ax=ax,point_size=point_size)
                    if plot_target_state:
                        for zz in valid_fate_list[1]:
                            idx_2=state_annote==zz
                            ax.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*1,alpha=alpha_target)
                    ax.set_title(f'Group B')
                    ax.axis('off')
                    
                    plt.tight_layout()
                    if savefig:
                        fig.savefig(f'{figure_path}/ancestor_state_groups.{settings.file_format_figs}')


                #diff_gene_A,diff_gene_B=differential_genes(adata,plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig,point_size=point_size)
        
                if 'dynamic_trajectory' not in adata.uns.keys():
                    adata.uns['dynamic_trajectory']={}

                # store the trajectory
                temp_list=[group_A_idx_full,group_B_idx_full]
                for j, selected_fate in enumerate(selected_fates):
                    fate_name,selected_idx=flexible_selecting_cells(adata,selected_fate)
                    combined_prob=temp_list[j].astype(int)+selected_idx.astype(int)

                    if map_backwards:
                        adata.uns['dynamic_trajectory'][fate_name]={'map_backwards':combined_prob} # include both the targeted fate cluster and the inferred earlier states
                    else:
                        adata.uns['dynamic_trajectory'][fate_name]={'map_forward':combined_prob} 


def differential_genes(adata,plot_groups=True,gene_N=100,plot_gene_N=5,
    savefig=False,point_size=1):
    """
    Perform differential gene expression analysis and plot top DGE genes.

    We use Wilcoxon rank-sum test to calculate P values, followed by
    Benjamini-Hochberg correction. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Need to contain gene expression matrix, and DGE cell groups A, B. 
    plot_groups: `bool`, optional (default: True)
        If true, plot the selected ancestor states for A, B
    gene_N: `int`, optional (default: 100)
        Number of top differentially expressed genes to selected.
    plot_gene_N: `int`, optional (default: 5)
        Number of top DGE genes to plot
    savefig: `bool`, optional (default: False)
        Save all plots.
    point_size: `int`, optional (default: 2)
        Size of the data point.

    Returns
    -------
    diff_gene_A: `pd.DataFrame`
        Genes differentially expressed in cell state group A, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    diff_gene_B: `pd.DataFrame`
        Genes differentially expressed in cell state group B, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    """

    diff_gene_A=[]
    diff_gene_B=[]
    idx_for_group_A=adata.obs['cell_group_A']
    idx_for_group_B=adata.obs['cell_group_B']
    #hf.check_available_map(adata)
    #set_up_plotting()
    if (np.sum(idx_for_group_A)==0) or (np.sum(idx_for_group_B)==0):
        logg.error("Group A or B has zero selected cell states. Could be that the cluser name is wrong; Or, the selection is too stringent. Consider use a smaller 'bias_threshold'")

    else:

        dge=hf.get_dge_SW(adata,idx_for_group_B,idx_for_group_A)

        dge=dge.sort_values(by='ratio',ascending=True)
        diff_gene_A=dge[:gene_N]
        #diff_gene_A=diff_gene_A_0[dge[:gene_N]['pv']<0.05]

        dge=dge.sort_values(by='ratio',ascending=False)
        diff_gene_B=dge[:gene_N]
        #diff_gene_B=diff_gene_B_0[dge[:gene_N]['pv']<0.05]

        x_emb=adata.obsm['X_emb'][:,0]
        y_emb=adata.obsm['X_emb'][:,1]
        figure_path=settings.figure_path
        
        if plot_groups:

            fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=8)
            ax = plt.subplot(nrow, ncol, 1)
            customized_embedding(x_emb,y_emb,idx_for_group_A,ax=ax,point_size=point_size)
            ax.set_title(f'Group A')
            ax.axis('off')
            ax = plt.subplot(nrow, ncol, 2)
            customized_embedding(x_emb,y_emb,idx_for_group_B,ax=ax,point_size=point_size)
            ax.set_title(f'Group B')
            ax.axis('off')
            
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups.{settings.file_format_figs}')
            
        #logg.error("Plot differentially-expressed genes for group A")
        if plot_gene_N>0:
            #logg.error(f"Plot the top {plot_gene_N} genes that are differentially expressed on group A")
            fig,nrow,ncol = start_subplot_figure(plot_gene_N, row_height=2.5, n_columns=5, fig_width=16)
            for j in range(plot_gene_N):
                ax = plt.subplot(nrow, ncol, j+1)

                #pdb.set_trace()
                gene_name=np.array(diff_gene_A['gene'])[j]
                customized_embedding(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                ax.set_title(f'{gene_name}')
                ax.axis('off')
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups_A_genes.{settings.file_format_figs}')
            
            #logg.error("Plot differentially-expressed genes for group B")
            #logg.error(f"Plot the top {plot_gene_N} genes that are differentially expressed on group B")
            fig,nrow,ncol = start_subplot_figure(plot_gene_N, row_height=2.5, n_columns=5, fig_width=16)
            for j in range(plot_gene_N):
                ax = plt.subplot(nrow, ncol, j+1)
                gene_name=np.array(diff_gene_B['gene'])[j]
                customized_embedding(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                ax.set_title(f'{gene_name}')
                ax.axis('off')
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups_B_genes.{settings.file_format_figs}')
        
            # logg.error('--------------Differentially expressed genes for group A --------------')
            # logg.error(diff_gene_A)
            
            # logg.error('--------------Differentially expressed genes for group B --------------')
            # logg.error(diff_gene_B)
        
    return diff_gene_A,diff_gene_B



def differential_genes_for_given_fates(adata,selected_fates=[],plot_time_points=[],
    plot_groups=True,gene_N=100,plot_gene_N=5,savefig=False,point_size=1):
    """
    Find and plot DGE genes between different clusters

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    `selected_fates` could contain a nested list of clusters. If so, we 
    combine each sub list into a mega-fate cluster and combine the fate 
    map correspondingly. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fates: `list`
        List of cluster_ids consistent with adata.obs['state_info']
    plot_time_points: `list`, optional (default: all)
        A list of time points to further restrict the cell states to plot. 
        If map_backwards=True, plot initial states that are among these time points;
        else, plot later states that are among these time points.
    plot_groups: `bool`, optional (default: True)
        If true, plot the selected ancestor states for A, B
    gene_N: `int`, optional (default: 100)
        Number of top differentially expressed genes to selected.
    plot_gene_N: `int`, optional (default: 5)
        Number of top DGE genes to plot
    savefig: `bool`, optional (default: False)
        Save all plots.
    point_size: `int`, optional (default: 2)
        Size of the data point.

    Returns
    -------
    diff_gene_A: `pd.DataFrame`
        Genes differentially expressed in cell state group A, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    diff_gene_B: `pd.DataFrame`
        Genes differentially expressed in cell state group B, ranked
        by the ratio of mean expressions between 
        the two groups, with the top being more differentially expressed.
    """

    diff_gene_A=[]
    diff_gene_B=[]
    #hf.check_available_map(adata)
    #set_up_plotting()


    time_info=np.array(adata.obs['time_info'])

    if (len(plot_time_points)>0):
        sp_idx=np.zeros(adata.shape[0],dtype=bool)
        for xx in plot_time_points:
            sp_id_temp=np.nonzero(time_info==xx)[0]
            sp_idx[sp_id_temp]=True

        adata_1=adata[np.nonzero(sp_idx)[0]]
    else:
        sp_idx=np.ones(adata.shape[0],dtype=bool)
        adata_1=adata

    

    state_annot_0=np.array(adata_1.obs['state_info'])
    if (len(selected_fates)==0) or (len(selected_fates)>2):
        logg.error(f"there should be only one or two fate clusters")
    else:
        fate_array_flat=[] # a flatten list of cluster names
        valid_fate_list=[] # a list of cluster lists, each cluster list is a macro cluster
        mega_cluster_list=[] # a list of string description for the macro cluster
        
        xx=selected_fates[0]
        idx_for_group_A=np.zeros(adata_1.shape[0],dtype=bool)
        if type(xx) is list:
            for zz in xx:
                idx=(state_annot_0==zz)
                idx_for_group_A[idx]=True
        else:
            idx=(state_annot_0==xx)
            idx_for_group_A[idx]=True   
            
        ##################
        idx_for_group_B=np.zeros(adata_1.shape[0],dtype=bool)
        if len(selected_fates)==2:
            xx=selected_fates[1]
            if type(xx) is list:
                for zz in xx:
                    idx=(state_annot_0==zz)
                    idx_for_group_B[idx]=True
            else:
                idx=(state_annot_0==xx)
                idx_for_group_B[idx]=True   
        else:
            idx_for_group_B=~idx_for_group_A

 
        group_A_idx_full=np.zeros(adata.shape[0],dtype=bool)
        group_A_idx_full[np.nonzero(sp_idx)[0]]=idx_for_group_A
        group_B_idx_full=np.zeros(adata.shape[0],dtype=bool)
        group_B_idx_full[np.nonzero(sp_idx)[0]]=idx_for_group_B
        adata.obs['cell_group_A']=group_A_idx_full
        adata.obs['cell_group_B']=group_B_idx_full
        #adata.uns['DGE_analysis']=[adata_1,idx_for_group_A,idx_for_group_B]

        diff_gene_A,diff_gene_B=differential_genes(adata,plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig,point_size=point_size)
            
    return diff_gene_A,diff_gene_B


######################

## Dynamic trajectory

######################


def flexible_selecting_cells(adata,selected_fate):
    """
    Selecting cells based on fate cluster.

    We allow `selected_fate` to be a `list` or `str`.
    In the former case, we combine all the fates to be a macro-fate cluster.

    return the fate_cluster_name, and selected idx
    """

    state_annote_0=np.array(adata.obs['state_info'])
    selected_idx=np.zeros(adata.shape[0],dtype=bool)
    fate_name=''
    if type(selected_fate) is list:
        for xx0 in selected_fate:
            idx=np.nonzero(state_annote_0==xx0)[0]
            selected_idx[idx]=True
            fate_name=fate_name+str(xx0)+'*'
    else:
        idx=np.nonzero(state_annote_0==selected_fate)[0]
        selected_idx[idx]=True
        fate_name=selected_fate
    return fate_name,selected_idx
        

def dynamic_trajectory_via_iterative_mapping(adata,selected_fate,used_map_name='transition_map',
    map_backwards=True,map_threshold=0.1,plot_separately=False,
    apply_time_constaint=False,point_size=2,plot_color_bar=True):
    """
    Infer trajectory towards/from a cluster

    It only works for Tmap from multi-time clones, since we are
    iteratively map states forward/backward in time. 

    If map_backwards=True, infer the trajectory backwards in time. 
    Using inferred transitino map, the inference is applied recursively. 
    Start with the cell states for the selected fate, then use selected 
    map to infer the immediate ancestor states. Then, using these putative 
    ancestor state as the input, find the immediate ancestors for these 
    input states. This goes on until all time points are exhausted.

    If map_backwards=False, infer the trajectory forward in time. 
    
    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster. 
    used_map_name: `str` 
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    plot_separately: `bool`, optional (default: False)
        Plot the inferred trajecotry separately for each time point.
    map_threshold: `float`, optional (default: 0.1)
        Relative threshold in the range [0,1] for truncating the fate map 
        towards the cluster. Only states above the threshold will be selected.
    apply_time_constaint: `bool`, optional (default: False)
        If true, in each iteration of finding the immediate ancestor states, select cell states
        at the corresponding time point.  
    point_size: `int`, optional (default: 2)
        Size of the data point.
    plot_color_bar: `bool`, optional (default: True)

    Returns
    -------
    Results are stored at adata.uns['dynamic_trajectory'] 
    """        

    # We always use the probabilistic map, which is more realiable. Otherwise, the result is very sensitive to thresholding
    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    hf.check_available_map(adata)
    resol=10**(-10) # small number used as a pseudo count to avoid numerical divergence in dividing
    #set_up_plotting()

    state_annote_0=np.array(adata.obs['state_info'])
    if map_backwards:
        cell_id_t1=adata.uns['Tmap_cell_id_t1']
        cell_id_t2=adata.uns['Tmap_cell_id_t2']

    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']
        cell_id_t1=adata.uns['Tmap_cell_id_t2']

    time_info=np.array(adata.obs['time_info'])
    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    figure_path=settings.figure_path

#     if selected_fate not in list(state_annote_0):
#     logg.error(f"selected_fate not valid. It should be among {set(state_annote_0)}")


    if used_map_name not in adata.uns['available_map']:
        logg.error(f"used_map_name should be among {adata.uns['available_map']}")

    else:
        ##### we normalize the map in advance to avoid normalization later in mapout_trajectories
        used_map_0=adata.uns[used_map_name]
        resol=10**(-10)
        used_map_0=hf.sparse_rowwise_multiply(used_map_0,1/(resol+np.sum(used_map_0,1).A.flatten()))

        if map_backwards:
            used_map=used_map_0
        else:
            used_map=used_map_0.T

        fate_name,selected_idx=flexible_selecting_cells(adata,selected_fate)


        if np.sum(selected_idx)==0:
            logg.error("No states selected. Please make sure the selected_fate are among {set(state_annote_0)")
        else:

            sort_time_info=np.sort(list(set(time_info)))[::-1]


            prob_0r=selected_idx
            if apply_time_constaint:
                prob_0r=prob_0r*(time_info==sort_time_info[0])

            prob_0r_temp=prob_0r>0
            prob_0r_0=prob_0r_temp.copy()
            prob_array=[]

            #used_map=hf.sparse_column_multiply(used_map,1/(resol+used_map.sum(0)))
            for j,t_0 in enumerate(sort_time_info[1:]):
                prob_1r_full=np.zeros(len(x_emb))

                prob_1r_full[cell_id_t1]=hf.mapout_trajectories(used_map,prob_0r_temp,threshold=map_threshold,cell_id_t1=cell_id_t1,cell_id_t2=cell_id_t2)

                ## thresholding the result 
                prob_1r_full=prob_1r_full*(prob_1r_full>map_threshold*np.max(prob_1r_full))

                if apply_time_constaint:
                    prob_1r_full=prob_1r_full*(time_info==t_0)

                # rescale
                prob_1r_full=prob_1r_full/(np.max(prob_1r_full)+resol)

                prob_array.append(prob_1r_full)
                prob_0r_temp=prob_1r_full


            cumu_prob=np.array(prob_array).sum(0)
            ### plot the results
            if plot_separately:
                col=len(sort_time_info);
                row=1
                fig = plt.figure(figsize=(4 * col, 3.5 * row))
                ax0=plt.subplot(row,col,1)
                if apply_time_constaint:
                    customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial: t={sort_time_info[0]}");
                    for k,t_0 in enumerate(sort_time_info[1:]):
                        ax1=plt.subplot(row,col,2+k)
                        customized_embedding(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"t={t_0}")

                else:
                    customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial");
                    for k,t_0 in enumerate(sort_time_info[1:]):
                        ax1=plt.subplot(row,col,2+k)
                        customized_embedding(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"{k+1}-th mapping")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_separate_BW{map_backwards}.{settings.file_format_figs}')  
            else:

                col=2; row=1
                fig = plt.figure(figsize=(4 * col, 3.5 * row))
                ax0=plt.subplot(row,col,1)
                customized_embedding(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"Initial: t={sort_time_info[0]}");

                ax1=plt.subplot(row,col,2)
                customized_embedding(x_emb,y_emb,cumu_prob,ax=ax1,point_size=point_size,title=f"All time")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_allTime_BW{map_backwards}.{settings.file_format_figs}')

            if plot_color_bar:
                fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1,label='Fate Probability')

            if 'dynamic_trajectory' not in adata.uns.keys():
                adata.uns['dynamic_trajectory']={}

            combined_prob=cumu_prob+prob_0r
            if map_backwards:
                adata.uns['dynamic_trajectory'][fate_name]={'map_backwards':combined_prob} # include both the targeted fate cluster and the inferred earlier states
            else:
                adata.uns['dynamic_trajectory'][fate_name]={'map_forward':combined_prob} 


def gene_expression_dynamics(adata,selected_fate,gene_name_list,traj_threshold=0.1,
    map_backwards=True,invert_PseudoTime=False,include_target_states=False,
    compute_new=True,fig_width=3.5,gene_exp_percentile=99,n_neighbors=8,
    plot_raw_data=False,point_size=2,stat_smooth_method='loess'):
    """
    Plot gene trend along inferred dynamic trajectory

    We assume that the dynamic trajecotry at given specification is already
    computed via :func:`.dynamic_trajectory`. Using the states belong to the 
    trajectory, it computes the pseudotime for these states, plot gene trend
    along the pseudo time. This needs to be updated from existing packages. 

    Given the states, we first construct KNN graph, compute spectral embedding,
    take the first component as the pseudotime. For the gene trend, we provide
    gene expression value of each cell, re-weighted by its probability belonging
    to this trajectory, and also rescaled by the global background. Finally, we
    fit a curve to the data points. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fate: `str`, or `list`
        targeted cluster of the trajectory, as consistent with adata.obs['state_info']
        When it is a list, the listed clusters are combined into a single fate cluster. 
    gene_name_list: `list`
        List of genes to plot on the dynamic trajectory. 
    traj_threshold: `float`, optional (default: 0.1), range: (0,1)
        Relative threshold, used to thresholding the inferred dynamic trajecotry to select states. 
    map_backwards: `bool`, optional (default: True)
        If `map_backwards=True`, plot initial cell states (rows of Tmap, at t1);
        else, plot later cell states (columns of Tmap, at t2)
    invert_PseudoTime: `bool`, optional (default: False)
        If true, invert the pseudotime: 1-pseuotime. This is useful when the direction
        of pseudo time does not agree with intuition.
    include_target_states: `bool`, optional (default: False)
        If true, include the target states to the dynamic trajectory.
    compute_new: `bool`, optional (default: True)
        If true, compute everyting from stratch (as we save computed pseudotime)
    fig_width: `float`, optional (default: 3.5)
        Figure width. 
    gene_exp_percentile: `int`, optional (default: 99)
        Plot gene expression below this percentile.
    n_neighbors: `int`, optional (default: 8)
        Number of nearest neighbors for constructing KNN graph.
    plot_raw_data: `bool`, optional (default: False)
        Plot the raw gene expression values of each cell along the pseudotime. 
    point_size: `int`, optional (default: 2)
        Size of the data point.
    stat_smooth_method: `str`, optional (default: 'loess')
        Smooth method used in the ggplot. Current availabel choices are:
        'auto' (Use loess if (n<1000), glm otherwise),
        'lm' (Linear Model),
        'wls' (Linear Model),
        'rlm' (Robust Linear Model),
        'glm' (Generalized linear Model),
        'gls' (Generalized Least Squares),
        'lowess' (Locally Weighted Regression (simple)),
        'loess' (Locally Weighted Regression)
        'mavg' (Moving Average)
        'gpr' (Gaussian Process Regressor)}

    Returns
    -------
    An adata object with only selected cell states. It can be used for dynamic inference with other packages. 
    """
    
    
    #final_id,temp_idx=state_info_SW(adata,sel_cell_id,x_emb,y_emb,which_branch,add_progenitors=True)
        

    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    state_annote_0=np.array(adata.obs['state_info'])
    time_info=np.array(adata.obs['time_info'])


    if map_backwards:
        cell_id_t1=adata.uns['Tmap_cell_id_t1']
        cell_id_t2=adata.uns['Tmap_cell_id_t2']

    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']
        cell_id_t1=adata.uns['Tmap_cell_id_t2']

    time_index_t1=np.zeros(len(time_info),dtype=bool)
    time_index_t2=np.zeros(len(time_info),dtype=bool)
    time_index_t1[cell_id_t1]=True
    time_index_t2[cell_id_t2]=True
    
    target_idx=np.zeros(adata.shape[0],dtype=bool)
    fate_name=''
    if type(selected_fate) is list:
        for xx0 in selected_fate:
            idx=np.nonzero(state_annote_0==xx0)[0]
            target_idx[idx]=True
            fate_name=fate_name+str(xx0)+'*'
    else:
        idx=np.nonzero(state_annote_0==selected_fate)[0]
        target_idx[idx]=True
        fate_name=selected_fate
        
        
    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    data_path=settings.data_path
    figure_path=settings.figure_path
    file_name=f'{data_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}_{map_backwards}.npy'


    
    if ('dynamic_trajectory' not in adata.uns.keys()) or (fate_name not in adata.uns['dynamic_trajectory'].keys()):
        if ('dynamic_trajectory' not in adata.uns.keys()):
            logg.error(f"The target fate trajectory have not been inferred yet." 
                "Please run dynamic_trajectory inference first")
        else:
            logg.error(f"Out of range. Current available selected_fate={list(adata.uns['dynamic_trajectory'].keys())}")
            logg.error(f"The target fate trajectory have not been inferred yet." 
                "Please run dynamic_trajectory inference first")
        
        #dynamic_trajectory(adata,fate_name='',used_map_name='transition_map',plot_separately=True,apply_time_constaint=False,map_threshold=0.1,point_size=point_size)
        
    else:
        if map_backwards:
            if 'map_backwards' not in adata.uns['dynamic_trajectory'][fate_name].keys():
                logg.error(f"The prongenitor states have not been inferred yet for *map_backwards=True*. Please first run ---- dynamic_trajectory ----")
            else:
                prob_0=np.array(adata.uns['dynamic_trajectory'][fate_name]['map_backwards'])
        else:
            if 'map_forward' not in adata.uns['dynamic_trajectory'][fate_name].keys():
                logg.error(f"The prongenitor states have not been inferred yet for *map_backwards=False*. Please first run ---- dynamic_trajectory ----")
            else:
                prob_0=np.array(adata.uns['dynamic_trajectory'][fate_name]['map_forward'])
        
        if not include_target_states:
            sel_cell_idx=(prob_0>traj_threshold*np.max(prob_0)) & time_index_t1
        else:
            sel_cell_idx=prob_0>traj_threshold*np.max(prob_0)
            
        #logg.error(sel_cell_idx)
        sel_cell_id=np.nonzero(sel_cell_idx)[0]


        #file_name=f"data/Pseudotime_{which_branch}_t2.npy"
        if os.path.exists(file_name) and (not compute_new):
            logg.info("Load pre-computed pseudotime")
            PseudoTime=np.load(file_name)
        else:
            
            from sklearn import manifold
            data_matrix=adata.obsm['X_pca'][sel_cell_idx]
            method=manifold.SpectralEmbedding(n_components=1,n_neighbors=n_neighbors)
            PseudoTime = method.fit_transform(data_matrix)
            np.save(file_name,PseudoTime)
            #logg.info("Run time:",time.time()-t)


        PseudoTime=PseudoTime-np.min(PseudoTime)
        PseudoTime=(PseudoTime/np.max(PseudoTime)).flatten()
        
        ## re-order the pseudoTime such that the target fate has the pseudo time 1.
        if invert_PseudoTime:
            # target_fate_id=np.nonzero(target_idx)[0]
            # convert_fate_id=hf.converting_id_from_fullSpace_to_subSpace(target_fate_id,sel_cell_id)[0]
            #if np.mean(PseudoTime[convert_fate_id])<0.5: PseudoTime=1-PseudoTime
            PseudoTime=1-PseudoTime
        
        #pdb.set_trace()
        if np.sum((PseudoTime>0.25)& (PseudoTime<0.75))==0: # the cell states do not form a contiuum. Plot raw data instead
            logg.error("The selected cell states do not form a connected graph. Cannot form a continuum of pseudoTime. Only plot the raw data")
            plot_raw_data=True

        ## plot the pseudotime ordering
        fig = plt.figure(figsize=(12,4))
        ax=plt.subplot(1,2,1)
        customized_embedding(x_emb,y_emb,sel_cell_idx,ax=ax,title='Selected cells',point_size=point_size)
        ax1=plt.subplot(1,2,2)
        customized_embedding(x_emb[sel_cell_idx],y_emb[sel_cell_idx],PseudoTime,ax=ax1,title='Pseudo Time',point_size=point_size)
        #customized_embedding(x_emb[final_id],y_emb[final_id],PseudoTime,ax=ax1,title='Pseudo time')
        Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1,label='Pseudo time')
        fig.savefig(f'{figure_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}_{map_backwards}.{settings.file_format_figs}')

        temp_dict={'PseudoTime':PseudoTime}
        for gene_name in gene_name_list:
            yy_max=np.percentile(adata.obs_vector(gene_name),gene_exp_percentile) # global blackground
            yy=np.array(adata.obs_vector(gene_name)[sel_cell_idx])
            rescaled_yy=yy*prob_0[sel_cell_idx]/yy_max # rescaled by global background
            temp_dict[gene_name]=rescaled_yy
        
        
        data2=pd.DataFrame(temp_dict)
        data2_melt=pd.melt(data2,id_vars=['PseudoTime'],value_vars=gene_name_list)
        gplot=ggplot(data=data2_melt,mapping=aes(x="PseudoTime", y='value',color='variable')) + \
        (geom_point() if plot_raw_data else stat_smooth(method=stat_smooth_method)) +\
        theme_classic()+\
        labs(x="Pseudo time",
             y="Normalized gene expression",
              color="Gene name")
   
        gplot.save(f'{figure_path}/{data_des}_fate_trajectory_pseutoTime_gene_expression_{fate_name}_{map_backwards}.{settings.file_format_figs}',width=fig_width, height=fig_width*0.618)
        gplot.draw()

        return adata[sel_cell_idx]


##################

# Clone related #

##################


def clones_on_manifold(adata,selected_clone_list=[0],point_size=1,
    color_list=['red','blue','purple','green','cyan','black'],time_points=[]):
    """
    Plot clones on top of state embedding.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_clone_list: `list`
        List of selected clone ID's.
    point_size: `int`, optional (default: 1)
        Size of the data point.
    color_list: `list`, optional (default: ['red','blue','purple','green','cyan','black'])
        The list of color that defines color at respective time points. 
    time_points: `list`, optional (default: all)
        Select time points to show corresponding states. If set to be [], use all states. 
    """

    #set_up_plotting()
    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    #data_path=settings.data_path
    figure_path=settings.figure_path
    clone_annot=adata.obsm['X_clone']
    time_info=np.array(adata.obs['time_info'])
    if len(time_points)==0:
        time_points=np.sort(list(set(time_info)))

        # using all data
    for my_id in selected_clone_list:
        if my_id < clone_annot.shape[1]:
            fig = plt.figure(figsize=(4, 3))
            ax=plt.subplot(1,1,1)
            idx_t=np.zeros(len(time_info),dtype=bool)
            for j, xx in enumerate(time_points):
                idx_t0=time_info==time_points[j]
                idx_t=idx_t0 | idx_t
            
            customized_embedding(x_emb[idx_t],y_emb[idx_t],np.zeros(len(y_emb[idx_t])),ax=ax,point_size=point_size)
            for j, xx in enumerate(time_points):
                idx_t=time_info==time_points[j]
                idx_clone=clone_annot[:,my_id].A.flatten()>0
                idx=idx_t & idx_clone
                ax.plot(x_emb[idx],y_emb[idx],'.',color=color_list[j%len(color_list)],markersize=12*point_size,markeredgecolor='white',markeredgewidth=point_size)


            fig.savefig(f'{figure_path}/{data_des}_different_clones_{my_id}.{settings.file_format_figs}')
        else:
            logg.error(f"No such clone id: {my_id}")


def clonal_fate_bias(adata,selected_fate='',clone_size_thresh=3,
    N_resampling=1000,compute_new=True):
    """
    Plot clonal fate bias towards a cluster.

    This is just -log(P_value), where P_value is for the observation 
    cell fraction of a clone in the targeted cluster as compared to 
    randomized clones, where the randomized sampling produces clones 
    of the same size as the targeted clone. The computed results will 
    be saved at path defined in adata.uns['data_path'].

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    selected_fate: `str`
        The targeted fate cluster, from adata.obs['state_info'].
    clone_size_thresh: `int`, optional (default: 3)
        Clones with size >= this threshold will be highlighted in 
        the plot in red.
    N_resampling: `int`, optional (default: 1000)
        Number of randomized sampling for asseesing the Pvalue of a clone. 
    compute_new: `bool`, optional (default: True)
        Compute from stratch, regardless of existing saved files. 

    Returns
    -------
    fate_bias: `np.array`
        Computed clonal fate bias.
    sort_idx: `np.array`
        Corresponding clone id list. 
    """

    #set_up_plotting()
    state_annote_0=adata.obs['state_info']
    data_des=adata.uns['data_des'][-1]
    clonal_matrix=adata.obsm['X_clone']
    data_path=settings.data_path
    data_des=adata.uns['data_des'][-1]
    figure_path=settings.figure_path
    state_list=list(set(state_annote_0))


    valid_clone_idx=(clonal_matrix.sum(0).A.flatten()>=2)
    valid_clone_id=np.nonzero(valid_clone_idx)[0]
    sel_cell_idx=(clonal_matrix[:,valid_clone_idx].sum(1).A>0).flatten()
    sel_cell_id=np.nonzero(sel_cell_idx)[0]
    clone_N=np.sum(valid_clone_idx)
    cell_N=len(sel_cell_id)

    ## annotate the state_annot
    clonal_matrix_new=clonal_matrix[sel_cell_id][:,valid_clone_id]
    state_annote_new=state_annote_0[sel_cell_id]
    hit_target=np.zeros(len(sel_cell_id),dtype=bool)

    new_sel_cluster=''
    if type(selected_fate) is list:
        for zz in selected_fate:
            if zz in state_list:
                idx=(state_annote_new==zz)
                hit_target[idx]=True
                new_sel_cluster=new_sel_cluster+zz
            else:
                logg.error(f'{zz} not in valid state annotation list: {state_list}')
    else:
        if selected_fate in state_list:
            idx=(state_annote_new==selected_fate)
            hit_target[idx]=True   
            new_sel_cluster=selected_fate
        else:
            logg.error(f'{selected_fate} not in valid state annotation list: {state_list}')

    if new_sel_cluster=='':
        return 



    file_name=f'{data_path}/{data_des}_clonal_fate_bias_{N_resampling}_{new_sel_cluster}.npz'

    if (not os.path.exists(file_name)) or compute_new:

        ## target clone
        target_ratio_array=np.zeros(clone_N)


        null_ratio_array=np.zeros((clone_N,N_resampling))
        P_value_up=np.zeros(clone_N)
        P_value_down=np.zeros(clone_N)
        P_value=np.zeros(clone_N)
        P_value_rsp=np.zeros((clone_N,N_resampling))

        for m in range(clone_N):
            if m%5==0:
                logg.info(f"Current clone id: {m}")
            target_cell_idx=(clonal_matrix_new[:,m].sum(1).A>0).flatten()
            target_clone_size=np.sum(target_cell_idx) 

            if target_clone_size>0:
                target_ratio=np.sum(hit_target[target_cell_idx])/target_clone_size
                target_ratio_array[m]=target_ratio
                #N_resampling=int(np.floor(cell_N/target_clone_size))


                sel_cell_id_copy=list(np.arange(cell_N))

                for j in range(N_resampling):
                    temp_id=np.random.choice(sel_cell_id_copy,size=target_clone_size,replace=False)
                    null_ratio_array[m,j]=np.sum(hit_target[temp_id])/target_clone_size


                ## reprogamming clone
                P_value_up[m]=np.sum(null_ratio_array[m]>=target_ratio)/N_resampling
                P_value_down[m]=np.sum(null_ratio_array[m]<=target_ratio)/N_resampling
                P_value[m]=np.min([P_value_up[m],P_value_down[m]])

                for j1,zz in enumerate(null_ratio_array[m]):
                    P_value_up_rsp=np.sum(null_ratio_array[m]>=zz)/N_resampling
                    P_value_down_rsp=np.sum(null_ratio_array[m]<=zz)/N_resampling
                    P_value_rsp[m,j1]=np.min([P_value_up_rsp,P_value_down_rsp])            


        np.savez(file_name,P_value_rsp=P_value_rsp,P_value=P_value)

    else:
        saved_data=np.load(file_name,allow_pickle=True)
        P_value_rsp=saved_data['P_value_rsp']
        P_value=saved_data['P_value']
        #target_ratio_array=saved_data['target_ratio_array']


    ####### Plotting
    clone_size_array=clonal_matrix_new.sum(0).A.flatten()

    resol=1/N_resampling
    P_value_rsp_new=P_value_rsp.reshape((clone_N*N_resampling,))
    sort_idx=np.argsort(P_value_rsp_new)
    P_value_rsp_new=P_value_rsp_new[sort_idx]+resol
    sel_idx=((np.arange(clone_N)+1)*len(sort_idx)/clone_N).astype(int)-1
    fate_bias_rsp=-np.log10(P_value_rsp_new[sel_idx])

    sort_idx=np.argsort(P_value)
    P_value=P_value[sort_idx]+resol
    fate_bias=-np.log10(P_value)
    idx=clone_size_array[sort_idx]>=clone_size_thresh


    fig=plt.figure(figsize=(4,3.5));ax=plt.subplot(1,1,1)
    ax.plot(np.arange(len(fate_bias))[~idx],fate_bias[~idx],'.',color='blue',markersize=5,label=f'Size <{int(clone_size_thresh)}')#,markeredgecolor='black',markeredgewidth=0.2)
    ax.plot(np.arange(len(fate_bias))[idx],fate_bias[idx],'.',color='red',markersize=5,label=f'Size >{int(clone_size_thresh-1)}')#,markeredgecolor='black',markeredgewidth=0.2)
    ax.plot(np.arange(len(fate_bias_rsp)),fate_bias_rsp,'.',color='grey',markersize=5,label='Randomized')#,markeredgecolor='black',markeredgewidth=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel('Clone rank')
    #plt.rc('text', usetex=True)
    #ax.set_ylabel('Fate bias ($-\\log_{10}P_{value}$)')
    ax.set_ylabel('Clonal fate bias')
    ax.legend()
    #ax.set_xlim([0,0.8])
    fig.tight_layout()
    fig.savefig(f'{figure_path}/{data_des}_clonal_fate_bias.{settings.file_format_figs}')
    #plt.show()


    target_fraction_array=(clonal_matrix_new.T*hit_target)/clone_size_array
    fig=plt.figure(figsize=(4,3.5));ax=plt.subplot(1,1,1)
    ax.hist(target_fraction_array,color='#2ca02c')
    ax.set_xlim([0,1])
    ax.set_xlabel('Clonal fraction in selected fates')
    ax.set_ylabel('Histogram')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f'Average: {int(np.mean(target_fraction_array)*100)/100};   Expect: {int(np.mean(hit_target)*100)/100}',fontsize=14)
    fig.savefig(f'{figure_path}/{data_des}_observed_clonal_fraction.{settings.file_format_figs}')

    return fate_bias,sort_idx



def heatmap(figure_path, X, variable_names,color_bar_label='cov',data_des='',color_bar=True):
    """
    Plot heat map of a two-dimensional matrix X.

    Parameters
    ----------
    figure_path: `str`
        path to save figures
    X: `np.array`
        The two-dimensional matrix to plot
    variable_names: `list`
        List of variable names
    color_bar_label: `str`, optional (default: 'cov')
        Color bar label
    data_des: `str`, optional (default: '')
        String to distinguish different saved objects.
    color_bar: `bool`, optional (default: True)
        If true, plot the color bar.
    """

    vmax = (np.percentile(X-np.diag(np.diag(X)),95) + np.percentile(X-np.diag(np.diag(X)),98))/2
    #vmax=np.max(X)
    plt.imshow(X, vmax=vmax)
    plt.xticks(np.arange(X.shape[0])+.4, variable_names, ha='right', rotation=60, fontsize=16);
    plt.yticks(np.arange(X.shape[0])+.4, variable_names, fontsize=16);
    if color_bar:
        cbar = plt.colorbar()
        cbar.set_label(color_bar_label, rotation=270, fontsize=16, labelpad=20)
        plt.gcf().set_size_inches((6,4))
    else:
        plt.gcf().set_size_inches((4,4))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_heat_map.{settings.file_format_figs}')



def ordered_heatmap(figure_path, data_matrix, variable_names,int_seed=10,
    data_des='',log_transform=False):
    """
    Plot ordered heat map of data_matrix matrix.

    Parameters
    ----------
    figure_path: `str`
        path to save figures
    data_matrix: `np.array`
        A matrix whose columns should match variable_names 
    variable_names: `list`
        List of variable names
    color_bar_label: `str`, optional (default: 'cov')
        Color bar label
    data_des: `str`, optional (default: '')
        String to distinguish different saved objects.
    int_seed: `int`, optional (default: 10)
        Seed to initialize the plt.figure object (to avoid 
        plotting on existing object).
    log_transform: `bool`, optional (default: False)
        If true, perform a log transform. This is needed when the data 
        matrix has entries varying by several order of magnitude. 
    """

    o = hf.get_hierch_order(data_matrix)
    #fig=plt.figure(figsize=(4,6)); ax=plt.subplot(1,1,1)
    plt.figure(int_seed)
    
    if log_transform:
        plt.imshow(data_matrix[o,:], aspect='auto',cmap=plt.cm.Reds, vmax=1)
    else:
        plt.imshow(np.log(data_matrix[o,:]+1)/np.log(10), aspect='auto',cmap=plt.cm.Reds, vmin=0,vmax=1)
            
    plt.xticks(np.arange(data_matrix.shape[1])+.4, variable_names, rotation=70, ha='right', fontsize=12)
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('Number of barcodes (log10)', rotation=270, fontsize=12, labelpad=20)
    plt.gcf().set_size_inches((4,6))
    plt.tight_layout()
    plt.savefig(figure_path+f'/{data_des}_data_matrix.{settings.file_format_figs}')



def barcode_heatmap(adata,plot_time_point,selected_fates=[],color_bar=True):
    """
    Plot barcode heatmap among different fate clusters.

    We select one time point with clonal measurement, and show the 
    heatmap of barcodes among selected fate clusters. 

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    plot_time_point: `str`
        Time point to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    """

    #hf.check_available_map(adata)
    #set_up_plotting()

    time_info=np.array(adata.obs['time_info'])
    sp_id=np.nonzero(time_info==plot_time_point)[0]

    clone_annot=adata[sp_id].obsm['X_clone']
    state_annote=adata[sp_id].obs['state_info']


    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    data_des=f'{data_des}_clonal'
    figure_path=settings.figure_path


    if len(selected_fates)==0:
        selected_fates=list(set(adata.obs['state_info']))

    #selected_fates=np.sort(list(set(cell_type_info_sort)))
    coarse_clone_annot=np.zeros((len(selected_fates),clone_annot.shape[1]))
    for j, cell_type in enumerate(selected_fates):
        idx=state_annote==cell_type
        coarse_clone_annot[j,:]=clone_annot[idx].sum(0)

    ordered_heatmap(figure_path, coarse_clone_annot.T, selected_fates,data_des=data_des)




def fate_coupling_from_clones(adata,plot_time_point,selected_fates=[],color_bar=True,rename_selected_fates=[]):
    """
    Plot fate coupling determined by just clones.

    We select one time point with clonal measurement, and show the normalized 
    clonal correlation among these fates.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
    plot_time_point: `str`
        Time point to select the cell states.
    selected_fates: `list`, optional (default: all)
        List of fate clusters to use. If set to be [], use all.
    color_bar: `bool`, optional (default: True)
        Plot color bar. 
    """

    #hf.check_available_map(adata)
    #set_up_plotting()


    time_info=np.array(adata.obs['time_info'])
    sp_id=np.nonzero(time_info==plot_time_point)[0]

    clone_annot=adata[sp_id].obsm['X_clone']
    state_annote=adata[sp_id].obs['state_info']


    x_emb=adata.obsm['X_emb'][:,0]
    y_emb=adata.obsm['X_emb'][:,1]
    data_des=adata.uns['data_des'][-1]
    data_des=f'{data_des}_clonal_fate_coupling'
    figure_path=settings.figure_path

    if len(selected_fates)==0:
        selected_fates=list(set(adata.obs['state_info']))


    #selected_fates=np.sort(list(set(cell_type_info_sort)))
    coarse_clone_annot=np.zeros((len(selected_fates),clone_annot.shape[1]))
    for j, cell_type in enumerate(selected_fates):
        idx=state_annote==cell_type
        coarse_clone_annot[j,:]=clone_annot[idx].sum(0)

    if len(rename_selected_fates)!=len(selected_fates):
        rename_selected_fates=selected_fates

    X_weinreb = hf.get_normalized_covariance(coarse_clone_annot.T,method='Weinreb')
    heatmap(figure_path, X_weinreb, rename_selected_fates,color_bar_label='Coupling',color_bar=color_bar,data_des=data_des)



