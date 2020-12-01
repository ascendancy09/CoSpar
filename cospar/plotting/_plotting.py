import numpy as np
# import scipy.special
# import scipy.stats as scs
# import scipy.linalg as scl
# import sklearn
# import ot.bregman as otb
# from ot.utils import dist
import time
from plotnine import *  
from sklearn import manifold
import pandas as pd
import scanpy as sc
import pdb
import os
import scipy.sparse as ssp
import cospar.help_functions as hf
# import os
from matplotlib import pyplot as plt
#import CoSpar.tmap as tmap

# import sys

def darken_cmap(cmap, scale_factor):
    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii, 0] = curcol[0] * scale_factor
        cdat[ii, 1] = curcol[1] * scale_factor
        cdat[ii, 2] = curcol[2] * scale_factor
        cdat[ii, 3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap

def start_subplot_figure(n_subplots, n_columns=5, fig_width=14, row_height=3, dpi=75):
    n_rows = int(np.ceil(n_subplots / float(n_columns)))
    fig = plt.figure(figsize = (fig_width, n_rows * row_height), dpi=dpi)
    return fig, n_rows, n_columns

def plot_one_gene(E, gene_list, gene_to_plot, x, y, normalize=False, ax=None, order_points=True, col_range=(0,100), buffer_pct=0.03, point_size=1, color_map=None, smooth_operator=None):
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



def plot_one_gene_SW(x, y, vector, normalize=False, title=None, ax=None, order_points=True, set_ticks=0, col_range=(0, 100), buffer_pct=0.03, point_size=1, color_map=None, smooth_operator=None,savefig=False,dpi=300,set_lim=True,vmax=np.nan,vmin=np.nan,color_bar=False):
    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds, .9)
    if ax is None:
        fig, ax = plt.subplots()

    # if normalize:
    #    E = tot_counts_norm(E, target_mean=1e6)[0]

    #k = list(gene_list).index(gene_to_plot)
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

    if savefig:
        fig.savefig(f'figure/plot_one_gene_SW_fig_{int(np.round(np.random.rand()*100))}.png',dpi=dpi)


def check_transition_map(adata):
    available_map=[]
    for xx in adata.uns.keys():
        if 'transition_map' in xx:
            available_map.append(xx)
    adata.uns['available_map']=available_map


def plot_fate_map(adata,selected_fates=[],used_map_name='transition_map',map_backwards=True,plot_time_points=[],plot_background=True,plot_target_state=True,normalize=False,
               auto_color_scale=False,plot_color_bar=True,point_size=2,plot_fate_bias=False):

    '''
        input:
            plot_target_state: plot the selected fate clusters 
            used_map_name: transition_map, demultiplexed_map, OT_transition_map, or HighVar_transition_map
            normalize: True, normalize the map to enhance the fate choice difference among selected clusters
            auto_color_scale: True, auto_scale the color range to match the minimum and maximum of the predicted fate probability; False, set the range to be [0,1]
            plot_fate_bias: plot the extent of fate bias as compared with null hypothesis: random transitions
                            bias adjusted by the relative size of the target fate cluster at the second time point

        return: the predicted fate map at selected time points

    '''

    check_transition_map(adata)

    if used_map_name not in adata.uns['available_map']:
        print(f"Error, used_map_name should be among {adata.uns['available_map']}")

    else:        
        state_annote=adata.obs['state_annotation']
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


        x_emb=adata.obsm['X_umap'][:,0]
        y_emb=adata.obsm['X_umap'][:,1]
        data_des=adata.uns['data_des'][0]
        figure_path=adata.uns['figure_path'][0]



        fate_map,fate_list_descrip,extent_of_bias,expected_bias,fate_list_array=hf.compute_fate_map_and_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)

        # normalize the map to enhance the fate choice difference among selected clusters
        if normalize and (fate_map.shape[1]>1):
            resol=10**-10 
            fate_map=hf.sparse_rowwise_multiply(fate_map,1/(resol+np.sum(fate_map,1)))
            #fate_entropy_temp=fate_entropy_array[x0]


        ################### plot fate probability
        vector_array=[vector for vector in list(fate_map.T)]
        description=[fate for fate in fate_list_descrip]
        row =1; col =len(vector_array); 
        fig = plt.figure(figsize=(4.5 * col, 3.5 * row))
        for j in range(len(vector_array)):
            ax0 = plt.subplot(row, col, j + 1)
            
            if plot_background:
                plot_one_gene_SW(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0,title=description[j])            
                if plot_target_state:
                    for zz in fate_list_array[j]:
                        idx_2=state_annote==zz
                        ax0.plot(x_emb[idx_2],y_emb[idx_2],'.',color='cyan',markersize=point_size*2,alpha=0.2)
            else:
                plot_one_gene_SW(x_emb[cell_id_t1],y_emb[cell_id_t1],np.zeros(len(y_emb[cell_id_t1])),point_size=point_size,ax=ax0,title=description[j])

            if auto_color_scale:
                plot_one_gene_SW(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],point_size=point_size,ax=ax0,title=description[j],set_lim=False)
            else:
                plot_one_gene_SW(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],vector_array[j][sp_idx],point_size=point_size,ax=ax0,title=description[j],set_lim=False,vmax=1,vmin=0)
        
        if plot_color_bar:
            fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Fate probability')
      

        fig.savefig(f'{figure_path}/{data_des}_fate_map_overview.png',transparent=True,dpi=300)


        if plot_fate_bias:
            ################# plot the extent of fate bias as compared with null hypothesis: random transitions
            vector_array=[vector for vector in list(extent_of_bias.T)]
            description=[fate for fate in fate_list_descrip]

            row =1;  col = len(vector_array); 
            fig = plt.figure(figsize=(4.5 * col, 3.5 * row))
            for j in range(len(vector_array)):
                ax0 = plt.subplot(row, col, j + 1)
                # plot_one_gene_SW(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax0,title=description[j])
                # if plot_target_state:
                #     for zz in fate_list_array[j]:
                #         idx_2=state_annote==zz
                #         ax0.plot(x_emb[idx_2],y_emb[idx_2],'.',color='darkorange',markersize=point_size*2,alpha=0.2)
                temp_array=vector_array[j][sp_idx]
                new_idx=np.argsort(abs(temp_array-0.5))
                if auto_color_scale:
                    plot_one_gene_SW(x_emb[cell_id_t1][sp_idx][new_idx],y_emb[cell_id_t1][sp_idx][new_idx],temp_array[new_idx],point_size=point_size,ax=ax0,title=description[j],color_map=plt.cm.bwr,set_lim=False)
                else:
                    plot_one_gene_SW(x_emb[cell_id_t1][sp_idx][new_idx],y_emb[cell_id_t1][sp_idx][new_idx],temp_array[new_idx],point_size=point_size,ax=ax0,title=description[j],color_map=plt.cm.bwr,set_lim=False,vmax=1,vmin=0)
            
            if plot_color_bar:
                Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), ax=ax0,label='Actual/expected bias')
                Clb.set_ticks([])
                #Clb.ax.set_title(f'description[j]')

            fig.savefig(f'{figure_path}/{data_des}_fate_map_overview_extent_of_bias.png',transparent=True,dpi=300)


            # return fate map robability, and fate bias at selected time points
            #return fate_map[sp_idx,:],extent_of_bias[sp_idx,:],expected_bias
            adata.uns['fate_map_output']={"fate_map":fate_map[sp_idx,:],"extent_of_bias":extent_of_bias[sp_idx,:],"expected_bias":expected_bias}




def plot_single_cell_transition_probability(adata,selected_state_id_list=[0],used_map_name='transition_map',map_backwards=True,savefig=False,point_size=3):
    '''
            prediction_array is a list with prediction matrix from different methods
            method_descrip is an array of description for each method

            map_backwards: if true, the selected_state_id_list will be the column index; otherwise, it is row index
            used_map_name: transition_map, demultiplexed_map, OT_transition_map, or HighVar_transition_map

    '''
    check_transition_map(adata)
    state_annote=adata.obs['state_annotation']
    cell_id_t2=adata.uns['Tmap_cell_id_t2']
    cell_id_t1=adata.uns['Tmap_cell_id_t1']
    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    data_des=adata.uns['data_des'][0]
    figure_path=adata.uns['figure_path'][0]

    if used_map_name not in adata.uns['available_map']:
        print(f"Error, used_map_name should be among {adata.uns['available_map']}")

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
                    plot_one_gene_SW(x_emb, y_emb, prob_vec, point_size=point_size, ax=ax0)
                    
                    ax0.plot(x_emb[cell_id_t1][target_cell_ID],y_emb[cell_id_t1][target_cell_ID],'*b',markersize=3*point_size)
                    if disp_name:
                        ax0.set_title(f"t1 state (blue star) ({cell_id_t1[target_cell_ID]})")
                    else:
                        ax0.set_title(f"ID: {cell_id_t1[target_cell_ID]}")

                else:
                    prob_vec=np.zeros(len(x_emb))
                    prob_vec[cell_id_t1]=matrix[:,target_cell_ID]
                    plot_one_gene_SW(x_emb, y_emb,prob_vec, point_size=point_size, ax=ax0)
                    
                    ax0.plot(x_emb[cell_id_t2][target_cell_ID],y_emb[cell_id_t2][target_cell_ID],'*b',markersize=3*point_size)
                    if disp_name:
                        ax0.set_title(f"t2 fate (blue star) ({cell_id_t2[target_cell_ID]})")
                    else:
                        ax0.set_title(f"ID: {cell_id_t2[target_cell_ID]}")


        Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax0,label='Probability')


        plt.tight_layout()
        if savefig:
            fig.savefig(f"{figure_path}/plotting_transition_map_probability_{map_backwards}.png",dpi=300)



def plot_binary_fate_choice(adata,selected_fates=[],used_map_name='transition_map',map_backwards=True,sum_fate_prob_thresh=0,plot_time_points=[],point_size=1,include_target_states=False):
    '''
        fate_map_list: a list of fate maps
        description_list: an list of description for each fate map
        used_map_name: transition_map, demultiplexed_map, OT_transition_map, or HighVar_transition_map
        sum_fate_prob_thresh: to show a state, it needs to have a total probability threshold into all states within the selected clusters.  
    '''
    check_transition_map(adata)
    if used_map_name not in adata.uns['available_map']:
        print(f"Error, used_map_name should be among {adata.uns['available_map']}")

    else:
        state_annote=adata.obs['state_annotation']
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        x_emb=adata.obsm['X_umap'][:,0]
        y_emb=adata.obsm['X_umap'][:,1]
        data_des=adata.uns['data_des'][0]
        figure_path=adata.uns['figure_path'][0]


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
            print("Error! Must have only two fates")
        else:
            fate_map,fate_list_descrip,extent_of_bias,expected_bias,fate_list_array=hf.compute_fate_map_and_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)


            resol=10**(-10)

            fig=plt.figure(figsize=(5,4))
            ax=plt.subplot(1,1,1)
            print(fate_map.shape)
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

            #plot_one_gene_SW(x_emb[cell_id_t1][sp_idx],y_emb[cell_id_t1][sp_idx],np.zeros(len(y_emb[cell_id_t1][sp_idx])),point_size=point_size,ax=ax)
            if include_target_states:
                plot_one_gene_SW(x_emb,y_emb,np.zeros(len(y_emb)),point_size=point_size,ax=ax)
     
                for zz in fate_list_array[0]:
                    idx_2=state_annote[cell_id_t2]==zz
                    ax.plot(x_emb[cell_id_t2[idx_2]],y_emb[cell_id_t2[idx_2]],'.',color='red',markersize=point_size*2,alpha=1)
                for zz in fate_list_array[1]:
                    idx_2=state_annote[cell_id_t2]==zz
                    ax.plot(x_emb[cell_id_t2[idx_2]],y_emb[cell_id_t2[idx_2]],'.',color='blue',markersize=point_size*2,alpha=1)

                    
            else:
                plot_one_gene_SW(x_emb[cell_id_t1_sp],y_emb[cell_id_t1_sp],np.zeros(len(y_emb[cell_id_t1_sp])),point_size=point_size,ax=ax)
            #plot_one_gene_SW(x_emb[cell_id_t2],y_emb[cell_id_t2],np.zeros(len(y_emb[cell_id_t2])),point_size=point_size,ax=ax)

            new_idx=np.argsort(abs(vector_array-0.5))
            plot_one_gene_SW(x_emb[cell_id_t1_sp][valid_idx][new_idx],y_emb[cell_id_t1_sp][valid_idx][new_idx],
                                vector_array[new_idx],vmax=1,vmin=0,
                                point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

    #         # remove un-wanted time points
    #         if len(cell_id_t1[~sp_idx])>0:
    #             plot_one_gene_SW(x_emb[cell_id_t1[~sp_idx]],y_emb[cell_id_t1[~sp_idx]],np.zeros(len(y_emb[cell_id_t1[~sp_idx]])),
    #                         point_size=point_size,set_lim=False,ax=ax,color_map=plt.cm.bwr,order_points=False)

            Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), ax=ax,label='Fate bias')
            Clb.ax.set_title(f'{fate_list_descrip[0]}')

            fig.savefig(f'{figure_path}/{data_des}_binary_fate_bias_BW{map_backwards}.png',transparent=True,dpi=300)




def plot_driver_genes(adata,selected_fates=[],used_map_name='transition_map',map_backwards=True,plot_time_points=[],bias_threshold=0.2,plot_groups=True,gene_N=100,plot_gene_N=5,savefig=False,point_size=1,avoid_target_states=True):
    '''
        If only one fate is provided, find genes highly expressed in the ancestor states of this fate cluster, and genes enriched in the remaining states. 

        If two fates provided (selected_fate=['A','B']), find genes highly enriched in ancester states for A, and enriched in ancestor states for B. 

        map_threshld: in the range [0,1], for the actual-over-expected bias. 0 selects any state where the actual bias is larger than the expected bias; 1 means that the commitment is 100%
        
        avoid_target_states: if the ancestor states has the same cell state annotations, remove these states
        
    '''
    diff_gene_A=[]
    diff_gene_B=[]
    check_transition_map(adata)
    

    if used_map_name not in adata.uns['available_map']:
        print(f"Error, used_map_name should be among {adata.uns['available_map']}")


    else:
        if map_backwards:
            cell_id_t1=adata.uns['Tmap_cell_id_t1']
            cell_id_t2=adata.uns['Tmap_cell_id_t2']

        else:
            cell_id_t2=adata.uns['Tmap_cell_id_t1']
            cell_id_t1=adata.uns['Tmap_cell_id_t2']

        figure_path=adata.uns['figure_path'][0]
        state_annote_t1=np.array(adata.obs['state_annotation'][cell_id_t1])

        if (len(selected_fates)!=1) and (len(selected_fates)!=2):
            print("Error! Must provide one or two fates.")

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
            fate_map,fate_list_descrip,extent_of_bias,expected_bias,fate_list_array=hf.compute_fate_map_and_bias(adata,selected_fates=selected_fates,used_map_name=used_map_name,map_backwards=map_backwards)


            if len(selected_fates)==1:
                zz=2*extent_of_bias[:,0]-1
                idx_for_group_A=zz>bias_threshold
                idx_for_group_B=~idx_for_group_A
              
                ### remove states already exist in the selected fate cluster 
                if avoid_target_states:
                    for zz in fate_list_array[0]:
                        id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                        idx_for_group_A[id_A_t1]=False
              
              
            else:
                zz=2*extent_of_bias[:,0]-1
                idx_for_group_A=zz>bias_threshold
                kk=2*extent_of_bias[:,1]-1
                idx_for_group_B=kk>bias_threshold
              
                ### remove states already exist in the selected fate cluster 
                if avoid_target_states:
                    for zz in fate_list_array[0]:
                        id_A_t1=np.nonzero(state_annote_t1==zz)[0]
                        idx_for_group_A[id_A_t1]=False

                    for zz in fate_list_array[1]:
                        id_B_t1=np.nonzero(state_annote_t1==zz)[0]
                        idx_for_group_B[id_B_t1]=False

            
            

            group_A_idx_full=np.zeros(adata.shape[0],dtype=bool)
            group_A_idx_full[cell_id_t1]=idx_for_group_A
            group_B_idx_full=np.zeros(adata.shape[0],dtype=bool)
            group_B_idx_full[cell_id_t1]=idx_for_group_B
            adata.obs['DGE_cell_group_A']=group_A_idx_full
            adata.obs['DGE_cell_group_B']=group_B_idx_full

            diff_gene_A,diff_gene_B=plot_differential_genes(adata[cell_id_t1[sp_idx]],idx_for_group_A[sp_idx],idx_for_group_B[sp_idx],plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig,point_size=point_size)
    
    return diff_gene_A,diff_gene_B


def plot_differential_genes(adata,idx_for_group_A,idx_for_group_B,plot_groups=True,gene_N=100,plot_gene_N=5,savefig=False,point_size=1):
    '''

    '''
    diff_gene_A=[]
    diff_gene_B=[]
    check_transition_map(adata)
    if (np.sum(idx_for_group_A)==0) or (np.sum(idx_for_group_B)==0):
        print("Group A or B has zero selected cell states. Could be that the cluser name is wrong; Or, the selection is too stringent. Consider use a smaller 'bias_threshold'")

    else:

        dge=hf.get_dge_SW(adata,idx_for_group_B,idx_for_group_A)

        dge=dge.sort_values(by='ratio',ascending=True)
        diff_gene_A=dge[:gene_N]
        #diff_gene_A=diff_gene_A_0[dge[:gene_N]['pv']<0.05]

        dge=dge.sort_values(by='ratio',ascending=False)
        diff_gene_B=dge[:gene_N]
        #diff_gene_B=diff_gene_B_0[dge[:gene_N]['pv']<0.05]

        x_emb=adata.obsm['X_umap'][:,0]
        y_emb=adata.obsm['X_umap'][:,1]
        figure_path=adata.uns['figure_path'][0]
        
        if plot_groups:

            fig,nrow,ncol = start_subplot_figure(2, row_height=4, n_columns=2, fig_width=16, dpi=75)
            ax = plt.subplot(nrow, ncol, 1)
            plot_one_gene_SW(x_emb,y_emb,idx_for_group_A,ax=ax,point_size=point_size)
            ax.set_title(f'Group A')
            ax.axis('off')
            ax = plt.subplot(nrow, ncol, 2)
            plot_one_gene_SW(x_emb,y_emb,idx_for_group_B,ax=ax,point_size=point_size)
            ax.set_title(f'Group B')
            ax.axis('off')
            
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups.png',dpi=200)
            
        #print("Plot differentially-expressed genes for group A")
        if plot_gene_N>0:
            fig,nrow,ncol = start_subplot_figure(plot_gene_N, row_height=2.5, n_columns=5, fig_width=16, dpi=75)
            for j in range(plot_gene_N):
                ax = plt.subplot(nrow, ncol, j+1)

                #pdb.set_trace()
                gene_name=np.array(diff_gene_A['gene'])[j]
                plot_one_gene_SW(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                ax.set_title(f'{gene_name}')
                ax.axis('off')
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups_A_genes.png',dpi=200)
            
            #print("Plot differentially-expressed genes for group B")
            fig,nrow,ncol = start_subplot_figure(plot_gene_N, row_height=2.5, n_columns=5, fig_width=16, dpi=75)
            for j in range(plot_gene_N):
                ax = plt.subplot(nrow, ncol, j+1)
                gene_name=np.array(diff_gene_B['gene'])[j]
                plot_one_gene_SW(x_emb,y_emb,adata.obs_vector(gene_name),ax=ax,point_size=point_size)
                ax.set_title(f'{gene_name}')
                ax.axis('off')
            plt.tight_layout()
            if savefig:
                fig.savefig(f'{figure_path}/dge_analysis_groups_B_genes.png',dpi=200)
        
            print('--------------Differentially expressed genes for group A --------------')
            print(diff_gene_A)
            
            print('--------------Differentially expressed genes for group B --------------')
            print(diff_gene_B)
        
    return diff_gene_A,diff_gene_B



def plot_differential_genes_for_given_fates(adata,selected_fates=[],plot_groups=True,gene_N=100,plot_gene_N=5,savefig=False,point_size=1):
    '''
        selected_fates: should be two state clusters

    '''
    diff_gene_A=[]
    diff_gene_B=[]
    check_transition_map(adata)

    state_annot_0=np.array(adata.obs['state_annotation'])
    if (len(selected_fates)==0) or (len(selected_fates)>2):
        print("Error; there should be only one or two fate clusters")
    else:
        fate_array_flat=[] # a flatten list of cluster names
        fate_list_array=[] # a list of cluster lists, each cluster list is a macro cluster
        fate_list_descrip=[] # a list of string description for the macro cluster
        
        xx=selected_fates[0]
        idx_for_group_A=np.zeros(adata.shape[0],dtype=bool)
        if type(xx) is list:
            for zz in xx:
                idx=(state_annot_0==zz)
                idx_for_group_A[idx]=True
        else:
            idx=(state_annot_0==xx)
            idx_for_group_A[idx]=True   
            
        ##################
        idx_for_group_B=np.zeros(adata.shape[0],dtype=bool)
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

 
        adata.obs['DGE_cell_group_A']=idx_for_group_A
        adata.obs['DGE_cell_group_B']=idx_for_group_B

        diff_gene_A,diff_gene_B=plot_differential_genes(adata,idx_for_group_A,idx_for_group_B,plot_groups=plot_groups,gene_N=gene_N,plot_gene_N=plot_gene_N,savefig=savefig,point_size=point_size)
            
    return diff_gene_A,diff_gene_B



        
        
def plot_clones(adata,selected_clone_list=[0],point_size=1,color_list=['red','blue','purple','green','cyan','black']):

    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    #state_annote_0=np.array(adata.obs['state_annotation'])
    #Tmap_cell_id_t2=adata.uns['Tmap_cell_id_t2']
    #Tmap_cell_id_t1=adata.uns['Tmap_cell_id_t1']
    #time_info=np.array(adata.obs['time_info'])


    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    data_des=adata.uns['data_des'][0]
    #data_path=adata.uns['data_path'][0]
    figure_path=adata.uns['figure_path'][0]
    clone_annot=adata.obsm['cell_by_clone_matrix']
    time_info=np.array(adata.obs['time_info'])
    time_points=np.sort(list(set(time_info)))

        # using all data
    for my_id in selected_clone_list:
        if my_id < clone_annot.shape[1]:
            fig = plt.figure(figsize=(4, 3))
            ax=plt.subplot(1,1,1)
            plot_one_gene_SW(x_emb,y_emb,np.zeros(len(y_emb)),ax=ax,point_size=point_size)
            if len(time_points)>len(color_list):
                idx=clone_annot[:,my_id].A.flatten()>0
                ax.plot(x_emb[idx],y_emb[idx],'.',color='black',markersize=5*point_size)
            else:
                for j, xx in enumerate(time_points):
                    idx_t=time_info==time_points[j]
                    idx_clone=clone_annot[:,my_id].A.flatten()>0
                    idx=idx_t & idx_clone
                    ax.plot(x_emb[idx],y_emb[idx],'.',color=color_list[j],markersize=12*point_size,markeredgecolor='black',markeredgewidth=1)


            fig.savefig(f'{figure_path}/{data_des}_different_clones_{my_id}.png',transparent=True,dpi=300)
        else:
            print(f"No such clone id: {my_id}")




def plot_progenitor_states_towards_a_given_fate(adata,selected_fate='',used_map_name='transition_map',map_backwards=True,map_threshold=0.1,plot_separately=False,apply_time_constaint=False,point_size=2):
    '''
        This function use the inferred transition map or demultiplexed map to infer the probability that an ancestor state will enter a designated fate cluster.
    
        The inference is applied recursively. Start with the cell states for the selected fate, then use selected map to infer the immediate ancestor states.
        Then, using these putative ancestor state as the input, find the immediate ancestors for these input states. This goes on until all time points are exhausted.
    
        selected_fate: a target fate or a list of them among the state_annotation. If multiple fates are provided, they are combined into a single one. It is assumed that this fate should be at the last time point.
        used_map_name: transition_map, demultiplexed_map, OT_transition_map, or HighVar_transition_map
        plot_separately: True, the putative ancerstor states from each recursion is plot separately; False, combine all results in a single plot
        apply_time_constraint: True, apply time constraint for each ancestor inference; False, not
        map_threshold: the relative threshold to call a state as an ancestor state. 
    
        We always use the probabilistic map, which is more realiable. Otherwise, the result is very sensitive to thresholding

    '''        

    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    check_transition_map(adata)

    state_annote_0=np.array(adata.obs['state_annotation'])
    if map_backwards:
        cell_id_t1=adata.uns['Tmap_cell_id_t1']
        cell_id_t2=adata.uns['Tmap_cell_id_t2']

    else:
        cell_id_t2=adata.uns['Tmap_cell_id_t1']
        cell_id_t1=adata.uns['Tmap_cell_id_t2']

    time_info=np.array(adata.obs['time_info'])
    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    data_des=adata.uns['data_des'][0]
    figure_path=adata.uns['figure_path'][0]

#     if selected_fate not in list(state_annote_0):
#     print(f"selected_fate not valid. It should be among {set(state_annote_0)}")


    if used_map_name not in adata.uns['available_map']:
        print(f"Error, used_map_name should be among {adata.uns['available_map']}")

    else:
        ##### we normalize the map in advance to avoid normalization later in mapout_trajectories
        used_map_0=adata.uns[used_map_name]
        resol=10**(-10)
        used_map_0=hf.sparse_rowwise_multiply(used_map_0,1/(resol+np.sum(used_map_0,1).A.flatten()))

        if map_backwards:
            used_map=used_map_0
        else:
            used_map=used_map_0.T

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

        if np.sum(selected_idx)==0:
            print("No states selected. Please make sure the selected_fate are among {set(state_annote_0)")
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
                    plot_one_gene_SW(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"t={sort_time_info[0]}, cell #:{np.sum(prob_0r_0>0)}");
                    for k,t_0 in enumerate(sort_time_info[1:]):
                        ax1=plt.subplot(row,col,2+k)
                        plot_one_gene_SW(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"Fate prob.: t={t_0}")

                else:
                    plot_one_gene_SW(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"cell #:{np.sum(prob_0r_0>0)}");
                    for k,t_0 in enumerate(sort_time_info[1:]):
                        ax1=plt.subplot(row,col,2+k)
                        plot_one_gene_SW(x_emb,y_emb,prob_array[k],ax=ax1,point_size=point_size,title=f"Fate prob: {k+1} back propagte")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_separate_BW{map_backwards}.png',transparent=True,dpi=300)  
            else:

                col=2; row=1
                fig = plt.figure(figsize=(4 * col, 3.5 * row))
                ax0=plt.subplot(row,col,1)
                plot_one_gene_SW(x_emb,y_emb,prob_0r_0,ax=ax0,point_size=point_size,title=f"t={sort_time_info[0]}, cell #:{np.sum(prob_0r_0>0)}");

                ax1=plt.subplot(row,col,2)
                plot_one_gene_SW(x_emb,y_emb,cumu_prob,ax=ax1,point_size=point_size,title=f"Fate prob. (all time)")

                fig.savefig(f'{figure_path}/{data_des}_predicting_fate_trajectory_allTime_BW{map_backwards}.png',transparent=True,dpi=300)

            if 'fate_trajectory' not in adata.uns.keys():
                adata.uns['fate_trajectory']={}

            combined_prob=cumu_prob+prob_0r
            if map_backwards:
                adata.uns['fate_trajectory'][fate_name]={'map_backwards':combined_prob} # include both the targeted fate cluster and the inferred earlier states
            else:
                adata.uns['fate_trajectory'][fate_name]={'map_forward':combined_prob} 


def plot_gene_trend_towards_a_given_fate(adata,selected_fate='',gene_name_list=[''],map_backwards=True,invert_PseudoTime=False,include_target_states=False,
                compute_new=True,fig_width=3.5,global_rescale_percentile=99,n_neighbors=8,plot_raw_data=False,point_size=2):
    '''
        This method assumes that the selected states can form a continuum after building a knn graph. 

        Always run 'plot_progenitor_states_towards_a_given_fate' before this function. This function use the inferred ancestor cell states, and compute the pseudotime for these states, and on top of that plot the gene trend. 

        Compute_new: True, compute the pseudotime from scratch; False, load previously saved results. Be cautious, as the previously saved one might not correspond to this dataset
        selected_fate: a target fate or a list of them among the state_annotation. If multiple fates are provided, they are combined into a single one. It is assumed that this fate should be at the last time point.
        gene_name_list: name of the genes to be queried.

    '''
    
    
    #final_id,temp_idx=state_annotation_SW(adata,sel_cell_id,x_emb,y_emb,which_branch,add_progenitors=True)
        

    #transition_map=adata.uns['transition_map']
    #demultiplexed_map=adata.uns['demultiplexed_map']
    state_annote_0=np.array(adata.obs['state_annotation'])
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
        
        
    x_emb=adata.obsm['X_umap'][:,0]
    y_emb=adata.obsm['X_umap'][:,1]
    data_des=adata.uns['data_des'][0]
    data_path=adata.uns['data_path'][0]
    figure_path=adata.uns['figure_path'][0]
    file_name=f'{data_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}_{map_backwards}.npy'


    
    if ('fate_trajectory' not in adata.uns.keys()) or (fate_name not in adata.uns['fate_trajectory'].keys()):
        print(f"The prongenitor states have not been inferred yet. Please first run ---- plot_progenitor_states_towards_a_given_fate ----")
        #plot_progenitor_states_towards_a_given_fate(adata,fate_name='',used_map_name='transition_map',plot_separately=True,apply_time_constaint=False,map_threshold=0.1,point_size=point_size)
        
    else:
        if map_backwards:
            if 'map_backwards' not in adata.uns['fate_trajectory'][fate_name].keys():
                print(f"The prongenitor states have not been inferred yet for *map_backwards=True*. Please first run ---- plot_progenitor_states_towards_a_given_fate ----")
            else:
                prob_0=np.array(adata.uns['fate_trajectory'][fate_name]['map_backwards'])
        else:
            if 'map_forward' not in adata.uns['fate_trajectory'][fate_name].keys():
                print(f"The prongenitor states have not been inferred yet for *map_backwards=False*. Please first run ---- plot_progenitor_states_towards_a_given_fate ----")
            else:
                prob_0=np.array(adata.uns['fate_trajectory'][fate_name]['map_forward'])
        
        if not include_target_states:
            sel_cell_idx=(prob_0>0) & time_index_t1
        else:
            sel_cell_idx=prob_0>0
            
        #print(sel_cell_idx)
        sel_cell_id=np.nonzero(sel_cell_idx)[0]


        #file_name=f"data/Pseudotime_{which_branch}_t2.npy"
        if os.path.exists(file_name) and (not compute_new):
            print("Load pre-computed pseudotime")
            PseudoTime=np.load(file_name)
        else:
            #t=time.time()
            #print("Compute the pseudotime now")
            ### Generate the pseudo time ordering
            
            from sklearn import manifold
            data_matrix=adata.obsm['X_pca'][sel_cell_idx]
            method=manifold.SpectralEmbedding(n_components=1,n_neighbors=n_neighbors)
            PseudoTime = method.fit_transform(data_matrix)
            np.save(file_name,PseudoTime)
            #print("Run time:",time.time()-t)


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
            print("The selected cell states do not form a connected graph. Cannot form a continuum of pseudoTime. Only plot the raw data")
            plot_raw_data=True

        ## plot the pseudotime ordering
        fig = plt.figure(figsize=(12,4))
        ax=plt.subplot(1,2,1)
        plot_one_gene_SW(x_emb,y_emb,sel_cell_idx,ax=ax,title='Selected cells',point_size=point_size)
        ax1=plt.subplot(1,2,2)
        plot_one_gene_SW(x_emb[sel_cell_idx],y_emb[sel_cell_idx],PseudoTime,ax=ax1,title='Pseudo Time',point_size=point_size)
        #plot_one_gene_SW(x_emb[final_id],y_emb[final_id],PseudoTime,ax=ax1,title='Pseudo time')
        Clb=fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Reds), ax=ax1,label='Pseudo time')
        fig.savefig(f'{figure_path}/{data_des}_fate_trajectory_pseudoTime_{fate_name}_{map_backwards}.png',transparent=True,dpi=300)

        temp_dict={'PseudoTime':PseudoTime}
        for gene_name in gene_name_list:
            yy_max=np.percentile(adata.obs_vector(gene_name),global_rescale_percentile) # global blackground
            yy=np.array(adata.obs_vector(gene_name)[sel_cell_idx])
            rescaled_yy=yy*prob_0[sel_cell_idx]/yy_max # rescaled by global background
            temp_dict[gene_name]=rescaled_yy
        
        
        data2=pd.DataFrame(temp_dict)
        data2_melt=pd.melt(data2,id_vars=['PseudoTime'],value_vars=gene_name_list)
        gplot=ggplot(data=data2_melt,mapping=aes(x="PseudoTime", y='value',color='variable')) + \
        (geom_point() if plot_raw_data else stat_smooth(method='loess')) +\
        theme_classic()+\
        labs(x="Pseudo time",
             y="Normalized gene expression",
              color="Gene name")
   
        gplot.save(f'{figure_path}/{data_des}_fate_trajectory_pseutoTime_gene_expression_{fate_name}_{map_backwards}.png',width=fig_width, height=fig_width*0.618,dpi=300)
        gplot.draw()



############# refine clusters

def refine_cell_state_annotation_by_leiden_clustering(adata,selected_time_points=[],leiden_resolution=0.5,n_neighbors=5,confirm_change=False,cluster_name_prefix='S'):

    time_info=adata.obs['time_info']
    availabel_time_points=list(set(time_info))
    
    if len(selected_time_points)==0:
        selected_time_points=availabel_time_points
        
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
        print("Change state annotation at adata.obs['state_annotation']")
        orig_state_annot=np.array(adata.obs['state_annotation'])
        temp_array=np.array(adata_sp.obs['leiden'])
        for j in range(len(temp_array)):
            temp_array[j]=cluster_name_prefix+temp_array[j]
        
        orig_state_annot[sp_idx]=temp_array
        adata.obs['state_annotation']=pd.Categorical(orig_state_annot)
        sc.pl.umap(adata,color='state_annotation')
    
    
    

def refine_cell_state_annotation_by_marker_genes(adata,marker_genes=[],gene_threshold=0.1,selected_time_points=[],new_cluster_name='',confirm_change=False,add_neighbor_N=5):
    '''
        A state is selected if it expressed all genes in the list of marker_genes, and above the relative threshold gene_threshold, and satisfy the time point constraint. In addition, we also include cell states neighboring to these valid states to smooth the selection.

        new_cluster_name: name for the new cluser. If not set, use the first gene name
        selected_time_points: selected states need to be within these time points. If unset, use all time points.
        add_neighbor_N: add neighboring cells according to the KNN graph with N=add_neighbor_N
        gene_threshold: the relative gene threshold, [0,1]
        confirm_change: default is False, to allow exploration of different marker genes. Once decided, change it to True to confirm the change.

    '''

    
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
            thresh=gene_threshold*np.max(expression)
            idx=expression>thresh
            selected_states_idx=selected_states_idx & idx
            
            tot_name=tot_name+marker_gene_temp
            
    # add temporal constraint
    selected_states_idx[~sp_idx]=0    
    
    if np.sum(selected_states_idx)>0:
        # add neighboring cells to smooth selected cells (in case the expression is sparse)
        selected_states_idx=hf.add_neighboring_cells_to_a_map(selected_states_idx,adata,neighbor_N=add_neighbor_N)

        fig=plt.figure(figsize=(4,3));ax=plt.subplot(1,1,1)
        plot_one_gene_SW(x_emb,y_emb,selected_states_idx,ax=ax)
        ax.set_title(f"{tot_name}; Selected #: {np.sum(selected_states_idx)}")
        #print(f"Selected cell state number: {np.sum(selected_states_idx)}")


        if confirm_change:
            print("Change state annotation at adata.obs['state_annotation']")
            if new_cluster_name=='':
                new_cluster_name=marker_genes[0]

            orig_state_annot=np.array(adata.obs['state_annotation'])
            orig_state_annot[selected_states_idx]=np.array([new_cluster_name for j in range(np.sum(selected_states_idx))])
            adata.obs['state_annotation']=pd.Categorical(orig_state_annot)
            sc.pl.umap(adata,color='state_annotation')

