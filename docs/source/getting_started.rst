Getting Started
---------------

Here, you will be briefly guided through the basics of how to use CoSpar.
Once you are set, the following tutorials go straight into analysis of transition dynamics. 

The input data for CoSpar are matrices for state and clonal information, and a vector for temporal annotation. We assume that the data have more than one time points. 


CoSpar workflow at a glance
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import CoSpar as::

    import cospar as cs

For beautified visualization you can change the matplotlib settings to our defaults with::

    cs.settings.set_figure_params()

Initialization
''''''''''''''
Given the gene expression matrix, clonal matrix, and other information, initialize the anndata object using::

    adata = cs.initialize_adata_object(RNA_count_matrix,gene_names,time_info,
    X_clone=[],X_pca=[],X_emb=[],state_info=[],**params)

The :class:`~anndata.AnnData` object adata stores the count matrix (``adata.X``), genes / variables (``adata.var``), and  annotation of cells / observations (``adata.obs['time_info']``).  The clonal matrix `X_clone` is optional, and will be stored at  ``adata.obsm['X_clone']``.  If not provided, you can still infer transition map based on state and temporal information alone, and proceed with the analysis. You can also provide the selected PCA matrix `X_pca`,  the embedding matrix `X_emb`, and the state annotation `state_info`, which will be stored at ``adata.obsm['X_pca']``, ``adata.obsm['X_emb']``, and ``adata.obs['state_info']``, respectively. 

.. raw:: html

    <img src="http://falexwolf.de/img/scanpy/anndata.svg" style="width: 300px">

If you do not have a datasets yet, you can still play around using one of the in-built datasets, e.g.::

    adata = cs.datasets.hematopoiesis_15perct()



Basic preprocessing
'''''''''''''''''''
Assuming basic quality control (excluding cells with low read count etc.) have been done, we provide basic preprocessing (gene selection and normalization) and dimension reduction related analysis (PCA, UMAP embedding etc.)  at (``cs.pp.*``)::

    cs.pp.get_highly_variable_genes(adata,**params)
    cs.pp.remove_cell_cycle_correlated_genes(adata,**params)
    cs.pp.get_X_pca(adata,**params)
    cs.pp.get_X_emb(adata,**params)
    cs.pp.get_state_info(adata,**params)

The first step `get_highly_variable_genes` alrealdy includes count matrix normalization. The second step removes cell cycle correlated genes among the selected highly variable genes.
This is optional, but recommended. 

These steps can also be performed by  external packages like :mod:`~scanpy`, which is also built around the :class:`~anndata.AnnData` object.  



Simple clonal analysis
''''''''''''''''''''''
We provide a few plotting functions to visually explore the clonal data before any downstream analysis. You can visualize clones on state manifold directly:: 
    
    cs.pl.clones_on_manifold(adata,**params)

You can generate the barcode heatmap across given clusters to inspect clonal behavior::
    
    cs.pl.barcode_heatmap(adata,**params)

You can quantify the clonal coupling across different fate clusters::
    
    cs.pl.fate_coupling_from_clones(adata,**params)

Strong coupling implies the existence of bi-potent or multi-potent cell states at the time of barcoding. Finally, you can infer the fate bias of each clone towards a designated fate cluster::
    
    cs.pl.clonal_fate_bias(adata,**params)

A biased clone towards this cluster has a statistically significant cell fraction in this cluster.




Transition map inference
''''''''''''''''''''''''
The core of the software is the efficient and robust inference of a transition map by integrating state and clonal information. If the data have multiple clonal time points, you can run::
    
    adata=cs.tmap.infer_Tmap_from_multitime_clones(adata_orig,selected_clonal_time_points,**params) 

It subsamples the input data according to selected time points (:math:`\ge 2`) with clonal information, computes the transition map (stored at `adata.uns['transition_map']`), and return the subsampled adata object. The inferred map allows transitions between neighboring time points. For example, if selected_clonal_time_points=['day1', 'day2', 'day3'], then it computes transitions for pairs ('day1', 'day2') and ('day2', 'day3'), but not ('day1', 'day3'). As a byproduct, it also returns a transition map that allows only intra-clone transitions (`adata.uns['intraclone_transition_map']`). The intra-clone transition map can also be computed from `adata.uns['transition_map']`) at preferred parameters by running:: 
    
    cs.tmap.infer_intraclone_Tmap(adata,**params)

If the data have only one clonal time point, or you wish to infer the transition map just based on a single clonal time point, you can run::

    cs.tmap.infer_Tmap_from_one_time_clones(adata_orig,initial_time_points, clonal_time_point,initialize_method='OT',**params)

You need to define both `initial_time_points` and `clonal_time_point`. We provide two methods for initializing the map using state information alone: 1) 'OT' for using standard optimal transport approach; 2) 'HighVar' for a customized approach that convert highly variable genes into pseudo multi-time clones and run `cs.tmap.infer_Tmap_from_multitime_clones` to construct the map. Depending on the choice,  the initialized map is stored at `adata.uns['OT_transition_map']` or  `adata.uns['HighVar_transition_map']`. Afterwards, CoSpar performs a joint optimization to infer both the initial clonal structure and also the transition map. The final product is stored at `adata.uns['transition_map']`. This method returns a map for transitions from all given initial time points to the designated clonal time point.  For example, if initial_time_points=['day1', 'day2'], and clonal_time_point='day3', then the method computes transitions for pairs ('day1', 'day3') and ('day2', 'day3'). 

If you do not have any clonal information, you can still run::
    
    cs.tmap.infer_Tmap_from_state_info_alone(adata_orig,initial_time_points,target_time_point,initialize_method='OT',**params)

It is the same as `cs.tmap.infer_Tmap_from_one_time_clones` except that we exclude the the final joint optimization that requires clonal information. 

We also provide simple methods that infer transition map from only the clonal information::

    cs.tmap.infer_Tmap_from_clonal_info_alone(adata,**params)

The result is stored at `adata.uns['clonal_transition_map']`. 

Visualization
'''''''''''''

Finally, each of the computed transition maps can be explored on state embedding at single cell level using a variaty of plotting functions. There are some common parameters: 

* `used_map_name` (`str`). Determines which transition map to use for analysis. Choices: {'transition_map', 'intraclone_transition_map', 'OT_transition_map', 'HighVar_transition_map','clonal_transition_map'}

* `selected_fates` (`list` of `str`). Selected clusters to aggregate differentiation dynamics and visualize fate bias etc.. The selected_fates allows nested structure, e.g., selected_fates=['1', ['0', '2']] selects two clusters:  cluster '1' and the other that combines '0' and '2'. 

* `map_backwards` (`bool`).  We can analyze either the forward transitions, i.e., where the selected states or clusters are going (`map_backwards=False`), or the backward transitions, i.e., where these selected states or clusters came from (`map_backwards=False`). The latter is more useful, and is the default. 

Below, we frame the task in the language of analyzing backward transitions for convenience. To see transition probability from one cell to others, run:: 
    
    cs.pl.single_cell_transition(adata,**params)

To see the probability of initial cell states to give rise to given fate clusters (or the other way around if `map_backwards=False`), run::
    
    cs.pl.fate_map(adata,**params)

To infer the relative fate bias of initial cell states to given fate clusters, run::

    cs.pl.fate_bias_intrinsic(adata,**params)
    cs.pl.fate_bias_from_binary_competition(adata,**params)

The first method (`fate_bias_intrinsic`) quantify the fate bias of a state towards each designated cluster by normalizing the predicted fate probability with the expected fate bias, the relative proportion of this cluster among all cell states in the target state space of the map. The second method evaluates the fate bias of a state towards one cluster over the other.

To infer the dynamic trajectory towards given fate clusters, run::

    cs.pl.dynamic_trajectory_from_intrinsic_bias(adata,**params)
    cs.pl.dynamic_trajectory_from_competition_bias(adata,**params)
    cs.pl.dynamic_trajectory_via_iterative_mapping(adata,**params)

The first two methods assumes two input fate clusters, and infer the each corresponding trajectory by thresholding the fate bias using either the intrinsic method or the binary competition method. They export the selected ancestor state for the two fate clusters at `adata.obs['Cell_group_A']` and `adata.obs['Cell_group_B']`, which can be used to infer the differentially expressed genes (specifically, driver genes for fate bifurcation here) by running::
    
    cs.pl.differential_genes(adata,**params)

The last method (`dynamic_trajectory_via_iterative_mapping`) infers the trajectory by iteratively tracing a selected fate cluster all the way back to the initial time point. The inferred trajectory for each fate from all three methods will be saved at `adata.uns['dynamic_trajectory'][fate_name]`, and we can explore the gene expression dynamics along this trajectory using:: 

    cs.pl.gene_expression_dynamics(adata,selected_fate,gene_name_list,**params)

The `selected_fate` should be among those that have run the dynamic trajectory inference stated above. 


If there are multiple mature fate clusters, you can infer the their differentiation coupling by::

    cs.pl.fate_coupling_from_Tmap(adata,**params)    



