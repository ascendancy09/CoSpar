|PyPI| |PyPIDownloads| |Docs|

CoSpar - dynamic inference by integrating transcriptome and lineage information
===============================================================================

.. image:: https://user-images.githubusercontent.com/4595786/104988296-b987ce00-59e5-11eb-8dbe-a463b355a9fd.png
   :width: 100px
   :align: left

**CoSpar** is a toolkit for dynamic inference from lineage-traced single cells. |br|
The methods are based on
`S.-W. Wang & A.M. Klein (ToBeSubmitted, 2021) <https://doi.org/xxx>`_.

Dynamic inference based on single-cell state measurement alone requires serious simplifications. On the other hand, direct dynamic measurement via lineage tracing only captures partial information and is very noisy. CoSpar integrates both state and lineage information to infer the transition map of a development/differentiation system. It gains superior robustness and accuracy by exploiting both the local coherence and sparsity of differentiation transitions, i.e., neighboring initial states share similar yet sparse fate outcomes.  Building around the most popular :class:`~anndata.AnnData` object in the single-cell community, CoSpar is dedicated to building an integrated analysis framework for datasets with both state and lineage information. It offers essential toolkits for analyzing clonal information, state information, or their integration. 

CoSpar's key applications
^^^^^^^^^^^^^^^^^^^^^^^^^
- infer transition maps using only clonal information, state information, or their integration. 
- identify early fate bias/commitment of a cell 
- infer differentiation trajectories leading to a fate.
- infer gene expression dynamics along the trajectory. 
- infer putative driver genes.
- infer fate coupling.



Reference
^^^^^^^^^
Shou-Wen Wang & Allon M. Klein (2021), Coherent sparsity optimization for dynamic inference by integrating state and lineage information,
`ToBeSubmitted <https://doi.org/xxx>`_.
|dim|


Support
^^^^^^^
Feel free to submit an `issue <https://github.com/ShouWenWang/cospar/issues/new/choose>`_
or send us an `email <mailto:shouwen_wang@hms.harvard.edu>`_.
Your help to improve CoSpar is highly appreciated.





.. |PyPI| image:: https://img.shields.io/pypi/v/scvelo.svg
   :target: https://pypi.org/project/scvelo

.. |PyPIDownloads| image:: https://pepy.tech/badge/scvelo
   :target: https://pepy.tech/project/scvelo

.. |Docs| image:: https://readthedocs.org/projects/scvelo/badge/?version=latest
   :target: https://scvelo.readthedocs.io

..
  .. |travis| image:: https://travis-ci.org/theislab/cospar.svg?branch=master
     :target: https://travis-ci.org/theislab/cospar


.. |br| raw:: html

  <br/>

..
 .. |meet| raw:: html



.. |dim| raw:: html

   <span class="__dimensions_badge_embed__" data-doi="xxx" data-style="small_rectangle"></span>
   <script async src="https://badge.dimensions.ai/badge.js" charset="utf-8"></script>