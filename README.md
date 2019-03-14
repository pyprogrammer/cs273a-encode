# CS273A - ENCODE Imputation Challenge Pipeline and Analysis Code

Required: Python3.6
Packages:
 - tensorflow
 - tensorflow-lattice
 - numpy
 - pandas
 - Pytables (tables)
 - PyBigWig
 - BioPython
 - Matplotlib
 
 
## Pipeline Setup
First, data will need to be downloaded from Synapse for the challenge. The code in the Data Analysis notebook provides a thin
wrapper around PyBigWig to convert these files into Numpy files, which are significantly easier to work with and can also be
disk-mapped to avoid large memory usage.

## Lattice Regression
The Lattice Regression model is available in lattice_cross.py, with either ETL or RTL Models. In my experience the RTL model trains
significantly better, as the ETL model tends towards a mean prediction. For more information about lattice models, refer to
[Tensorflow Lattice](https://github.com/tensorflow/lattice).

## Heatmap Computation
There is currently a bug in the computation of the Assay-similarity-score (And similarly the cell-line-similarity-score) where
the vector normalization term should be computed with respect to the overlapping assay values instead of the full values.
The uncorrected values are found in Chr22ASS.npz and FullChromASS.npz.

