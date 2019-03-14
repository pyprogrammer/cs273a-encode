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

The current model uses a combination of neighboring assay values (same cell-line, different assays, different cell-line, same assay), chromosome number, current assay number (i.e. M05), as well as an envelope of surrounding bases (currently 21).

## Heatmap Computation
There is currently a bug in the computation of the Assay-similarity-score (And similarly the cell-line-similarity-score) where
the vector normalization term (denominator) should be computed with respect to the overlapping assay values instead of the full values. A simple approximate correction is given by dividing by number of shared cell lines, but that essentially assumes these assays show an "average" behavior. This is likely untrue on these cell lines.
The uncorrected values can be found in Chr22ASS.npz and FullChromASS.npz.

