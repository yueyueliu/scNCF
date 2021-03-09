##scNCF: Imputation for single-cell RNA-seq da-ta via neural collaborative filtering

we introduce an effective method for scRNA-seq data imputation, called scNCF (Neural Collaborative Filtration for scRNA-seq data imputation), which combines multi-layer perceptrons with matrix factorization to capture linear and nonlinear associations in scRNA-seq profile data. Our method is an extensible and effective scRNA-seq data imputation method. 

## Prerequisites
+ numpy>=1.14.2 
+ pandas>=0.22.0 
+ scipy>=0.19.1 
+ scikit-learn>=0.19.1 
+ torch>=1.0.0 
+ tqdm>=4.28.1 
+ matplotlib>=3.0.2 
+ seaborn>=0.9.0

## Data
A small dataset from Biase is included for demonstration.

## Demo
We gave a predict.py to demonstrate the use of scNCF. The output consists of clustering assignment and imputation results.

## Codes
Two python files are included:
- predict.py: Perform interpolation tasks and many auxiliary functions 
- NCF.py: A class of neural collaborative filtering
