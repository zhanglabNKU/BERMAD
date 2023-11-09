# BERMAD: batch effect removal for single-cell RNA-seq data using a multi-layer adaptation autoencoder with dual-channel framework
Code and data for using BERMAD, a novel deep learning based framework for batch effect removal in scRNA-seq data. 

## Install
git clone https://github.com/zhanglabNKU/BERMAD.git  
cd BERMAD/

## R Dependencies
* Seurat 2.3.0

## Python Dependencies
* Python 3.7.7
* scikit-learn 0.23.2
* pytorch 1.3.1
* pandas 1.0.4

## Usage
Given several datasets (each treated as a batch) for combination, there are two main steps: (i) preprocess the datasets and run metaneighbor algorithm to compute cluster similarities; (ii) train a BERMAD model for batch correction.
### Data preprocessing
Run the R script pre_processing.R as follows:
```
Rscript pre_processing.R folder_name file1 file2 ...
```
For example:
```
Rscript pre_processing.R example batch1.csv batch2.csv
```
> The two datasets batch1.csv and batch2.csv (must be csv form) will be processed by the script and you will get three files saved in the same folder: the processed data named batch1_seurat.csv and batch2_seurat.csv, a file named metaneighbor.csv containing values of the cluster similarities between different batches.
### Batch correction
Run the python script main.py to combine the datasets and remove batch effects as follows:
```
python main.py -data_folder folder -files file1 file2 ... -similarity_thr thr_value
```
For example:
```
python main.py -data_folder example/ -files batch1_seurat.csv batch2_seurat.csv -similarity_thr 0.9
```
> This command will train a BERMAD model for the selected files in the data_folder with the similarity threshold. When the training is finished, the datasets will be combined without batch effectes and the result file named combined.csv will be saved in the same data folder.  

In addition, some optional parameters are also available:
* `-num_epochs`: number of the training epochs (default=2000)
* `-code_dim`: dimension of the embedded code (default=20)
* `-base_lr`: base learning rate for network training (default=1e-3)
* `-lr_step`: step decay of learning rates (default=200)
* `-gamma`: hyperparameter for adversarial learning (default=1)
* `-alpha`: hyperparameter for hidden1 layer of BERMAD (default=0.1)
* `-beta`: hyperparameter for hidden2 layer of BERMAD (default=0.1)
* `-delta`: hyperparameter for code layer of BERMAD (default=0.5)

Under most circumstances, you don't need to change the optional parameters.  

Use the help command to print all the options:
```
python main.py --help
```

## Data availability
The download links of all the datasets are given in the folder named data.
