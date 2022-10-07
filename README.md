# quick_ml

biodocker.py is just a python script to run dockerized FeatSEE analyses. 

./biodocker.py MODE OPERATION PARAMS...


MODE = { batch, docker }




## Docker mode

Run a FeatSEE analysis (feature selection, evaluation and so on). 

OPERATION = { evaluate, selection, GA, ...}

For more information and required parameters, use the -h (--help) option: 

./biodocker.py docker OPERATION -h

(e.g. ./biodocker.py docker evaluate -h)



## Batch mode:

Run a batch of analyses on the same data. 
For example, to run a classification task by changing the feature target, or the classes to be considered as positive and negative. 

OPERATION = { init, run }


* Init operation: initialize the batch by creating two configuration files. You have to provide the batch folder, the datasets and the features to be used.  
You have to specify the FEATSEE_OPERATION (e.g. evaluate, selection, GA) to be performed. 

./biodocker.py batch init --op FEATSEE_OPERATION -o BATCH_FOLDER -i INPUT_DATASET... -f FEATURES... -v TEST_SETS...


1. Create the BATCH_FOLDER. 
2. Create two files within BATCH_FOLDER:

    . **batch.tsv** is a tab-separated file that will contains a row for each run you want to perform. For each of those runs, you have to specify (a) the target feature, (b) the 1+ value(s) to be considered as positive and (c) negative classes. If you want to specify more than one value, separate them with commas (aka virgole).

    . **params.json** contains the parameters of the analysis. You can adjust all the parameters except for IO files, which have to be provided during "batch init". 


* Run operation: once you finished to write in the configuration files, you can go to the run batch phase. 

./biodocker.py batch run -b BATCH_FOLDER

It will create a bashino.sh script containing the commands to run your analyses. Run that script manually. The end. 








