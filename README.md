# xcon

This directory contains the implementation used in the AAAI23 paper and [CoRR report](https://arxiv.org/abs/2206.09551/). The implementation can use the MaxSAT-based rule induction techniques to efficiently extract background information from a dataset. The implementation can also apply the extracted background knowledge as the additional constraints when computing a single formal abductive or contrastive explanation for a machine learning model prediction. The experiments evaluated the tool in 3 machine learning models, i.e. *decision lists(DLs)*, *boosted trees(BTs)* and *binarized neural networks(BNNs)*. 

## Instruction <a name="instrt"></a>
Before using the implementation, we need to extract the datasets stored in ```dataset.tar.xz```. To extract the datasets, please ensure ```tar``` is installed and run:
```
$ tar -xvf dataset.tar.xz
```
If interested in the logs and rules generated in the our experiments, please run:
```
$ tar -xvf logs.tar.xz && tar -xvf eclat_logs.tar.xz && tar -xvf rules.tar.xz
```

## Table of Content
* **[Requirements](#require)**
* **[Usage](#usage)**
	* [Rule extraction](#rextract)
	* [Enumerating formal explanations](#enumexpls)
		* [Preparing a dataset](#prepare)
		* [Training machine learning models](#train)
		* [Enumerating explanations for decision lists (DLs)](#enumdl)
		* [Enumerating explanations for boosted trees (BTs)](#enumbt)
		* [Enumerating explanations for binarized neural netwroks (BNNs)](#enumbnn)
* **[Reproducing Experimental Results](#expr)**

## Requirements <a name="require"></a>
The implementation is written as a set of Python scripts. The python version used in the experiments is 3.8.5. Some packages are required. To install requirements:
```
$ pip install requirements.txt
``` 

In addition to the packages above, PyFIM is also required. To install PyFIM, please follow the [instruction](https://borgelt.net/pyfim.html/).

## Usage <a name="usage"></a>

### Rule extraction <a name="rextract"></a>
`rextract.py` has a number of parameters for rule extraction, which can be set from the command line. To see the list of options, run (the executable script is located in `src`):
```
$ python ./rextract.py -h
```

To extract rules given a dataset, run:
```
$ python ./rextract.py -B --solver glucose3 -l <int> --no-ccheck -vv --blk --save-to <rules.json> -D <dataset.csv>
```
Option `-B` enables breaking symmetric rules to enumerate in the rule extracting process. `--solver glucose3 ` indicates that *glucose3* is selected as the SAT solver used in the tool. `--no-ccheck` means the inconsistency in the given dataset is not checked. The value of `-l` means compute at most this number of primes per sample in the rule extracting process. `--blk` enables blocking duplicate rules. `--save-to` is the location to save the extracted rules. `-D` indicates the dataset to process. The approach can be augmented with the option `-o size -C <int>`  , which provides an extraction limit of size `<int>`.
 
 Example:
 ```
 $ python ./rextract.py -B --solver glucose3 -l 1 --no-ccheck -vv --blk --save-to ../rules/size/q6_compas_train1.csv_size5.json -o size -C 5 -D ../bench/cv/train/quantise/q6/other/fairml/compas/compas_train1.csv
 ```

### Enumerating formal explanations <a name="enumexpls"></a>

#### Preparing a dataset <a name="prepare"></a>

Before enumerating explanations, we need to prepare the datasets for computing explanations for the 3 models and generating BT and BNN models (for generating DLs and extracting rules we use the original quantized dataset). 

1. Assume your dataset is stored in file `somepath/dataset.csv`.

2. Create another file named `somepath/dataset.csv.catcol` that contains the indices of the categorical columns of `somepath/dataset.csv`. Since all the datasets we used are *quantized* and so the file should contain the indices of all columns of `somepath/dataset.csv`. For instance, if there are 3 columns, the file should contain the lines
```
0
1
2
```
3. Then the following command:
```
$ python ./bt/xdual.py -p --pfiles dataset.csv,somename somepath/
```
creates a new file `somepath/somename_data.csv` with the categorical features properly handled. For example
```
$ python ./bt/xdual.py -p --pfiles compas.csv,compas ../bench/complete/quantise/q6/other/fairml/compas/
```

#### Training machine learning models <a name="train"></a>
A machine learning model must be trained before enumeration of explanations.
Use *compas* dataset as the example.

Training a DL by CN2 algorithm:
```
$ python ./dl/cn2.py -l ../bench/cv/train/quantise/q6/other/fairml/compas/compas_train1.csv > ./dl/dlmodels/quantise/q6/other/fairml/compas/compas_train1.csv.cn2
```
The value of `-l` indicates the train dataset. Here, we use use the original quantized train dataset.

Training a BT:

```
$ python ./bt/xdual.py -c -t -o ./bt/btmodels/q6 -d 3 -n 25 ../bench/cv/train/quantise/q6/other/fairml/compas/compas_train1_data.csv
```

Here, 25 trees per class are trained. Also, parameter `-c` is used because the data is categorical. We emphasize that parameter `-c` should be used, which is the case in the experimental results we obtained in the report. By default, the trained model is saved in the file `./bt/btmodels/q6/compas_train1_data/compas_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl`

Training a BNN:

```
$ python ./bnns/main_binary.py  -c ./bnns/configs/config_large.json  -d ../bench/complete/quantise/q6/other/fairml/compas/compas_data.csv -a ../bench/cv/train/quantise/q6/other/fairml/compas/compas_train1_data.csv -t ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv -r ./bnns/bnnmodels/large/quantise/q6/other/fairml/compas/compas_test1/
```
`-c` is the configuration of the BNN model. `-d` is the dataset. `-a` indicates the train data, while `-t` is the test (unseen) data. `-r` indicate the location of the generated BNN model. 


#### Enumerating explanations for decision lists (DLs) <a name="enumdl"></a>
`./dl/xdl.py` has a number of parameters, which can be set from the command line. To see the list of options, run:
```
$ python ./dl/xdl.py -h
```

To enumerate abductive or contrastive explanations for DLs, run:

```
$ python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n <int> -M -vv  -D <dataset.csv> -m <model.dl> -x <string> -k <rules.json>
```
`-n <int>` means <int> explanations are enumerated in an instance. `-M` enables the preference of smallest size explanations. The value of `-D` is the given dataset to compute explanations. The value of `-x` is either *abd* or *con*, indicating computing abductive or contrastive explanations. `-k` is optional, which specify the background knowledge used as the additional constraints.
 
For example, the following command is used to enumerating at most 20 smallest size abductive explanations for at most 100 instances in `q6_compas_test1.csv` with background knowledge.
 
```
$ python ./dl/xdl.py -B --solver glucose3 --no-ccheck -n 20 -M -vv  -D ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/fairml/compas/compas_train1.csv.cn2 -x abd -k ../rules/size/q6_compas_train1.csv_size5.json
```
 
To comopute which of the rules are used to reduce an explanation in DLs, you can fix the value of `-x` as *abd* and augment with option `-a use`. For example:

```
$ python ./dl/xdl.py -a use -B --solver glucose3 --no-ccheck -x abd -n 20 -M -vv -D ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv -m ./dl/dlmodels/quantise/q6/other/fairml/compas/compas_train1.csv.cn2 -S lin -k ../rules/all/q6_compas_train1.csv.json 
```
 
#### Enumerating explanations for boosted trees (BTs) <a name="enumbt"></a>

`./bt/xdual.py` has a number of parameters, which can be set from the command line. To see the list of options, run:

 ```
 $ python ./bt/xdual.py -h
 ```
 
 To enumerate abductive or contrastive explanations for BTs, run:
 
 ```
$ python ./bt/xdual.py -N <int> -M -c -e smt -s z3 -x all -v --xtype <string> -i <dataset.pkl> -k <rules.json> <dataset.csv> <model.pkl> 
 ```
 
Here, parameter `-e` specifies the model encoding (SMT) while parameter `-s` identifies an SMT solver to use (various SMT solvers can be installed in pySMT - here we use Z3). `-N <int>` means <int> explanations are enumerated in an instance. `-x all` enables the enumeration of explanations. The value of `--xtype` is either *abductive* or *contrastive*, indicating computing abductive or contrastive explanations. `-k` is optional, which specify the background knowledge used as the additional constraints. `-i <dataset.pkl>` indicate the data features information. `<dataset.csv>` and `<model.pkl>` specify the dataset and BT model.
 
For example:
```
$ python ./bt/xdual.py -N 20 -M -c -e smt -s z3 -x all -v --xtype abductive -i ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv.pkl -k ../rules/size/q6_compas_train1.csv_size5.json ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv ./bt/btmodels/q6/compas_train1_data/compas_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl
```
 
To compute which of the rules are used to reduce an explanation in BTs, you can fix the value of `--xtype` as *abductive* and augment with option `-A use`. For example:
 
```
$ python ./bt/xdual.py -A use -N 20 -M -c -e smt -s z3 -x all -v --xtype abductive -i ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv.pkl -k ../rules/all/q6_compas_train1.csv.json ../bench/cv/test/quantise/q6/other/fairml/compas/compas_test1_data.csv ./bt/btmodels/q6/compas_train1_data/compas_train1_data_nbestim_25_maxdepth_3_testsplit_0.2.mod.pkl
```

#### Enumerating explanations for binarized neural netwroks (BNNs) <a name="enumbnn"></a>

`./bnns/explain.py` has a number of parameters, which can be set from the command line. To see the list of options, run:
 
```
$ python ./bnns/explain.py -h
```

To enumerate abductive or contrastive explanations for BNNs:

```
$ python ./bnns/explain.py -m -N 20 -v 2 --load <string> -t <string> -k <string>
```
`-N <int>` means <int> explanations are enumerated in an instance. `-m` provides the preference of smallest size explanations.  `--load <string>` specifies the direcotory that contains a BNN model and the dataset location and . The value of `-t` is either *abd* or *con*, indicating the explanation type. `-k` is optional, which specify the background knowledge used as the additional constraints.

For example:
```
$ python ./bnns/explain.py -m -N 20 -v 2 --load ./bnns/bnnmodels/large/quantise/q6/other/fairml/compas/compas_test1/ -t abd -k ../rules/size/q6_compas_train1.csv_size5.json 
```

To compute which of the rules are used to reduce an explanation in BNN, you can fix the value of `-t` as *abd* and augment with option `-a use`. For example:
useful rules:

```
$ python ./bnns/explain.py -a use -v 2 -t abd -m -N 20 --load ./bnns/bnnmodels/large/quantise/q6/other/fairml/compas/compas_test1/ -k ../rules/all/q6_compas_train1.csv.json 
```


## Reproducing Experimental Results <a name="expr"></a>
 
Due to randomization used in the training phase, it seems unlikely that the experimental results reported in the report can be completely reproduced.
Similar results can be obtained by the following script:
```
$ ./src/experiment/repro_exp.sh
```

Since 62 dataset and 3 machine learning models are considered, running the experiments will take a while. These scripts collect the necessary data including extracted rules, running time and explanations size, et cetera.
