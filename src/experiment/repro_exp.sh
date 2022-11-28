#!/bin/sh

# Go to the `src` directory
cd src/

# Reproducing Xcon Experimental Results 

# Extracting rules with and without size limit. The extracted rules will be saved in the `rules` directory.
./experiment/rextract_size5.sh && ./experiment/rextract_all.sh 

# Train all the DL, BT and BNN models:
./experiment/dl.sh && ./experiment/bt_nbestim_25_maxdepth_3.sh && ./experiment/bnn.sh
 
# Given the trained models and extracted rules, enumerate explanations for DLs, BTs and BNNs by the following scripts. Logs are saved in `logs/dl`, `logs/bt` and `logs/bnn` directories respectively.

# For DLs:
./experiment/dl_exp_abd_ori.sh && ./experiment/dl_exp_abd_size5.sh && ./experiment/dl_exp_con_ori.sh && ./experiment/dl_exp_con_size5.sh

# For BTs:
./experiment/bt_exp_abd_ori.sh && ./experiment/bt_exp_abd_size5.sh && ./experiment/bt_exp_con_ori.sh && ./experiment/bt_exp_con_size5.sh

# For BNNs:
./experiment/bnn_exp_abd_ori.sh && experiment/bnn_exp_abd_size5.sh && ./experiment/bnn_exp_con_ori.sh && experiment/bnn_exp_con_size5.sh

# Compute the usefulness of background knowledge in 4 selected datasets:
./experiment/examples_userules.sh
 
#Since 62 dataset and 3 machine learning models are considered, running the experiments will take a while. These scripts collect the necessary data including extracted rules, running time and explanations size, et cetera. All the logs will be saved in the `logs` directory.

# Parse logs and generate plots and tables. All plots are saved in `plots` directory.

# Compute accuracy of extracted rules and generate a cactus plot:
python ./gnrt_plots/racry.py acry && python ./gnrt_plots/racry.py plot

# Parse explanation logs and generate plots and table.
python ./gnrt_plots/parse_explog.py xcon

# Parse logs of usefulness of background knowledge and generate a table
python ./gnrt_plots/parse_uselog.py


# Reproducing Experimental Apriori and Eclat Results 

# Extracting rules with and without size limit. The extracted rules will be saved in the `apriori_rules` and `eclat_rules` directories.

# Extracting background knowledge by Apriori
./experiment/apriori/rextract_size5.sh 

# Extracting background knowledge by Eclat
./experiment/eclat/rextract_size5.sh 

# Train all the DL, BT and BNN models:
./experiment/dl.sh && ./experiment/bt_nbestim_25_maxdepth_3.sh && ./experiment/bnn.sh
 
# Given the trained models and rules extract by Eclat, enumerate explanations for DLs, BTs and BNNs by the following scripts. Logs are saved in `eclat_logs/dl`, `eclat_logs/bt` and `eclat_logs/bnn` directories respectively.

# For DLs:
./experiment/eclat/dl_exp_abd_ori.sh && ./experiment/eclat/dl_exp_abd_size5.sh && ./experiment/eclat/dl_exp_con_ori.sh && ./experiment/eclat/dl_exp_con_size5.sh

# For BTs:
./experiment/eclat/bt_exp_abd_ori.sh && ./experiment/eclat/bt_exp_abd_size5.sh && ./experiment/eclat/bt_exp_con_ori.sh && ./experiment/eclat/bt_exp_con_size5.sh

# For BNNs:
./experiment/eclat/bnn_exp_abd_ori.sh && experiment/eclat/bnn_exp_abd_size5.sh && ./experiment/eclat/bnn_exp_con_ori.sh && experiment/eclat/bnn_exp_con_size5.sh
 
# Similar to xcon experiments, since 58 dataset and 3 machine learning models are considered, running the experiments will take a while. These scripts collect the necessary data including extracted rules, running time and explanations size, et cetera. All the logs will be saved in the `eclat_logs` directory.

# Parse logs and generate plots. All plots are saved in `plots/apriori` and `plots/eclat` directories.

# Generate the plots of the comparison among Apriori, Eclat and xcon regarding rule extraction runtime and the number of rules extracted.
python ./gnrt_plots/parse_bglog.py

# Parse explanation logs and generate plots.
python ./gnrt_plots/parse_explog.py eclat


# Reproducing Experimental Results of LIME, SHAP, and Anchor <a name="hexp"></a>

# Train all the DL, BT and BNN models:
./experiment/dl.sh && ./experiment/bt_nbestim_25_maxdepth_3.sh && ./experiment/bnn.sh

# Given the trained models, LIME, SHAP, and Anchor generate explanations for DLs, BTs, and BNNs by the following scripts. Logs are saved in `logs/hexp/dl`, `logs/hexp/bt` and `logs/hexp/bnn` directories respectively.
./experiment/hexp/dl_lime.sh && ./experiment/hexp/dl_shap.sh && ./experiment/hexp/dl_anchor.sh 
./experiment/hexp/bt_lime.sh && ./experiment/hexp/bt_shap.sh && ./experiment/hexp/bt_anchor.sh 
./experiment/hexp/bnn_lime.sh && ./experiment/hexp/bnn_shap.sh && ./experiment/hexp/bnn_anchor.sh 

# Parse logs and generate plots. All plots are saved in `plots/hexp`
# Parse logs to get explanations and generate plots of runtime.
python ./gnrt_plots/parse_hexplog.py

# Compute the correctness of explanations.
# For DL explanations:
./experiment/hexp/dl_lime_cor.sh && ./experiment/hexp/dl_shap_cor.sh && ./experiment/hexp/dl_anchor_cor.sh

# For BT explanations:
./experiment/hexp/bt_lime_cor.sh && ./experiment/hexp/bt_shap_cor.sh && ./experiment/hexp/bt_anchor_cor.sh

# For BNN explanations:
./experiment/hexp/bnn_lime_cor.sh && ./experiment/hexp/bnn_shap_cor.sh && ./experiment/hexp/bnn_anchor_cor.sh

# Generate plots of correctness comparison and size comparison 
python ./gnrt_plots/parse_correctcsv.py
