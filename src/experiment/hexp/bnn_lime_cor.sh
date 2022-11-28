#!/bin/sh
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q6/emmanuel/car/car_test1/ -c ../stats/hexp_expls/lime/bnn/q6_car.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q6/emmanuel/car/car_test1/ -c ../stats/hexp_expls/lime/bnn/q6_car.json -k ../rules/size/q6_car_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/emmanuel/cancer/cancer_test1/ -c ../stats/hexp_expls/lime/bnn/q6_cancer.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/emmanuel/cancer/cancer_test1/ -c ../stats/hexp_expls/lime/bnn/q6_cancer.json -k ../rules/size/q6_cancer_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/mlic/parkinsons_test1/ -c ../stats/hexp_expls/lime/bnn/q6_parkinsons.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/mlic/parkinsons_test1/ -c ../stats/hexp_expls/lime/bnn/q6_parkinsons.json -k ../rules/size/q6_parkinsons_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/mlic/pima_test1/ -c ../stats/hexp_expls/lime/bnn/q6_pima.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/mlic/pima_test1/ -c ../stats/hexp_expls/lime/bnn/q6_pima.json -k ../rules/size/q6_pima_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/house-votes-84/house-votes-84_test1/ -c ../stats/hexp_expls/lime/bnn/q6_house-votes-84.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/house-votes-84/house-votes-84_test1/ -c ../stats/hexp_expls/lime/bnn/q6_house-votes-84.json -k ../rules/size/q6_house-votes-84_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/zoo/zoo_test1/ -c ../stats/hexp_expls/lime/bnn/q6_zoo.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/other/zoo/zoo_test1/ -c ../stats/hexp_expls/lime/bnn/q6_zoo.json -k ../rules/size/q6_zoo_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/other/anchor/lending/lending_test1/ -c ../stats/hexp_expls/lime/bnn/q6_lending.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/other/anchor/lending/lending_test1/ -c ../stats/hexp_expls/lime/bnn/q6_lending.json -k ../rules/size/q6_lending_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/other/anchor/recidivism/recidivism_test1/ -c ../stats/hexp_expls/lime/bnn/q6_recidivism.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/other/anchor/recidivism/recidivism_test1/ -c ../stats/hexp_expls/lime/bnn/q6_recidivism.json -k ../rules/size/q6_recidivism_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/other/anchor/adult/adult_test1/ -c ../stats/hexp_expls/lime/bnn/q6_adult.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/other/anchor/adult/adult_test1/ -c ../stats/hexp_expls/lime/bnn/q6_adult.json -k ../rules/size/q6_adult_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q6/other/fairml/compas/compas_test1/ -c ../stats/hexp_expls/lime/bnn/q6_compas.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q6/other/fairml/compas/compas_test1/ -c ../stats/hexp_expls/lime/bnn/q6_compas.json -k ../rules/size/q6_compas_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/penn-ml/liver-disorder/liver-disorder_test1/ -c ../stats/hexp_expls/lime/bnn/q6_liver-disorder.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/penn-ml/liver-disorder/liver-disorder_test1/ -c ../stats/hexp_expls/lime/bnn/q6_liver-disorder.json -k ../rules/size/q6_liver-disorder_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/australian/australian_test1/ -c ../stats/hexp_expls/lime/bnn/q6_australian.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/australian/australian_test1/ -c ../stats/hexp_expls/lime/bnn/q6_australian.json -k ../rules/size/q6_australian_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/hungarian/hungarian_test1/ -c ../stats/hexp_expls/lime/bnn/q6_hungarian.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/hungarian/hungarian_test1/ -c ../stats/hexp_expls/lime/bnn/q6_hungarian.json -k ../rules/size/q6_hungarian_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/heart-c/heart-c_test1/ -c ../stats/hexp_expls/lime/bnn/q6_heart-c.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/heart-c/heart-c_test1/ -c ../stats/hexp_expls/lime/bnn/q6_heart-c.json -k ../rules/size/q6_heart-c_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/heart-statlog/heart-statlog_test1/ -c ../stats/hexp_expls/lime/bnn/q6_heart-statlog.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/heart-statlog/heart-statlog_test1/ -c ../stats/hexp_expls/lime/bnn/q6_heart-statlog.json -k ../rules/size/q6_heart-statlog_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/diabetes/diabetes_test1/ -c ../stats/hexp_expls/lime/bnn/q6_diabetes.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/diabetes/diabetes_test1/ -c ../stats/hexp_expls/lime/bnn/q6_diabetes.json -k ../rules/size/q6_diabetes_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/appendicitis/appendicitis_test1/ -c ../stats/hexp_expls/lime/bnn/q6_appendicitis.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/appendicitis/appendicitis_test1/ -c ../stats/hexp_expls/lime/bnn/q6_appendicitis.json -k ../rules/size/q6_appendicitis_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/schizo/schizo_test1/ -c ../stats/hexp_expls/lime/bnn/q6_schizo.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/schizo/schizo_test1/ -c ../stats/hexp_expls/lime/bnn/q6_schizo.json -k ../rules/size/q6_schizo_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/cleve/cleve_test1/ -c ../stats/hexp_expls/lime/bnn/q6_cleve.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/cleve/cleve_test1/ -c ../stats/hexp_expls/lime/bnn/q6_cleve.json -k ../rules/size/q6_cleve_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/penn-ml/biomed/biomed_test1/ -c ../stats/hexp_expls/lime/bnn/q6_biomed.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/penn-ml/biomed/biomed_test1/ -c ../stats/hexp_expls/lime/bnn/q6_biomed.json -k ../rules/size/q6_biomed_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q6/penn-ml/bupa/bupa_test1/ -c ../stats/hexp_expls/lime/bnn/q6_bupa.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q6/penn-ml/bupa/bupa_test1/ -c ../stats/hexp_expls/lime/bnn/q6_bupa.json -k ../rules/size/q6_bupa_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/penn-ml/glass2/glass2_test1/ -c ../stats/hexp_expls/lime/bnn/q6_glass2.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q6/penn-ml/glass2/glass2_test1/ -c ../stats/hexp_expls/lime/bnn/q6_glass2.json -k ../rules/size/q6_glass2_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/hepatitis/hepatitis_test1/ -c ../stats/hexp_expls/lime/bnn/q6_hepatitis.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/hepatitis/hepatitis_test1/ -c ../stats/hexp_expls/lime/bnn/q6_hepatitis.json -k ../rules/size/q6_hepatitis_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/heart-h/heart-h_test1/ -c ../stats/hexp_expls/lime/bnn/q6_heart-h.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q6/penn-ml/heart-h/heart-h_test1/ -c ../stats/hexp_expls/lime/bnn/q6_heart-h.json -k ../rules/size/q6_heart-h_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/emmanuel/cancer/cancer_test1/ -c ../stats/hexp_expls/lime/bnn/q5_cancer.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/emmanuel/cancer/cancer_test1/ -c ../stats/hexp_expls/lime/bnn/q5_cancer.json -k ../rules/size/q5_cancer_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/other/mlic/parkinsons_test1/ -c ../stats/hexp_expls/lime/bnn/q5_parkinsons.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/other/mlic/parkinsons_test1/ -c ../stats/hexp_expls/lime/bnn/q5_parkinsons.json -k ../rules/size/q5_parkinsons_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/other/mlic/pima_test1/ -c ../stats/hexp_expls/lime/bnn/q5_pima.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/other/mlic/pima_test1/ -c ../stats/hexp_expls/lime/bnn/q5_pima.json -k ../rules/size/q5_pima_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/other/zoo/zoo_test1/ -c ../stats/hexp_expls/lime/bnn/q5_zoo.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/other/zoo/zoo_test1/ -c ../stats/hexp_expls/lime/bnn/q5_zoo.json -k ../rules/size/q5_zoo_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q5/other/fairml/compas/compas_test1/ -c ../stats/hexp_expls/lime/bnn/q5_compas.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q5/other/fairml/compas/compas_test1/ -c ../stats/hexp_expls/lime/bnn/q5_compas.json -k ../rules/size/q5_compas_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q5/penn-ml/liver-disorder/liver-disorder_test1/ -c ../stats/hexp_expls/lime/bnn/q5_liver-disorder.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q5/penn-ml/liver-disorder/liver-disorder_test1/ -c ../stats/hexp_expls/lime/bnn/q5_liver-disorder.json -k ../rules/size/q5_liver-disorder_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/australian/australian_test1/ -c ../stats/hexp_expls/lime/bnn/q5_australian.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/australian/australian_test1/ -c ../stats/hexp_expls/lime/bnn/q5_australian.json -k ../rules/size/q5_australian_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/hungarian/hungarian_test1/ -c ../stats/hexp_expls/lime/bnn/q5_hungarian.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/hungarian/hungarian_test1/ -c ../stats/hexp_expls/lime/bnn/q5_hungarian.json -k ../rules/size/q5_hungarian_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/heart-c/heart-c_test1/ -c ../stats/hexp_expls/lime/bnn/q5_heart-c.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/heart-c/heart-c_test1/ -c ../stats/hexp_expls/lime/bnn/q5_heart-c.json -k ../rules/size/q5_heart-c_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/heart-statlog/heart-statlog_test1/ -c ../stats/hexp_expls/lime/bnn/q5_heart-statlog.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/heart-statlog/heart-statlog_test1/ -c ../stats/hexp_expls/lime/bnn/q5_heart-statlog.json -k ../rules/size/q5_heart-statlog_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/diabetes/diabetes_test1/ -c ../stats/hexp_expls/lime/bnn/q5_diabetes.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/diabetes/diabetes_test1/ -c ../stats/hexp_expls/lime/bnn/q5_diabetes.json -k ../rules/size/q5_diabetes_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/appendicitis/appendicitis_test1/ -c ../stats/hexp_expls/lime/bnn/q5_appendicitis.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/appendicitis/appendicitis_test1/ -c ../stats/hexp_expls/lime/bnn/q5_appendicitis.json -k ../rules/size/q5_appendicitis_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/schizo/schizo_test1/ -c ../stats/hexp_expls/lime/bnn/q5_schizo.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/schizo/schizo_test1/ -c ../stats/hexp_expls/lime/bnn/q5_schizo.json -k ../rules/size/q5_schizo_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/cleve/cleve_test1/ -c ../stats/hexp_expls/lime/bnn/q5_cleve.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/cleve/cleve_test1/ -c ../stats/hexp_expls/lime/bnn/q5_cleve.json -k ../rules/size/q5_cleve_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q5/penn-ml/biomed/biomed_test1/ -c ../stats/hexp_expls/lime/bnn/q5_biomed.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q5/penn-ml/biomed/biomed_test1/ -c ../stats/hexp_expls/lime/bnn/q5_biomed.json -k ../rules/size/q5_biomed_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q5/penn-ml/bupa/bupa_test1/ -c ../stats/hexp_expls/lime/bnn/q5_bupa.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q5/penn-ml/bupa/bupa_test1/ -c ../stats/hexp_expls/lime/bnn/q5_bupa.json -k ../rules/size/q5_bupa_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q5/penn-ml/glass2/glass2_test1/ -c ../stats/hexp_expls/lime/bnn/q5_glass2.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q5/penn-ml/glass2/glass2_test1/ -c ../stats/hexp_expls/lime/bnn/q5_glass2.json -k ../rules/size/q5_glass2_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/hepatitis/hepatitis_test1/ -c ../stats/hexp_expls/lime/bnn/q5_hepatitis.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/hepatitis/hepatitis_test1/ -c ../stats/hexp_expls/lime/bnn/q5_hepatitis.json -k ../rules/size/q5_hepatitis_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/heart-h/heart-h_test1/ -c ../stats/hexp_expls/lime/bnn/q5_heart-h.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q5/penn-ml/heart-h/heart-h_test1/ -c ../stats/hexp_expls/lime/bnn/q5_heart-h.json -k ../rules/size/q5_heart-h_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/emmanuel/cancer/cancer_test1/ -c ../stats/hexp_expls/lime/bnn/q4_cancer.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/emmanuel/cancer/cancer_test1/ -c ../stats/hexp_expls/lime/bnn/q4_cancer.json -k ../rules/size/q4_cancer_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/other/mlic/parkinsons_test1/ -c ../stats/hexp_expls/lime/bnn/q4_parkinsons.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/other/mlic/parkinsons_test1/ -c ../stats/hexp_expls/lime/bnn/q4_parkinsons.json -k ../rules/size/q4_parkinsons_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/other/mlic/pima_test1/ -c ../stats/hexp_expls/lime/bnn/q4_pima.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/other/mlic/pima_test1/ -c ../stats/hexp_expls/lime/bnn/q4_pima.json -k ../rules/size/q4_pima_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/other/zoo/zoo_test1/ -c ../stats/hexp_expls/lime/bnn/q4_zoo.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/other/zoo/zoo_test1/ -c ../stats/hexp_expls/lime/bnn/q4_zoo.json -k ../rules/size/q4_zoo_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q4/other/fairml/compas/compas_test1/ -c ../stats/hexp_expls/lime/bnn/q4_compas.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q4/other/fairml/compas/compas_test1/ -c ../stats/hexp_expls/lime/bnn/q4_compas.json -k ../rules/size/q4_compas_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q4/penn-ml/liver-disorder/liver-disorder_test1/ -c ../stats/hexp_expls/lime/bnn/q4_liver-disorder.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q4/penn-ml/liver-disorder/liver-disorder_test1/ -c ../stats/hexp_expls/lime/bnn/q4_liver-disorder.json -k ../rules/size/q4_liver-disorder_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/australian/australian_test1/ -c ../stats/hexp_expls/lime/bnn/q4_australian.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/australian/australian_test1/ -c ../stats/hexp_expls/lime/bnn/q4_australian.json -k ../rules/size/q4_australian_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/hungarian/hungarian_test1/ -c ../stats/hexp_expls/lime/bnn/q4_hungarian.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/hungarian/hungarian_test1/ -c ../stats/hexp_expls/lime/bnn/q4_hungarian.json -k ../rules/size/q4_hungarian_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/heart-c/heart-c_test1/ -c ../stats/hexp_expls/lime/bnn/q4_heart-c.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/heart-c/heart-c_test1/ -c ../stats/hexp_expls/lime/bnn/q4_heart-c.json -k ../rules/size/q4_heart-c_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/heart-statlog/heart-statlog_test1/ -c ../stats/hexp_expls/lime/bnn/q4_heart-statlog.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/heart-statlog/heart-statlog_test1/ -c ../stats/hexp_expls/lime/bnn/q4_heart-statlog.json -k ../rules/size/q4_heart-statlog_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/diabetes/diabetes_test1/ -c ../stats/hexp_expls/lime/bnn/q4_diabetes.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/diabetes/diabetes_test1/ -c ../stats/hexp_expls/lime/bnn/q4_diabetes.json -k ../rules/size/q4_diabetes_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/appendicitis/appendicitis_test1/ -c ../stats/hexp_expls/lime/bnn/q4_appendicitis.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/appendicitis/appendicitis_test1/ -c ../stats/hexp_expls/lime/bnn/q4_appendicitis.json -k ../rules/size/q4_appendicitis_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/schizo/schizo_test1/ -c ../stats/hexp_expls/lime/bnn/q4_schizo.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/schizo/schizo_test1/ -c ../stats/hexp_expls/lime/bnn/q4_schizo.json -k ../rules/size/q4_schizo_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/cleve/cleve_test1/ -c ../stats/hexp_expls/lime/bnn/q4_cleve.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/cleve/cleve_test1/ -c ../stats/hexp_expls/lime/bnn/q4_cleve.json -k ../rules/size/q4_cleve_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q4/penn-ml/biomed/biomed_test1/ -c ../stats/hexp_expls/lime/bnn/q4_biomed.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q4/penn-ml/biomed/biomed_test1/ -c ../stats/hexp_expls/lime/bnn/q4_biomed.json -k ../rules/size/q4_biomed_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q4/penn-ml/bupa/bupa_test1/ -c ../stats/hexp_expls/lime/bnn/q4_bupa.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/large/quantise/q4/penn-ml/bupa/bupa_test1/ -c ../stats/hexp_expls/lime/bnn/q4_bupa.json -k ../rules/size/q4_bupa_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q4/penn-ml/glass2/glass2_test1/ -c ../stats/hexp_expls/lime/bnn/q4_glass2.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/medium/quantise/q4/penn-ml/glass2/glass2_test1/ -c ../stats/hexp_expls/lime/bnn/q4_glass2.json -k ../rules/size/q4_glass2_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/hepatitis/hepatitis_test1/ -c ../stats/hexp_expls/lime/bnn/q4_hepatitis.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/hepatitis/hepatitis_test1/ -c ../stats/hexp_expls/lime/bnn/q4_hepatitis.json -k ../rules/size/q4_hepatitis_train1.csv_size5.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/heart-h/heart-h_test1/ -c ../stats/hexp_expls/lime/bnn/q4_heart-h.json
python ./bnns/explain.py -a check --load ./bnns/bnnmodels/small/quantise/q4/penn-ml/heart-h/heart-h_test1/ -c ../stats/hexp_expls/lime/bnn/q4_heart-h.json -k ../rules/size/q4_heart-h_train1.csv_size5.json
