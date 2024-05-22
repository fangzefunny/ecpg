#!/bin/bash

## delete the DS.Store
find . -name ".DS_Store" -delete

declare data_sets=("exp2") #"exp1" "exp2" "exp2-benchmark"
declare fit_method='map'
declare alg='BFGS'

for data_set in "${data_sets[@]}"; do
 
    ## step 0: preprocess the data  
    #python m0_preprocess.py -d=$data_set -m=$fit_method

    # ## step 1: simulate to show the model behaviors
    # if [ "$data_set" = "exp1" ]; then
    #     for lmbda in 0 .05 .1 .2 .5; do
    #         python m1_predict.py -n='ecPG_sim' -d='exp1' -c=50 -p="[40, 8, $lmbda]"
    #     done
    # else
    #     for lmbda in 0 .05 .1 .2 .5; do
    #         python m1_predict.py -n='ecPG_fea_sim' -d='exp2' -c=50 -p="[11, 4, $lmbda]"
    #     done
    #     python m1_predict.py -n='ACL' -d='exp2' -c=50 -p="[.65, 10, 0.04, .6]"
    # fi
    

    ## step 2: fit models to data
    case "$data_set" in
        # case 1
        "exp1") declare models=("ecPG");; #"rmPG" "caPG" "ecPG" "l2PG"
        # case 2 
        "exp2") declare models=("LC");; #  "LC" "ACL"   "l2PG_fea"  "caPG_fea"
    esac 
    for model in "${models[@]}"; do  
        echo Data set=$data_set Model=$model Method=$fit_method Algorithm=$alg
            python m2_fit.py      -d=$data_set -n=$model -s=420 -f=50 -c=50 -m=$fit_method -a=$alg
            python m3_simulate.py -d=$data_set -n=$model -s=422 -f=10 -c=20 -m=$fit_method -a=$alg
            #python m4_analyze.py  -d=$data_set -n=$model -m=$fit_method 
    done
done
