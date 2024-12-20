#!/bin/bash

## delete the DS.Store
find . -name ".DS_Store" -delete

declare data_sets=("exp1" "exp2") 
declare fit_method='map'
declare alg='BFGS'

for data_set in "${data_sets[@]}"; do

    case "$data_set" in
        "exp1") models=("rlPG" "caPG" "ecPG" "LC" "MA" "l1PG" "l2PG" "rndPG" "dcPG");;
        "exp2") models=("rlPG_fea" "caPG_fea" "ecPG_fea" "LC" "MA" "ACL" "l1PG_fea" "l2PG_fea" "rndPG_fea" "dcPG_fea");;
    esac
 
    ## step 0: preprocess the data  
    python m0_preprocess.py -d=$data_set -m=$fit_method

    ## step 1: simulate to show the model behaviors
    if [ "$data_set" = "exp1" ]; then
        for lmbda in 0 .05 .07 .1 .2 .5; do
            python m1_predict.py -n 'ecPG_sim' -d 'exp1' -c 50 -p "[40, 8, $lmbda]"
        done
    else
        for lmbda in 0 .05 .07 .1 .2 .5; do
            python m1_predict.py -n 'ecPG_fea_sim' -d 'exp2' -c 50 -p "[11, 4, $lmbda]"
        done
    fi

    ## step 2: fit models to data, simulate, and analyze
    for model in "${models[@]}"; do  
        echo Data set=$data_set Model=$model Method=$fit_method Algorithm=$alg
            python m2_fit.py      -d $data_set -n $model -s 420 -f 50 -c 50 -m $fit_method -a $alg
            python m3_simulate.py -d $data_set -n $model -s 422 -f 10 -c 20 -m $fit_method -a $alg
            python m4_analyze.py  -d $data_set -n $model -m $fit_method 
    done

    # step 3: model recovery for validation
    if [ "$data_set" = "exp2" ]; then
        models=("rlPG_fea" "caPG_fea" "ecPG_fea" "LC" "MA" "ACL" "l1PG_fea" "l2PG_fea") 
        for model in "${models[@]}"; do
            echo Model recovery... for $model
            python m5_recover.py -d "exp2" -n $model -m $fit_method -a $alg -o "${models[@]}" -c 10 -s 423 
        done
    fi

done
