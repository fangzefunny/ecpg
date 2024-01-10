# Efficient Coding Policy Gradient

Code for the paper "learning generalizable representations through efficient coding"
This Github repository will also be stored within an Open Science Framework (OSF) repository: https://osf.io/uctdb/

Here is the guideline of reproducing the main figures in the paper. 

# Steps for Reproducing the Main Analyses
1. Install the required Python environment
2. Download and unzip the saved data and model fitting results
3. Preprocessing
4. Model simulations
5. Model fitting (optional)
6. Plot the figures in the main text

## 1) Install the required Python environment

Create a virtual environment and install the dependencies.

Bash commands:
```
conda create --name ecpg python=3.8
conda activate ecpg
pip install -r requirements.txt
```
You can use the command `bash run.sh` in the terminal to complete all modeling and analysis, or run each section step-by-step following the instructions below.

Note that the main files are named in the format of `m#_xxxx.py`. The number after 'm' indicates the order of the scripts. Please run the main script in order, from 0 to 4. 

## 2) Download and unzip the saved data and model fitting results

Downloaded from the OSF (https://osf.io/uctdb/) zip files `data`, `simulations`, `fits`, and `analyses`, and unzip them under the main folder `ecpg/`.

Required (must be downloaded and unzipped before being able to run the code): 
* `data`: contains the experiment data collected online. 

Optional (highly recommand to downlaod if you plan to run the scripts on your local computer, all scripts require parallel computing): 
* `simulations`: the model simulations output.
* `fits`: the model fitting results.
* `analyses`: the analyses results. 

## 3) Preprocessing

Bash commands:
```
python m0_preprocess.py -d='exp1' -m='mle'
```
Run `-d='exp2'` for Experiment 2.

## 4) Model simulations

The is the perequiste for producing Fig.2 (`visualization/Fig2`) and Fig.4 (`visualization/Fig4`). If your have downloaded the `simulations` folders, skip the step 4.

Run Bash commands:
```
for lmbda in 0 .05 .1 .2 .5; do
    python m1_predict.py -n='ecPG_sim' -d='exp1' -c=50 -p="[40, 8, $lmbda]"
done
```
and 
```
for lmbda in 0 .05 .1 .2 .5; do
   python m1_predict.py -n='ecPG_fea_sim' -d='exp2' -c=50 -p="[11, 4, $lmbda]"
done
python m1_predict.py -n='ACL' -d='exp2' -c=50 -p="[.65, 10, 0.04, .6]"
```
You will obtain a folder called `simulations` with your simulated data inside. 

## 5) Model fitting

The is the perequiste for producing Fig.3-7 (`visualization/Fig3`-`visualization/Fig7`). If your have downloaded folders `fits`, `simulations`, `analyses`, skip the step 5.

```
declare data_sets=("exp1" "exp2")  #
declare fit_method='mle'
declare alg='Nelder-Mead'

for data_set in "${data_sets[@]}"; do

    case "$data_set" in
        # case 1
        "exp1") declare models=("rmPG" "caPG" "ecPG");; 
        # case 2 
        "exp2") declare models=("rmPG_fea" "caPG_fea" "ecPG_fea" "LC" "ACL");; # 
    esac 
    for model in "${models[@]}"; do  
        echo Data set=$data_set Model=$model Method=$fit_method
            python m2_fit.py      -d=$data_set -n=$model -s=42 -f=50 -c=50 -m=$fit_method -a=$alg
            python m3_simulate.py -d=$data_set -n=$model -s=42 -f=10 -c=20 -m=$fit_method -a=$alg
            python m4_analyze.py  -d=$data_set -n=$model -m=$fit_method 
    done
done
```
You will obtainfolders `fits`, `simulations`, `analyses`.

6. Plot the figures in the main text

Run `visualization/Fig#` to produce the figures you preferred.
 









