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
6. Model recovery (optional)
7. Plot the figures in the main text

## 1) Install the required Python environment

Create a virtual environment
```
conda create --name ecpg python=3.10
```
Activate the environment
```
conda activate ecpg
```
Install the dependencies
```
cd ecpg
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
* `fits`: the model fitting and model recovery results.
* `analyses`: the analyses results. 

## 3) Preprocessing

The preprocessing primarily transforms raw data into an analyzable format, including variable transformation, data cleaning, and filling missing values. This script also screens for valid participants.

Bash commands:
```
python m0_preprocess.py -d='exp1' -m='mle'
```
Run `-d='exp2'` for Experiment 2.

## 4) Model simulations

The model simulation section illustrates the behavior of models under various sets of free parameters. In our simulation, we focused on examining the behaviors of the ECPG model, but you are also welcome to explore other models using the following bash commands.

The is the perequiste for producing Fig.2 (`visualization/Fig2`) and Fig.4 (`visualization/Fig4`). If your have downloaded the `simulations` folders, skip the step 4.

Run Bash commands:
```
for lmbda in 0 .07 .1 .2 .5; do
    python m1_predict.py -n='ecPG_sim' -d='exp1' -c=50 -p="[40, 8, $lmbda]"
done
```
and 
```
for lmbda in 0 .07 .1 .2 .5; do
   python m1_predict.py -n='ecPG_fea_sim' -d='exp2' -c=50 -p="[11, 4, $lmbda]"
done
python m1_predict.py -n='ACL' -d='exp2' -c=50 -p="[.65, 10, 0.04, .6]"
```
You will obtain a folder called `simulations` with your simulated data inside. 

## 5) Model fitting

The model fitting section is crucial and the most time-consuming part of the project, as it forms the foundation for model comparison and model recovery analysis. You can reproduce the model fitting results by running the code below, though it requires significant time — approximately 4 days on a >50-core computer. We recommend downloading our results from OSF to expedite result reproduction.

The is the perequiste for producing Fig.3-7 (`visualization/Fig3`-`visualization/Fig7`). If your have downloaded folders `fits`, `simulations`, `analyses`, skip the step 5.

The Bash commands are: 
```
for data_set in "${data_sets[@]}"; do

    case "$data_set" in
        "exp1") models=("rmPG" "caPG" "ecPG" "LC" "MA" "l1PG" "l2PG" "rndPG" "dcPG");;
        "exp2") models=("rmPG_fea" "caPG_fea" "ecPG_fea" "LC" "MA" "ACL" "l1PG_fea" "l2PG_fea" "rndPG_fea" "dcPG_fea");;
    esac

    ## step 2: fit models to data, simulate, and analyze
    for model in "${models[@]}"; do  
        echo Data set=$data_set Model=$model Method=$fit_method Algorithm=$alg
            python m2_fit.py      -d $data_set -n $model -s 420 -f 50 -c 50 -m $fit_method -a $alg
            python m3_simulate.py -d $data_set -n $model -s 422 -f 10 -c 20 -m $fit_method -a $alg
            python m4_analyze.py  -d $data_set -n $model -m $fit_method 
    done

done
```
You will obtainfolders `fits`, `simulations`, `analyses`.

## 6) Model recovery

The model recovery process closely resembles the model fitting process, reusing much of the same code. The primary difference is that, in model recovery, models are fitted to synthesized data. This step is even more computationally intensive than model fitting, requiring approximately 12 days on a computer with more than 50 cores. We highly recommend downloading the fits folder from OSF.

## 7) Plot the figures in the main text

After downloading or running the code to obtain all prerequisite folders, please place them within this repository.

```
- ECPG (this github repo)
|- analyses
|- data
|- fits
|- simulations
|- visualization
|- utils
```

Run `visualization/Fig#` in the visaulization folder to produce the figures you preferred.
 









