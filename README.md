# Efficient Coding Policy Gradient

Code for the paper "learning generalizable representations through efficient coding"
This Github repository will also be stored within an Open Science Framework (OSF) repository: https://osf.io/uctdb/

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
`data`: contains the experiment data collected online. 

Optional (highly recommand to downlaod to save your time): 
`simulations`: the model simulations output.
`fits`: the model fitting results.
`analyses`: the analyses results. 

## 3) Preprocessing

Bash commands:
```
python m0_preprocess.py -d='exp1' -m='mle'
```
Run `-d=exp2` for Experiment 2.

## 3) Model simulations

If your have downloaded the `simulations` folders, skip the step 3.1.

Step 3.1: run Bash commands:
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
```
You will obtain a folder called `simulations` with your simulated data inside. 

Step 3.2:
Run `visualization/Fig2` to visualize your simulated results. This is correspond to the Fig.2 in the main text. 








