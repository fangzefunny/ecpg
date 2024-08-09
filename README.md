# Individuals with anxiety and depression use atypical decision strategies in an uncertain world

Here is the guideline of reproducing the main figures in the paper. 

# Steps for Reproducing the Main Analyses
1. Install the required Python environment
2. Preprocessing
3. Model fitting (optional)
4. Model simulation 
5. Plot the figures in the main text

## 1) Install the required Python environment

Git clone the repository.

Create a virtual environment
```
conda create --name mos python=3.10
```
Activate the environment
```
conda activate mos
```
Install the dependencies
```
cd mos
pip install -r requirements.txt
```

## 2) Preprocessing

Run the script 

```
python m0_preprocess.py
```
to preprocess the data to include the necessary variables for model fitting and analysis. 

Note that the main files are named in the format of `m#_xxxx.py`. The number after 'm' indicates the order of the scripts. Please run the main script in order, from 0 to 4. 

You can use the command `bash run.sh` in the terminal to complete all modeling and analysis. 

## 3) Model fitting (optinal)

To estimate the parameters for each participant, please run
```
python m1_fit.py -d="exp1" -n=$model -s=420 -f=40 -c=40 -m="map" -a="BFGS"
```
to fit any model you like. Replace the `$model` with the name of the model you want to fit. 

You can fit the model by yourself or download our the fitting results from xxxxx.  


## 4) Model simulation

To see the model behaviors, run
```
python m2_rollout.py -d="exp1" -n=$model -f=20 -c=20 -m="map" -a="BFGS"
```
to generate a .csv file that record 

## 5). Reproduce the figures in the paper

Find `visualization/Fig#` to produce the figures you like.
 









