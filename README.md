# Semester project - Galaxy Image Generation
## A “2-in-1” deep neural model for generating and scoring galaxy images
### Computational Intelligence Lab - ETH Zurich, Spring 2018
#### Mélanie Bernhardt - Laura Manduchi - Mélanie Gaillochet

In this project we build a 2-in-1 model that can generate images of galaxies observed by astronomical telescopes, and predict the similarity score of a set of query images with the same model.
The scientific report associated to this project can be found in this repository (see `report.pdf`). This report describes the goal of the project, the model and the experiments we performed.

#### Training Data available for this project:
* 9600 scored images according to their similarity to the concept of a prototypical 'cosmology image'. Score ranges from 0.0 to 8.0, with 0.0 indicating a very low similarity. Most of the training images have a score under 3, none of them has a score above 6.5.  
* 1200 labeled images containing galaxy images (label = 1) and corrupted/irrelevant images (label = 0).

The associated Kaggle Competition for the score prediction task can be found under: https://www.kaggle.com/c/cil-cosmology-2018

## Set up your environment to run our code
_NOTE_: our code is working with Python 3.6 we assume that is the version installed on your computer.

_IMPORTANT_: keep the exact structure of the code folder as submitted and place the `data`, `runs`; `produced`, `predictions` and `baselines_models` folders in this main folder (as indicated in step 1 of the following set-up procedure).

Before using our code, please follow this procedure:
* Download the `data` folder, the `runs` folder, the `produced` folder, the `predictions` folder and the `baselines_models` folder from polybox (under: https://polybox.ethz.ch/index.php/s/AViHe1MXIJSVnzB). The `data` folder contains all 3 subfolders containing the training images as well as the csv file containing the associated labels and scores. The `runs` folder contains all fitted models (to avoid recomputation for reproducing the results).The `baselines_models` contain the saved fitted estimators for the baselines. Please download everything (even if you think you don't need it - all subfolders have to exist in order to run the code smoothly).
* Place all 5 folders in the root folder of this repository `CIL2018` (keeping the structure). And set this root folder `CIL2018` as your current directory.
* Run the pip requirement file to get all necessary packages for the project using `pip3 install -r requirements.txt`

## Reproducing the report results on the validation set
In order to make it easier for the reader to reproduce the results on our validation (aka development) set presented in the results section of our article we created 2 files (run time max. 10 min): 
 * `model_reproduce_results_dev.py`: 
    - Simply run this file to get the MAE score on the development set for our final trained 2-in-1 model. Results are printed to the console.
 * `baseline_reproduce_results_dev.py`:
    - Simply run this file to get the MAE score on the development set for our two baselines. Results are printed to the        console. It used the 2 saved estimators located in the `baselines_models` subfolder (provided in the polybox).
    
 ## Reproducing the csv files for submission to Kaggle competition (test set)
 To produce the csv file to submit to Kaggle you can use (run time max. 10 min):
 * `model_kaggle_prediction.py`: 
    - Running this file creates a csv file containing the predicted score associated to a particular training run of our final                  model.
        * To choose the training model for which you want to predict the score, please specify `run_number` (name of the corresponding subfolder containing the checkpoints) and `model_number` (specify the number of the `.meta` file) in the 2 first lines of the file. 
        * Parameter to use to predict from our final model are:
            - `run_number: 1530273051`
            - `model_number: 22088`
    - Results are placed in the predictions subfolder of the root folder, the train run_number is the name of the csv output file.
 * `baseline_kaggle_prediction.py`:
    - This file is used to output the predictions from the baselines on the query dataset for Kaggle. It can also be run to output the prediction on any custom dataset (for example to check the predicted score on the generated images).
It assumes that you have trained your baseline estimator first and saved it to the `baselines_models` subfolder. The `baselines_models` downloaded from the polybox during set up of your environment already contains the saved estimators used for the report.
    - You have to enter the:
        * `model name`: 'Ridge' and 'RandomForest' for the model to build to use for prediction.
        * `feat_size`: the number of features that were used for training (we used 10).
        * `train_ratio`: the train/dev split ratio (we used 0.99 training split).
        * `folder`: the folder were the images to evaluate are located. By default it is the query images set.
    - Results are placed in the `predictions` subfolder. The filename is `[name of model]_[number of features]_query_pred.csv` if calculated on the query dataset, `[name of model]_[number of features]_trial_pred.csv` if calculated on another custom dataset.
    - If you run this file on the folder of the generated images (instead on the query images folder) you will get the mean score and standard error of the score mentioned in the report printed to the console.
           
 ## Generating images
 To generate images for our final 2-in-1 model simply run `produce_images.py`. 
 This file launches the image generation from a saved checkpoint file and save those whose score is above 3.0 (run time max. 5 mins).
 
 If you would like to use your own trained model just change `run_number` and `model_number` as in the previous sections.
 The produced images are placed in the `/produced/run_number/` subfolder of the current directory.
 
 ## Training the model from scratch
 If you wish to re-train completely our 2-in-1 model instead of using the provided saved checkpoints you can use `train_2in1_model.py`:
 * if you run it as it is, you will relaunch the exact same training procedure and parameters we used to build our saved final model (these are the default number of epochs, checkpoints, generator threshold training, batch size etc...)
 * you can modify the parameters you want to test other training procedure in the first section of this file.

Note: training the model with the default parameters on 1 GPU on the ETH Leonhard cluster took approximately 3 hours.
 
 
 ## Training the baselines from scratch
 If you wish to re-train completely our baseline estimators instead of using the provided saved estimators you can use `baseline_train.py`.
 The parameters to enter are:
 * `model`: you can choose between 'Ridge' and 'RandomForest' for the model to build.
 * `feat_size`: number of features to use for the feature matrix (it defaults to 10 parameter used for the
experiments in the report)
 * `train_ratio`: train/dev split ratio (it defaults to 0.99% training split, parameter used for the
experiments in the report).

If the preprocessed feature matrix does not exist for the input parameters it will first compute the features matrix (this takes 10 min, grab a coffee), if this matrix was already saved previously it just loads it.
It then fits the chosen estimator (another 10 mins if RandomForest, go for a cup of tea this time) and saves it the the `baselines_models` subfolder.
The MAE on the dev set will be printed to the console at the end of the run.

## Calculating descriptive statistics on the SCORED training dataset
The report mentions some descriptive statistics on the SCORED training dataset (20% percentile, mean score, standard deviation). The results can be retrieved by running `descriptive_stats_scored_data.py`,
 If you wish to re-train completely our baseline estimators instead of using the provided saved estimators you can use `baseline_train.py`, statistics are printed to the console.
 If you wish to re-train completely our baseline estimators instead of using the provided saved estimators you can use `baseline_train.py`.
