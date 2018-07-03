# Semester project - Galaxy Image Generation 
## Computational Intelligence Lab - ETH Zurich, Spring 2018
#### Mélanie Bernhardt - Laura Manducchi - Mélanie Gaillochet

In this project we build a 2-in-1 model that can generate images of galaxies observed by astronomical telescopes, and predict the similarity score of a set of query images with the same model.
The scientific report associated to this project can be found in this repository (see `report.pdf`). This report describes the goal of the project, the model and the experiments we performed.

## Training Data available for this project:
* 9600 scored images according to their similarity to the concept of a prototypical 'cosmology image' - number from 0.0 to 8.0, with 0.0 indicating a very low similarity. Most of the training images have a score under 3, none of them has a score above 6.5.  
* 1200 labeled images containing galaxy images (label = 1) and corrupted/irrelevant images (label = 0).

## How to use the code ?
### Set up your environment
NOTE : our code is working with Python 3.6 we assume that is the version installed on your computer.

IMPORTANT : keep the exact structure of the code folder as submitted and to place the data and runs folder in this main folder (as indicated in step 1 of the following set-up procedure).

Before using our code, please follow this procedure:
* Download the `data` folder and the `runs` folder from polybox (under: https://polybox.ethz.ch/index.php/f/1032408684). The data folder contains all 3 subfolders containing the training images as well as the csv file containing the associated labels and scopres. The runs folder contains all fitted models (to avoid recomputation for reproducing the results).
* Place them in the root folder `CIL2018` (keeping the structure). And set the root folder `CIL2018` as current directory.
* Run the pip requirement file to get all necessary packages for the project using `pip3 install -r requirements.txt`

### Reproducing the report results on the development set
In order to make it easier for the reader to reproduce the results on our development set presented in the results section of our article we created 2 files: 
 * `reproduce_model_results_dev.py`: 
    - Simply run this file to get the MAE score on the development set for our final trained 2-in-1 model as well as for the discriminator trained alone. Results are printed to the console.
 * `reproduce_baseline_results_dev.py`:
    - Simply run this file to get the MAE score on the development set for our two baselines. Results are printed to the        console.
 
 ### Reproducing the csv files for submission to Kaggle competition (test set)
 To produce the csv file to submit to Kaggle you can use:
 * `kaggle_prediction_model_results.py`: 
    - Running this file creates a csv file containing the predicted score associated to a particular training run of our final                  model. This file can also be used to create a csv file with the predictions associated to a particular training run of the model that trains the score discriminator alone. 
        * To choose the training model for which you want to predict the score, please specify `run_number` (name of the corresponding subfolder containing the checkpoints) and `model_number` (specify the number of the `.meta` file) in the 2 first lines of the file. 
        * Parameter to use to predict from our final model are:
            - `run_number: 1530273051`
            - `model_number: 22088`
        * Parameter to use to predict from the model which trains the score discriminant alone are:
            - `run_number: TO COMPLETE TOMORROW`
            - `model_number: TO COMPLETE TOMORROW`
    - Results are placed in the predictions subfolder of the root folder, the train run_number is the name of the csv output file.
 * `kaggle_prediction_baseline.py`:
    - Running this file creates a csv file containing the predicted score associated to a particular baseline (as described in the final report).
       * Parameters to specify are:
    - Results are placed ....
            - place here the parameters once i have the final file.
           
 ### Generating images
 To generate images for our final 2-in-1 model simply run `produce_images.py`. 
 This file launches the image generation from a saved checkpoint file and save those whose score is above 2.5.
 
 If you would like to use your own trained model just change `run_number` and `model_number` as in the previous sections.
 The produced images are placed in the "/produced/run_number/" subfolder of the current directory.
 ### Training the model from scratch
 If you wish to re-train completely our 2-in-1 model instead of using the provided saved checkpoints you can use `train_2in1_model.py`:
 * if you run it as is you will relaunch the exact same training procedure and parameters we used to build our saved final model (these are the default number of epochs, checkpoints, generator threshold training, batch size etc...)
 * you can modify the parameters you want to test other training procedure in the first section of this file.
 
 If you wish to re-train the score discriminator alone (without any generator or label discriminator) you can run `train_score_discr_alone.py`. Again running this file as it corresponds to the training we used for the model used to get the experiments results in the paper. You can modify the training procedure by modifying the parameters section at the beginning of the file. 
 
