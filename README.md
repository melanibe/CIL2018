# Computational Intelligence Lab - ETH Zurich, Spring 2018

## Project - Galaxy Image Generation
In this project we build a 2-in-1 model that can generate images of galaxies observed by astronomical telescopes, and predict the similarity score of a set of query images with the same model.
The scientific report associated to this project can be found in this repository (see report.pdf). This report describes the goal of the project, the model and the experiments we performed.

## Training Data available for this project: <br/>
* 9600 scored images according to their similarity to the concept of a prototypical 'cosmology image' - number from 0.0 to 8.0, with 0.0 indicating a very low similarity <br/>
* 1200 labeled images

## How to use the code ?

### Reproducing the report results on the development set
In order to make it easier for the reader to reproduce the results on our development set presented in the results section of our article we created 2 files: 
 * reproduce_neural_results_dev: 
    - This file 
 * reproduce_baseline_results_dev:
 
 ### Reproducing the csv files for submission to Kaggle competition (test set)
 To produce the csv file to submit to Kaggle you can use:
 - kaggle_prediction_neural_results: 
 - kaggle_prediction_baseline:
