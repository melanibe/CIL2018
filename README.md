# Computational Intelligence Lab - ETH Zurich, Spring 2018
## Project - Galaxy Image Generation

The goal is to train a generative model that can generate images of galaxies observed by astronomical telescopes, and predict the similarity score of a set of query images.

### Training Data: <br/>
* 9600 scored images according to their similarity to the concept of a prototypical 'cosmology image' - number from 0.0 to 8.0, with 0.0 indicating a very low similarity <br/>
* 1200 labeled images

### Evaluation Metrics: <br/>
Similarity score prediction (MAE i.e. Mean Absolute Error)
