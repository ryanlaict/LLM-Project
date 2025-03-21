# LLM Project

## Project Task
This project task was to create a sentiment analysis of Yelp reviews based on the Yelp dataset: https://huggingface.co/datasets/Yelp/yelp_review_full. The goal was to be able to correctly identify the sentiment of the Yelp review based on the text of the review. 

## Dataset
The dataset used is the Yelp Review Full dataset from Hugging Face Datasets. It contains 650,000 reviews for training and 50,000 reviews for testing. Each review is labeled with a star rating from 1 to 5. The dataset was cleaned and processed 

## Pre-trained Model
The Pre-trained model that was utilized in this project was distilbert-base-uncased. It is lighter version of Bert which is important due to resource limitations in this project. It was still able to accurately model the dataset and create labels while balancing resourcing demands. 

## Performance Metrics
To accurately assess the performance of this model. The metrics of Accuracy, Recall, and F1-Score were used in combination to evaluate the performance of each model. The Bert's performance was adequate before tuning the parameters with a 

## Hyperparameters
- Learning Rate was 1e-5: A lower learning rate helped stabilize training and avoid overshooting during fine-tuning. It was also used as it kept the resource demands low while not        afffecting the accuracy scores noticably.
- Batch Size were kept beetween 16 and 32: Balanced training speed and memory usage.
- Epochs 2-3): Enough for convergence without overfitting and for maintaining relatively low resource demands
- The max sequence length used was 512. 

