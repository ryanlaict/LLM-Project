# LLM Project

## Project Task
This project task was to create a sentiment analysis of Yelp reviews based on the Yelp dataset: https://huggingface.co/datasets/Yelp/yelp_review_full. The goal was to be able to correctly identify the sentiment of the Yelp review based on the text of the review. 

## Dataset
The dataset used is the Yelp Review Full dataset from Hugging Face Datasets. It contains 650,000 reviews for training and 50,000 reviews for testing. Each review is labeled with a star rating from 1 to 5. The dataset was cleaned and processed. A subset of the dataset was created for testing purpose with 10,000 rows. The CSV was created and saved for faster recall. The Train/Test splits were stratified based on labels to ensure that there was a fair distribution of types of reviews for the model to train against and that the subset of the data did not unfairly bias one type of review. 

## Pre-trained Model
The Pre-trained model that was utilized in this project was distilbert-base-uncased. It is lighter version of Bert which is important due to resource limitations in this project. It was still able to accurately model the dataset and create labels while balancing resourcing demands.

## Performance Metrics
To accurately assess the performance of this model. The metrics of Accuracy, Recall, and F1-Score were used in combination to evaluate the performance of each model. The Bert's performance was adequate before tuning the parameters. 

After tuning and performing a cross-validation with 5 K-folds, the final results for the model were: 

Accuracy: 0.9077
Recall: 0.8986
F1: 0.9066

Accuracy was used to evaluate the overall prediction of the model and the combination of Recall and F1 were used to ensure a balanced performance. The high scores confirmed that the model was accurately making predictions and it was balanced. 
## Hyperparameters
During the tuning process of this project, the following training parameters were used: 

  - learning_rate: 1e-05
  - train_batch_size: 8
  - eval_batch_size: 8
  - gradient_accumulation_steps: 2
  - total_train_batch_size: 16
  - num_epochs: 2

The reason for these choices were to reduce the resource demands of the training process while still maintaining reasonable results from the model. Future steps will be to increase the training size of the dataset and increasing the parameters to be more detailed to build a more robust model. 


