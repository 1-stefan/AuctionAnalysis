To watch the video presentation, click the following YouTube link:

[https://youtu.be/z8RUKotE0fk]


This repo contains the instructions for a machine learning project.

Abstract:

Title: Auction Price Prediction

The Purpose of this project is to build a prediction model to predict the closing price of an auction on Ebay. The dataset used is a list of auctions for 3 types of items, Cartier watches, Xbox Consoles, and Palm Pilot digital assistants. The auctions data consists of the closing price, along with accompanying info about the auctioned item such as the id, bids, bidding time, open bid, etc. We want to determine the closing price as well as an investigation on what features contribute to a high selling price for sellers to maximize revenue and for buyers to get a good price.

By definition, I am predicting a continuous value, so this is a regression problem. For this, I will explore some machine learning techniques that can predict a continuous variable such as Linear Regression (or Polynomial Regression if necessary), KNN regressor, Random Forest regressor, and a support vector regressor. For model performance I will use hyperparameter tuning for KNN, Random Forest, and SVM through a grid search. Finally I will select the best performing model for final use.

Continuing this, I will create a classification problem to predict the type of item based on information of the auction. This is important because when creating an automated bidder in the future I want to be able to identify the type of items I am bidding on. This will be done in the same process of tuning as above, but with different techniques such as Logistic Regression, KNNClassifier, RFclassifier and a Support vector classifier. Tuning will be done on all 4 of these to show the difference in the models as in the previous regression problem.

The purpose of this project is for myself to get a better understanding on the feasability of prediction models for auction bidding. This is an interest of mine since I often use Yahoo Japan Auctions to bid on and resell items from. Given the results of this project I plan to expand my ideas here to apply to my personal endeavours.

Project Organization
All of the project can be executed within the 'src' section. However, for formatting reasons, copies of files were copied in the appropriate sections as per instructions. Py modules that were not used have been mentioned below.

To run project from top to bottom (my way), start with pre-processing.ipynb in src, and then run main.ipynb to test and visualize all models and selection. Please read the below to see how this project is organized.

├── LICENSE
├── README.md          <- The top-level README for describing highlights for using this ML project.
├── data
│   ├── external       <- Data from third party sources. (The original data I used from kaggle)
│   ├── interim        <- (NOT USED) Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling. Contains 3 datasets: cleaned_auction (original processed data), then two new datasets stem 
|                         from this to be used for classification and regression, respectfully
│   └── raw            <- (NOT USED) The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
|└── processing.ipynb <- my version of the pre-processing.py file which contains exploratory data analysis and
|                        does all preprocessing of data and sends a copy of the cleaned dataset to the processed directory (another copy exists in src)
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            
│   └── figures        <- Generated graphics and figures to be used in reporting
│   └── README.md      <- Youtube Video Link
│   └── final_project_report <- final report .pdf format and supporting files
│   └── presentation   <-  final power point presentation 
|
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
   ├── __init__.py    <- (NOT USED) Makes src a Python module <- 
   │
   │
   ├── features       <- (NOT USED) Scripts to turn raw data into features for modeling
   │   └── build_features.py
   │
   ├── models         <- Scripts to train models and then use trained models to make
   │   │               predictions
   │   |
   │   └── train_model_regression.py <-           Separate regression problem and classification problem since models are different
   |   ├── train_model_classification.py <-       All training and prediction is done within these two py files
   │
   └── visualization  <- Scripts to create exploratory and results oriented visualizations
   |   └── visualize.py  <- contains classes used for visualization of model accuracy 
   |
   └── main.py    <- where testing of all models occurs and model selection is done along with visualization (run time is ~25 min)
   └── main.ipynb   <- same as above but uses relative path in folder for data reading (read_csv), (above uses relative os path) 
   |                   Easier to visualize graphs at the end and see all results using jupyter chunks (your preference)
   └── pre-processing.ipynb    <- same as processing.ipynb 
