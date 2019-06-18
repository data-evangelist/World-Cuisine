# World-Cuisine
A machine learning model using Python and Weka to predict cuisine types based on ingredients

# The Dataset
Part of the Kaggle.com “What’s Cooking” challenge (https://www.kaggle.com/c/whats-cooking)
Two JSON files, a training set and a test set
Number of records: 39,774 
Number of attributes: 3 
id, 
cuisine, 
list of Ingredients
No missing values, but many outliers (because many ingredients are only associated with one recipe)

# Goal
Classify each recipe by cuisine based on the ingredients used in the recipe
(e.g., tomato sauce, pasta, and basil may indicate the recipe is Italian)

# Data Preprocessing
Generated five CSVs, each containing one of the following ingredient subsets:
20 most common ingredients across all cuisines,
20 random ingredients, 
20 most common ingredients combined with the previously selected 20 random ingredients.
5 most common ingredients from each cuisine
10 most common from each cuisine, minus the 20 most common overall
Converted CSVs to ARFF files:
top20.arff
rand20.arff
top20andrand20.arff
top_from_each.arff
top_from_each_minus_common.arff

# Classification
Algorithms used:
NaiveBayes,
J48,
OneR
Results of each algorithm are compared for:
Training set,
10-fold cross validation,
Percentage split (66%)

## Classification Results : top20.arff

Results of Test Options - 

### NaiveBayes: 
Training Set 31.45 % |
Percentage Split (66%) 30.68 % |
10-Fold Cross Validation 31.34 % 

### J48:
Training Set 52.38 % |
Percentage Split (66%) 27.69 % |
10-Fold Cross Validation 28.60 % 

### OneR 
Training Set 24.62 % |
Percentage Split (66%)16.58 % |
10-Fold Cross Validation 16.83 % 

## Classification Results: rand20.arff

Results of Test Options 

### NaiveBayes 
Training Set 25.02 % |
Percentage Split (66%) 24.79 % |
10-Fold Cross Validation 24.95 % 
### J48 
Training Set 50.18 % |
Percentage Split (66%) 20.37 % |
10-Fold Cross Validation 20.44 % 
### OneR 
Training Set 24.62 % |
Percentage Split (66%) 16.58 % |
10-Fold Cross Validation 16.83 % 

## Classification Results: top20andrand20.arff

Results of Test Options 

### NaiveBayes 
Training Set 32.88 % |
Percentage Split (66%) 32.03 % |
10-Fold Cross Validation 32.68 % 
### J48 
Training Set 54.11 % |
Percentage Split (66%) 30.23 % |
10-Fold Cross Validation 30.79 % 
### OneR 
Training Set 24.62 % |
Percentage Split (66%) 16.58 % |
10-Fold Cross Validation 16.83 % 

## Classification Results: top_from_each.arff

Results of Test Options 

### NaiveBayes 
Training Set 44.71% |
Percentage Split (66%) 44.75% |
10-Fold Cross Validation 44.54%
### J48 
Training Set 49.44% |
Percentage Split (66%) 44.99% |
10-Fold Cross Validation 45.44%
### OneR 
Training Set 23.10% |
Percentage Split (66%) 22.89% |
10-Fold Cross Validation 23.10%

## Classification Results: top_from_each_minus_common.arff

Results of Test Options 

### NaiveBayes 
Training Set 48.11% |
Percentage Split (66%) 47.83% |
10-Fold Cross Validation 47.94%
### J48 
Training Set 51.22% |
Percentage Split (66%) 48.61% |
10-Fold Cross Validation 49.04%
### OneR 
Training Set 23.01% |
Percentage Split (66%) 22.92% |
10-Fold Cross Validation 23.01%


# Conclusions
The combination of top_from_each_minus_common and J48 yielded the best results.
While ingredients could serve as predictive indicators for cuisine type, other factors can affect the accuracy of our modeling.
One such factor is the amount of each ingredient used.
Our model still serves as a useful cuisine profiling model for machine learning, especially when the top ingredients for each cuisine are determined along with the removal of the ingredients they have in common.

# Future Research
Explore Feature Selection techniques
Remove redundant and irrelevant ingredients
Try to improve accuracy by finding the optimal subset of ingredients
Explore other classification algorithms
Naïve Bayes assumes independence between features
OneR creates one rule per feature and then chooses the one rule with the lowest error to predict the class

