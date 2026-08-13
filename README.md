# Titanic Survival Prediction

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

Binary classification project predicting passenger survival on the Titanic, built for the [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.

## Overview

This project walks through a full analysis pipeline in R: exploratory data analysis, feature engineering, missing value imputation, and comparison of two classification models (logistic regression and random forest), evaluated with 10-fold cross-validation.

## Dataset

- `train.csv` — 891 passengers with known survival outcome
- `test.csv` — 418 passengers to predict
- Source: Kaggle Titanic competition

## Exploratory Data Analysis

Key patterns identified before modeling:

**Sex was the strongest single predictor** — women survived at a much higher rate than men.

![Survival by Sex](Plots/Survival%20by%20Sex.png)

**Passenger class showed a clear gradient** — 1st class survived more often than 2nd, and 2nd more than 3rd.

![Survival by Class](Plots/survival%20by%20class.png)

**Age mattered at the extremes** — young children showed a survival advantage.

![Age Distribution by Survival](Plots/age_distribution.png)

**Fare correlated with survival**, largely as a proxy for class.

![Fare Distribution by Survival](Plots/fare_distribution.png)

**Family size had a non-linear relationship** — passengers traveling alone or in very large families survived less than those in small-to-mid-sized families. Interestingly, this signal was largely absorbed by other engineered features (Title, IsAlone) once included in the models — a reminder that EDA patterns don't always translate into independent model importance.

![Survival by Family Size](Plots/survival_by_family.png)

**Missing data**: Age (~20% missing), Cabin (~77% missing/blank), Embarked (2 missing).

## Feature Engineering

| Feature | Description | Rationale |
|---|---|---|
| `Title` | Extracted from passenger name (Mr, Mrs, Miss, Master, Rare) | Captures age/social status even when `Age` is missing |
| `FamilySize` | `SibSp + Parch + 1` | Combines sibling/spouse and parent/child counts into total family size |
| `IsAlone` | Binary flag for `FamilySize == 1` | Simplified signal for solo travelers |
| `Deck` | First letter of `Cabin`, or "Unknown" if missing | Approximates cabin location on the ship |

## Handling Missing Values

- **Age**: imputed with the median age within each `Title` group
- **Fare**: imputed with the median fare within each `Pclass` group
- **Embarked**: filled with the mode ("S")

## Models

Two models were trained and compared using 10-fold cross-validation:

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 82.83% |
| Random Forest | 83.06% |

The two models perform within a fraction of a percent of each other, suggesting the engineered features (particularly `Title` and `FamilySize`) carry most of the predictive signal, regardless of which algorithm is used on top of them.

Logistic regression also gives interpretable odds ratios for each predictor, while random forest provides a feature importance ranking:

![Random Forest Feature Importance](Plots/feature_importance.png)

## How to Run

```r
# Install dependencies
install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "ggplot2", "corrplot", "gridExtra", "stringi"))

# Run the script
source("titanic_clean.R")
```

Update the `read.csv()` file paths at the top of the script to point to your local `train.csv` and `test.csv`.

## Output

`outputs/titanic_submission.csv` — final predictions in Kaggle's required submission format (`PassengerId`, `Survived`).

## Tools

R, tidyverse, caret, randomForest, ggplot2

## Author

Neo Mthembu
[GitHub](https://github.com/neomthembu25-maker) · [Portfolio](https://datascienceportfol.io/neomthembu25) · [LinkedIn](https://linkedin.com/in/neo-mthembu-902b4621a)
