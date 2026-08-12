# Packages
install.packages(c("randomForest","xgboost"))
install.packages("stringi")
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(dplyr)

# Set seed for reproducibility
set.seed(42)

# Load data
train <- read.csv("~/Data Projects/RStudio/Titanic Prediction/train.csv")
test <- read.csv("~/Data Projects/RStudio/Titanic Prediction/test.csv")

test_ids <- test$PassengerId
train$IsTrain <- T
test$IsTrain <- F
test$Survived <- NA

all_data <- bind_rows(train, test)

head(train)
summary(train)


# Exploratory Data Analysis

print(colSums(is.na(train)))

# Survival rate
cat("\noverall survival rate:", round(mean(train$Survived), 3), "\n")

# Survival by gender
p1 <- ggplot(train, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("red", "green"), labels = c("Deceased", "Survived")) +
  labs(title = "Survival Rate by Sex", y = "Proportion")
print(p1)

# Survival by Class
p2 <- ggplot(train, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("red", "blue"), labels = c("Deceased", "Survived")) +
  labs(title = "Survival Rate by Class", y = "Proportion")
print(p2)

# Age distribution by survival
p3 <- ggplot(train, aes(x = Age, fill = factor(Survived))) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
  scale_fill_manual(values = c("red", "blue"), labels = c("Deceased", "Survived")) +
  labs(title = "Age Distribution by Survival", fill = "Survived")
print(p3)

# Fare distribution by survival (log scale, since Fare is right-skewed)
p4 <- ggplot(train, aes(x = Fare, fill = factor(Survived))) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
  scale_x_log10() +
  scale_fill_manual(values = c("red", "blue"), labels = c("Deceased", "Survived")) +
  labs(title = "Fare Distribution by Survival (log scale)", fill = "Survived")
print(p4)

# Survival by family size
train_fam <- train %>% mutate(FamilySize = SibSp + Parch + 1)
p5 <- ggplot(train_fam, aes(x = FamilySize, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("red", "blue"), labels = c("Deceased", "Survived")) +
  labs(title = "Survival Rate by Family Size", y = "Proportion", fill = "Survived") +
  theme_minimal()
print(p5)

# Feature Engineering
all_data <- all_data %>%
  mutate(
    Title = gsub('(.*, )|(\\..*)', '', Name),
    Title = case_when(
      Title %in% c("Mlle", "Ms") ~ "Miss",
      Title == "Mme" ~ "Mrs",
      Title %in% c("Mr", "Mrs", "Miss", "Master") ~ Title,
      TRUE ~ "Rare"
    ),
    FamilySize = SibSp + Parch + 1,
    IsAlone = if_else(FamilySize == 1, 1, 0),
    Deck = if_else(is.na(Cabin) | Cabin == "", "Unknown", substr(Cabin, 1, 1))
  )

# Handle missing values

print(colSums(is.na(all_data)))

# Age missing value
all_data <- all_data %>%
  group_by(Title) %>%
  mutate(Age = if_else(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup()

sum(is.na(all_data$Age))   # should now print 0

# Fare missing value
all_data <- all_data %>%
  group_by(Pclass) %>%
  mutate(Fare = if_else(is.na(Fare), median(Fare, na.rm = TRUE), Fare)) %>%
  ungroup()

sum(is.na(all_data$Fare))   # should now print 0

# Embarked missing value
all_data$Embarked[is.na(all_data$Embarked) | all_data$Embarked == ""] <- "S"

sum(is.na(all_data$Embarked) | all_data$Embarked == "")   # should now print 0

# ============================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================

all_data <- all_data %>%
  mutate(
    Survived = as.factor(Survived),
    Pclass   = as.factor(Pclass),
    Sex      = as.factor(Sex),
    Embarked = as.factor(Embarked),
    Title    = as.factor(Title),
    Deck     = as.factor(Deck)
  )

all_data <- all_data %>%
  mutate(Age = round(Age))

all_data <- all_data %>%
  mutate(Fare = round(Fare, digits = 2))

# Split back Train/Test
model_vars <- c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
                "Fare", "Embarked", "Title", "FamilySize", "IsAlone", "Deck")

train_clean <- all_data %>% filter(IsTrain) %>% select(all_of(model_vars))
test_clean  <- all_data %>% filter(!IsTrain) %>% select(all_of(model_vars[-1]))


# Logistic Regression
log_model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked
                  + Title,
                  data = train_clean, family = "binomial")

cat("\n--- Logistic Regression Summary ---\n")
print(summary(log_model))
cat("\nOdds ratios:\n")
print(exp(coef(log_model)))


# Random Forest
rf_model <- randomForest(Survived ~ ., data = train_clean, ntree = 500, importance = TRUE)
print(rf_model)   # includes out-of-bag error estimate

# Feature importance plot
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# ============================================================
# MODEL COMPARISON VIA CROSS-VALIDATION
# ============================================================

train_control <- trainControl(method = "cv", number = 10)

cv_log <- train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title,
                data = train_clean, method = "glm", family = "binomial",
                trControl = train_control)

cv_rf <- train(Survived ~ ., data = train_clean, method = "rf",
               trControl = train_control, tuneLength = 3)

cat("\n--- 10-Fold Cross-Validation Accuracy ---\n")
cat("Logistic Regression:", round(max(cv_log$results$Accuracy), 4), "\n")
cat("Random Forest:      ", round(max(cv_rf$results$Accuracy), 4), "\n")


# Final prediction
final_predictions <- ifelse(predict(log_model, newdata = test_clean, type = "response") > 0.5, 1, 0)

submission <- data.frame(
  PassengerId = test_ids,
  Survived = as.integer(final_predictions)
)

write.csv(submission, "titanic_submission.csv", row.names = FALSE)
write.csv(all_data, "titanic_cleaned.csv", row.names = FALSE)
