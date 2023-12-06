#title: "CYO_Project"
#git: https://github.com/chrislcp/CYO_Project.git

  #1-Introduction
  
#As I work with environmental issues, I have chosen a dataset in Kaggle to study the water quality of a river and I try a different question. The dataset was extracted from: River Water Quality EDA and Forecasting (kaggle.com), with is a river in Uckraine.
#Reference: https://www.kaggle.com/datasets/vbmokin/wq-southern-bug-river-0105202
#Dataset formats: two csv files. There were called PB_All_2000_2021 and PB_stations.
#Define the variables
# PB_stations-> selected: id and the length (Distance from the mouth of the river, km). The name of the station was removed, as it was needed.
# PB_All_2000_2021
#Water Quality Parameters : O2 ,CL ,SO4, PO4, BSK5 ,Suspended, NO2, NH4
#The main goal of the project: Develop some numeric Regression Models to calculate the target column (NH4).The question is can we substitute the weekly measurement of NH4 for this Regression Model?
  
##1.1-Loading Libraries
library(tidyverse)
library(dplyr)
library(caret)
library(corrplot)
library(gam)
library(rpart)
library(Boruta)
library(randomForest)
## 1.2-Data Preprocessing
###Data Loading 

PB_All_2000_2021 <-read_delim("https://raw.githubusercontent.com/chrislcp/CYO_Project/main/archive/PB_All_2000_2021.csv",delim = ";", escape_double = FALSE, trim_ws = TRUE)
PB_stations <- read_delim("https://raw.githubusercontent.com/chrislcp/CYO_Project/main/archive/PB_stations.csv",delim = ";", escape_double = FALSE, trim_ws = TRUE)
###Join, Selecting Columns
stations <- PB_stations%>%select(id,length)
All <-PB_All_2000_2021%>%mutate(date=str_replace_all(date,"[.]","-"))%>%mutate(date=dmy(date))%>%mutate(date=as.Date(date))
#Add the column: length
df <- left_join(All, stations, by = "id")
df_total <- df
#Filter: select stations(rows). Considering the measures realized, the similar stations are:14, 15 and 16.
df <-df%>%filter(id==c(14,15,16))
# Remove Columns
df <- df %>% select(-one_of('id', 'date')) 

#2-Analysis -a methods/analysis section that explains the process and techniques used, including data cleaning, data exploration and visualization, insights gained, and your modeling approaches (you must use at least two different models or algorithms);

##2.1 Preprocessing - Handling Missing Data
#The option applied here was to remove the entire rows that contain any NA, with the function drop_na(). 
# 2.2 - Data Normalization
#The option applied here was to standardization mean centering and scaling in train function: preProcess = c("center","scale") 
#Reference:https://www.rdocumentation.org/packages/caret/versions/6.0-92/topics/preProcess
##Handling Missing Data - 
df <- df %>% drop_na()
## 2.2 Dataviz - Exploratory Data Analysis (EDA)
# Box Plot of the water quality parameters 
df %>% select(-one_of('length')) %>% boxplot(df)
# Histogram of the water Target 
hist(df$NH4)
#Ask if there is an near Zero variance
nzv <- nearZeroVar(df) #OK

#See all the stations 
hist(df_total$id)
#Boruta function helps to identify unimportant variables
boruta_results <- Boruta(NH4~., df)
boruta_results
plot(boruta_results)
### 2.2.2 Correlograms (Pearson and Spearman)

CorPearson <-cor(df[1:9], method = "pearson") 
corrplot(CorPearson, method="circle",title = "Pearson", outline = TRUE, addCoefasPercent = TRUE) 

CorSpearman <-cor(df[1:9], method = "spearman") 
corrplot(CorSpearman, method="circle",title ="Spearman")

## 2.3 - Data Preparation to Machine Learning, Create Partition: train and test sets.
#The train(70%) and the test(30%) sets were created to run the models. 
# Data Normalization or Standardization 
#Define the target 

df <- df %>% rename(y=NH4)

# Remove Columns
#df <- df %>% select(-one_of('BSK5')) 

# Split the dataset: train and test sets.
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(df$y, times = 1, p = 0.7, list = FALSE)
train_set <- df%>%slice(-test_index)
test_set <- df%>%slice(test_index)

#3-Results -a results section that presents the modeling results and discusses the model performance 
##3.1 - method: Generalized Linear Model(glm)

set.seed(1, sample.kind="Rounding") 
#getModelInfo("glm")
#modelLookup("glm")

train_glm <- train(y~.,method="glm",data=train_set)
y_hat_glm <- predict(train_glm,test_set)
head(train_glm$results)
plot(y_hat_glm,test_set$y)
##3.2 - method: k-Nearest Neighbors(knn)

set.seed(1, sample.kind="Rounding") 

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(y~.,method = "knn",trControl = control,data=train_set,preProc = c("center","scale"),tuneGrid = data.frame(k = seq(1, 20, 1)))

train_knn$results
##3.3 -  method: Random Forest(rf)
y_hat_knn <- predict(train_knn, test_set, type = "raw")
set.seed(1, sample.kind="Rounding") 
train_rf<- train(y~.,method="rf",data=train_set,preProc = c("center","scale"))
y_hat_rf <- predict(train_rf, test_set, type = "raw")
train_rf$results
plot(y_hat_rf,test_set$y)
ggplot(train_rf, highlight = TRUE)

train_rf$results

#fit our final model
mtry = train_rf$bestTune$mtry
fit_rf <- randomForest(y~., mtry = train_rf$bestTune$mtry, data=train_set)

## 3.4- Results - Comparing Models
"To compare different models or to see how well we’re doing compared to a baseline, we will use root mean squared error (RMSE) as our loss function."


glm <- train_glm$results$RMSE
knn <- train_knn$results$RMSE
rf <- fit_rf$results$RMSE


methods <- c("glm", "knn","rf")
RMSE_results <- c(glm,knn,rf)

Comparing_Models_RMSE <- as.data.frame(cbind(methods,RMSE_results))
Comparing_Models_RMSE <- Comparing_Models_RMSE %>% arrange(RMSE_results)
Comparing_Models_RMSE
selected_model <- Comparing_Models_RMSE[1,1]

selected_model

#4-Conclusion

#After have compared the 3 models and have optimizated the model (rf) and calculate the Importance(IncNodePurity) of each attribute to the target (NH4) in this model. We could remove from "df" the 3 less important attributes e: BSK5,Suspended,length. 
#4.1- Future works
#In this chunk, called "Future works", I try to run all the code for rf again and compare or just the selected model as showed in and see if we are going to find a lower RMSE. After removing and running, I found a higher RMSE. For future, I can also optimizated more, try another methods for models and remove 1 column at a time.    	


#We can calculate the Importance of the variables 
imp <- as.data.frame(importance(fit_rf))
imp%>%arrange(desc(IncNodePurity))
#REMOVE the 3 less importants
df <- df %>% select(-one_of('BSK5','Suspended','length')) 
# Split the dataset: train and test sets.
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(df$y, times = 1, p = 0.7, list = FALSE)
train_set <- df%>%slice(-test_index)
test_set <- df%>%slice(test_index)
#Random Forest(rf)
set.seed(1, sample.kind="Rounding") 
train_rf<- train(y~.,method="rf",data=train_set,preProc = c("center","scale"))
y_hat_rf <- predict(train_rf, test_set, type = "raw")
train_rf$results
plot(y_hat_rf,test_set$y)
ggplot(train_rf, highlight = TRUE)

train_rf$results

#fit our final model
mtry = train_rf$bestTune$mtry
fit_rf <- randomForest(y~., mtry = train_rf$bestTune$mtry, data=train_set)
fit_rf$results$RMSE


#The lower RMSE (0.379581960443479) was from rf 


# Do we have enough trees?
plot(fit_rf)
#Results
y_hat_rf <- predict(fit_rf, test_set)
plot(y_hat_rf,test_set$y)

