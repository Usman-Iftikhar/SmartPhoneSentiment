# Title: Sentiment Analysis

# Last update: 2019.04.22

# File/project name: C4T3.R
# RStudio Project name: C4T3


## ----- PROJECT NOTES ----- ##
######################################################################################

# Summarize project: 
# Develop models to predict sentiment of iPhone/Galaxy
# Use the best model to predict sentiment in the LargeMatrix

######################################################################################


## ----- HOUSEKEEPING ----- ##
######################################################################################

# CLEAR OBJECTS IF NECESSARY
rm(list = ls())

# GET WORKING DIRECTORY
getwd()

# SET WORKING DIRECTORY
setwd("C:/users/muift/Documents/UT Data Analytics Data/R_projects/C4T3")
dir()

######################################################################################


## ----- LOAD PACKAGES ----- ##
######################################################################################
# https://cran.r-project.org/web/packages/available_packages_by_name.html


library(ggplot2) # Create Elegant Data Visualisations Using the Grammar of Graphics
#library(ggmap) # Spatial Visualization with ggplot2
#library(ggfortify) # Data Visualization Tools for Statistical Analysis Results
library(caret) # CLASSIFICATION AND REGRESSION TRAINING - STREAMLINE THE PROCESS FOR CREATING PREDICTIVE MODELS
library(corrplot) # GRAPHICAL DISPLAY OF CORRELATION MATRIX
#library(C50) # DECISION TREE AND RULE-BASED MODELS FOR PATTERN RECOGNITION
library(parallel)
library(doParallel) # PROVIDES A MECHANISM NEEDED TO EXECUTE FOREACH LOOPS IN PARALLEL
#library(mlbench) # ML BENCHMARK PROBLEMS
#library(readr) # READ RECTANGULAR TEXT DATA (CSV, TSV, FWF)
library(plyr) # TOOLS FOR SPLITTING, APPLYING AND COMBINING DATA
#library(knitr) # REPORT GENERATION
#library(arules) # PROVIDES THE INFRASTRUCTURE FOR  REPRESENTING, MANIPULATING AND ANALYZING TRANSACTION DATA AND PATTERNS (ITEMSETS AND ASSOCIATION RULES)
#library(caTools) # MOVING WINDOW STATISTICS, GIF, BASE64, ROC AUC
#library(prabclus) # FUNCTIONS FOR CLUSTERING OF PRESENCE-ABSENCE, ABUNDANCE AND MULTILOCUS GENETIC DATA
#library(DiceOptim) # KRIGING-BASED OPTIMIZATION FOR COMPUTER EXPERIMENTS
#library(DiceDesign) # DESIGN FOR COMPUTER EXPERIMENTS
#library(trimcluster) # CLUSTER ANALYSIS WITH TRIMMING
#library(arulesViz) # VISUALIZATING ASSOCIATION RULES AND FREQUENT ITEMSETS
#library(dbplyr) # A 'DPLYR' BACK END FOR DATABASES
library(dplyr) # DATA MANIPULATION (filter, summarize, mutate)
#library(tidyverse) #EASILY INSTALL AND LOAD THE 'TIDYVERSE'
#library(tidyr) # EASILY TIDY DATA WITH SPREADY() AND GATHER() FUNCTIONS
#library(tibble) # SIMPLE DATA FRAMES
#library(RMySQL) # DATABASE INTERFACE AND MYSQL DRIVER FOR R
#library(lubridate) # MAKE DEALING WITH DATES A LITTLE EASIER
library(plotly) # CREATE INTERACTIVE WEB GRAPHICS
#library(forecast) #FORECASTING FUNCTIONS FOR TIME SERIES AND LINEAR MODELS
#library(TTR) # TECHNICAL TRAINING RULES
#library(backports) # Reimplementations of Functions Introduced Since R-3.0.0
#library(cellranger) # Translate Spreadsheet Cell Ranges to Rows and Columns
#library(labeling) # AXIS LABELING
#library(forecast) # FORECASING FUNCTIONS FOR TIME SERIES AND LINEAR MODELS
#library(dataMaid) # A Suite of Checks for Identification of Potential Errors in a Data Frame as Part of the Data Screening Process
#library(scatterplot3d) # 3D SCATTER PLOT
#library(data.table) # EXTENTION OF DATA.FRAME
#library(som) # SELF-ORGANIZING MAP

######################################################################################

## ----- PARALLEL PROCESSING ----- ##
######################################################################################
# PARALLEL PROCESSING for Windows

# NOTE: Be sure to use the correct package for your operating system. 
# Remember that all packages should loaded in the 'Load packages' section.

detectCores()  # detect number of cores
cl <- makeCluster(2)  # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster. 
stopCluster(cl)
######################################################################################

## ----- IMPORT DATA ----- ##
######################################################################################

# Import iPhone data set for training
#####################################################
iPhone <- read.csv("iphone_smallmatrix_labeled_8d.csv")
iPhoneData <- iPhone

#####################################################

# Import iPhone LargeMatrix data set
#####################################################
iPhoneLM <- read.csv("iphoneLargeMatrix.csv")

iPhoneLM_ex <- read.csv("LargeMatrixYH1904.csv")

#####################################################

######################################################################################


## ----- EVALUTE DATA ----- ##
######################################################################################

# iPhone data set
#####################################################
summary(iPhoneData)
str(iPhoneData) #'data.frame':	12973 obs. of  59 variables:


## ------ Evaluate NA values ----------##

# Are there any NAs in df?
?is.na
any(is.na(iPhoneData)) 
# Use summary to identify where NAs are 
#summary(iPhoneata)

#####################################################

# iPhone LargeMatrix data set
#####################################################
summary(iPhoneLM)
str(iPhoneLM)
# 'data.frame':	26140 obs. of  60 variables:
#####################################################


######################################################################################


## ----- PREPROCESS ----- ##
######################################################################################

# iPhone data set
#####################################################

# Recode values (iPhoneData)
###########################################
iPhoneData$iphonesentiment[iPhoneData$iphonesentiment == 0] <- "Very Negative"
iPhoneData$iphonesentiment[iPhoneData$iphonesentiment == 1] <- "Negative"
iPhoneData$iphonesentiment[iPhoneData$iphonesentiment == 2] <- "Somewhat Negative"
iPhoneData$iphonesentiment[iPhoneData$iphonesentiment == 3] <- "Somewhat Positive"
iPhoneData$iphonesentiment[iPhoneData$iphonesentiment == 4] <- "Positive"
iPhoneData$iphonesentiment[iPhoneData$iphonesentiment == 5] <- "Very Positive"

# CHANGE DATA TYPES 
iPhoneData$iphonesentiment <- as.factor(iPhoneData$iphonesentiment)
str(iPhoneData)

plot_ly(iPhoneData, x= ~iPhoneData$iphonesentiment, type='histogram') %>%
  layout(
    title = 'Sentiment Towards iPhone',
    scene = list(
      xaxis = list(title = 'Sentimet'),
      yaxis = list(title = 'Count'))
  )

###########################################

# Examine correlation (iPhoneDataCOR)
###########################################
iPhoneData_corr <- cor(iPhoneData[,1:58])
corrplot(iPhoneData_corr, method = "circle", type="upper", order="hclust")
iPhoneData_corr
iPhoneData_corr.high <- findCorrelation(iPhoneData_corr, cutoff=0.8, names = TRUE, exact = TRUE)
iPhoneData_corr.high
# [1] "samsungdisneg" "samsungperneg" "samsungdispos" "htcdisneg"     "googleperneg"  "googleperpos" 
#[7] "samsungdisunc" "samsungcamunc" "htcperpos"     "nokiacamunc"   "nokiadisneg"   "nokiadispos"  
#[13] "nokiaperunc"   "nokiacampos"   "nokiadisunc"   "nokiaperneg"   "nokiacamneg"   "iphonedisneg" 
#[19] "iphonedispos"  "sonydispos"    "iosperunc"     "iosperneg"     "ios"           "htcphone"   

# Create dataframe from which the features will be removed from
iPhoneDataCOR <- iPhoneData
str(iPhoneDataCOR)
iPhoneDataCOR$samsungdisneg <- NULL
iPhoneDataCOR$samsungperneg <- NULL
iPhoneDataCOR$samsungdispos <- NULL
iPhoneDataCOR$htcdisneg <- NULL
iPhoneDataCOR$googleperneg <- NULL
iPhoneDataCOR$googleperpos <- NULL

iPhoneDataCOR$samsungdisunc <- NULL
iPhoneDataCOR$samsungcamunc <- NULL
iPhoneDataCOR$htcperpos <- NULL
iPhoneDataCOR$nokiacamunc <- NULL
iPhoneDataCOR$nokiadisneg <- NULL
iPhoneDataCOR$nokiadispos <- NULL

iPhoneDataCOR$nokiaperunc <- NULL
iPhoneDataCOR$nokiacampos <- NULL
iPhoneDataCOR$nokiadisunc <- NULL
iPhoneDataCOR$nokiaperneg <- NULL
iPhoneDataCOR$nokiacamneg <- NULL
iPhoneDataCOR$iphonedisneg <- NULL

iPhoneDataCOR$iphonedispos <- NULL
iPhoneDataCOR$sonydispos <- NULL
iPhoneDataCOR$iosperunc <- NULL
iPhoneDataCOR$iosperneg <- NULL
iPhoneDataCOR$ios <- NULL
iPhoneDataCOR$htcphone <- NULL

str(iPhoneDataCOR)

iPhoneData.corr2 <- cor(iPhoneDataCOR[,1:34])
corrplot(iPhoneData.corr2, method = "circle", type="upper", order="hclust")
iPhoneData.corr2

###########################################

# Examine Feature Variance (iPhoneDataNZV)
###########################################
nzvMetrics <- nearZeroVar(iPhoneData, saveMetrics = TRUE)
nzvMetrics
#                   freqRatio percentUnique zeroVar   nzv
# iphone             5.041322    0.20812457   FALSE FALSE
# samsunggalaxy     14.127336    0.05395822   FALSE FALSE
# sonyxperia        44.170732    0.03854159   FALSE  TRUE
# nokialumina      497.884615    0.02312495   FALSE  TRUE
# htcphone          11.439614    0.06937486   FALSE FALSE
# ios               27.735294    0.04624990   FALSE  TRUE
# googleandroid     61.247573    0.04624990   FALSE  TRUE
# iphonecampos      10.524697    0.23124952   FALSE FALSE
# samsungcampos     93.625000    0.08479149   FALSE  TRUE
# sonycampos       348.729730    0.05395822   FALSE  TRUE
# nokiacampos     1850.142857    0.08479149   FALSE  TRUE
# htccampos         79.272152    0.16958298   FALSE  TRUE
# iphonecamneg      19.517529    0.13104139   FALSE  TRUE
# samsungcamneg    100.132812    0.06937486   FALSE  TRUE
# sonycamneg      1851.285714    0.04624990   FALSE  TRUE
# nokiacamneg     2158.833333    0.06166654   FALSE  TRUE
# htccamneg         93.444444    0.11562476   FALSE  TRUE
# iphonecamunc      16.764205    0.16187466   FALSE FALSE
# samsungcamunc     74.308140    0.06937486   FALSE  TRUE
# sonycamunc       588.318182    0.03854159   FALSE  TRUE
# nokiacamunc     2591.200000    0.05395822   FALSE  TRUE
# htccamunc         50.548000    0.12333308   FALSE  TRUE
# iphonedispos       6.792440    0.24666615   FALSE FALSE
# samsungdispos     97.061069    0.13104139   FALSE  TRUE
# sonydispos       331.076923    0.06937486   FALSE  TRUE
# nokiadispos     1438.777778    0.09249981   FALSE  TRUE
# htcdispos         64.694301    0.20041625   FALSE  TRUE
# iphonedisneg      10.084428    0.18499961   FALSE FALSE
# samsungdisneg     99.155039    0.10791644   FALSE  TRUE
# sonydisneg      2159.333333    0.06937486   FALSE  TRUE
# nokiadisneg     1850.142857    0.08479149   FALSE  TRUE
# htcdisneg         88.492958    0.14645803   FALSE  TRUE
# iphonedisunc      11.471875    0.20812457   FALSE FALSE
# samsungdisunc     74.255814    0.09249981   FALSE  TRUE
# sonydisunc       719.222222    0.05395822   FALSE  TRUE
# nokiadisunc     1619.375000    0.04624990   FALSE  TRUE
# htcdisunc         50.590361    0.13874971   FALSE  TRUE
# iphoneperpos       9.297834    0.19270793   FALSE FALSE
# samsungperpos     94.200000    0.10791644   FALSE  TRUE
# sonyperpos       416.870968    0.06166654   FALSE  TRUE
# nokiaperpos     2158.000000    0.08479149   FALSE  TRUE
# htcperpos         74.279762    0.19270793   FALSE  TRUE
# iphoneperneg      11.054137    0.16958298   FALSE FALSE
# samsungperneg    101.650794    0.10020812   FALSE  TRUE
# sonyperneg      2159.666667    0.07708317   FALSE  TRUE
# nokiaperneg     3237.250000    0.09249981   FALSE  TRUE
# htcperneg         94.428571    0.15416635   FALSE  TRUE
# iphoneperunc      13.018349    0.12333308   FALSE FALSE
# samsungperunc     86.500000    0.09249981   FALSE  TRUE
# sonyperunc      3240.250000    0.04624990   FALSE  TRUE
# nokiaperunc     1850.428571    0.06937486   FALSE  TRUE
# htcperunc         50.055556    0.15416635   FALSE  TRUE
# iosperpos        153.373494    0.09249981   FALSE  TRUE
# googleperpos      98.592308    0.06937486   FALSE  TRUE
# iosperneg        141.744444    0.09249981   FALSE  TRUE
# googleperneg      99.403101    0.08479149   FALSE  TRUE
# iosperunc        135.893617    0.07708317   FALSE  TRUE
# googleperunc      96.443609    0.07708317   FALSE  TRUE
# iphonesentiment    3.843017    0.04624990   FALSE FALSE


# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(iPhoneData, saveMetrics = FALSE) 
nzv
# [1]  3  4  6  7  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39 40 41 42 44 45 46
# [37] 47 49 50 51 52 53 54 55 56 57 58

# Data set with nzv features removed
iPhoneDataNZV <- iPhoneData[,-nzv]
str(iPhoneDataNZV)
# 'data.frame':	12973 obs. of  12 variables:
#   $ iphone         : int  1 1 1 1 1 41 1 1 1 1 ...
# $ samsunggalaxy  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 0 0 0 1 1 0 0 0 ...
# $ iphonecamunc   : int  0 0 0 0 0 7 1 0 0 0 ...
# $ iphonedispos   : int  0 0 0 0 0 1 13 0 0 0 ...
# $ iphonedisneg   : int  0 0 0 0 0 3 10 0 0 0 ...
# $ iphonedisunc   : int  0 0 0 0 0 4 9 0 0 0 ...
# $ iphoneperpos   : int  0 1 0 1 1 0 5 3 0 0 ...
# $ iphoneperneg   : int  0 0 0 0 0 0 4 1 0 0 ...
# $ iphoneperunc   : int  0 0 0 1 0 0 5 0 0 0 ...
# $ iphonesentiment: Factor w/ 6 levels "Negative","Positive",..: 5 5 5 5 5 2 2 5 5 5 ...

###########################################

# Recursive Feature Elimination (iPhoneRFE)
###########################################
# Sample the data before using RFE
set.seed(123)
iPhoneSample <- iPhoneData[sample(1:nrow(iPhoneData), 1000, replace=FALSE),]

ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)


# Use rfe and omit the response variable (attribute 59 iphonesentiment)
rfeResults <- rfe(iPhoneSample[,1:58], 
                  iPhoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)
rfeResults
# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold, repeated 5 times) 
# 
# Resampling performance over subset size:
#   
# #  Variables Accuracy  Kappa AccuracySD KappaSD Selected
# #         26   0.7638 0.5464    0.02777 0.05992        *
# 
# The top 5 variables (out of 26):
#   iphone, samsunggalaxy, googleandroid, iphonedisunc, htcphone

predictors(rfeResults)
# [1] "iphone"        "samsunggalaxy" "googleandroid" "iphonedisunc"  "htcphone"      "iphonedispos" 
#[7] "iphoneperneg"  "iphoneperpos"  "iphonedisneg"  "iphonecamunc"  "htccampos"     "sonyxperia"   
#[13] "iphonecamneg"  "iphoneperunc"  "ios"           "htcperpos"     "htccamunc"     "htcdispos"    
#[19] "iphonecampos"  "htccamneg"     "htcperneg"     "htcperunc"     "googleperpos"  "htcdisneg"    
#[25] "samsungdispos" "samsungperpos"

plot(rfeResults, type = c('g', 'o'))

# Data set with RFE recommended features
iPhoneRFE <- iPhoneData[,predictors(rfeResults)]
# Add the dependant variable (iphonesentiment)
iPhoneRFE$iphonesentiment <- iPhoneData$iphonesentiment

str(iPhoneRFE)
#'data.frame':	12973 obs. of  27 variables:

###########################################

# Recode and reduce levels to 4 (iPhoneRC)
###########################################
## Recode values since some levels have poor sensitivity and balanced accuracy
# create a new dataset that will be used for recoding sentiment
iPhoneRC <- iPhone
summary(iPhoneRC[,59])
# recode (dplyr package) sentiment to combine factor levels 0 & 1 and 4 & 5
iPhoneRC$iphonesentiment <- recode(iPhoneRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iPhoneRC[,59])
str(iPhoneRC[,59])

iPhoneRC$iphonesentiment[iPhoneRC$iphonesentiment == 1] <- "Negative"
iPhoneRC$iphonesentiment[iPhoneRC$iphonesentiment == 2] <- "Somewhat Negative"
iPhoneRC$iphonesentiment[iPhoneRC$iphonesentiment == 3] <- "Somewhat Positive"
iPhoneRC$iphonesentiment[iPhoneRC$iphonesentiment == 4] <- "Positive"
# make iphonesentiment a factor
iPhoneRC$iphonesentiment <- as.factor(iPhoneRC$iphonesentiment)
summary(iPhoneRC[,59])
str(iPhoneRC)
# $ iphonesentiment: Factor w/ 4 levels "Negative","Positive",..: 1 1 1 1 1 2 2 1 1 1 ...

plot_ly(iPhoneRC, x= ~iPhoneRC$iphonesentiment, type='histogram') %>%
  layout(
    title = 'Sentiment Towards iPhone',
    scene = list(
      xaxis = list(title = 'Sentimet'),
      yaxis = list(title = 'Count'))
  )

###########################################

# Recoded w/reduced level=4 and Recursive Feature Elimination (iPhoneRC.RFE)
###########################################
# Sample the data before using RFE
set.seed(123)
iPhoneRC.Sample <- iPhoneRC[sample(1:nrow(iPhoneRC), 1000, replace=FALSE),]

# use the same control from the RFE data set (ctrl)

# Use rfe and omit the response variable (attribute 59 iphonesentiment)
rfeRC.Results <- rfe(iPhoneRC.Sample[,1:58],
                     iPhoneSample$iphonesentiment,
                     rfeControl=ctrl)
rfeRC.Results
# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold, repeated 5 times) 
# 
# Resampling performance over subset size:
#   
# Variables Accuracy  Kappa AccuracySD KappaSD Selected
#         4   0.7160 0.4298    0.02701 0.06469         
#         8   0.7402 0.4945    0.02655 0.05851         
#        16   0.7606 0.5459    0.02924 0.05888        *
#        58   0.7416 0.4860    0.02402 0.05688         
# 
# The top 5 variables (out of 16):
#   iphone, samsunggalaxy, googleandroid, iphonedisunc, htcphone

predictors(rfeRC.Results)
# [1] "iphone"        "samsunggalaxy" "googleandroid" "iphonedisunc"  "htcphone"      "iphonedispos"  "iphoneperneg" 
# [8] "iphoneperpos"  "iphonedisneg"  "iphonecamunc"  "htccampos"     "sonyxperia"    "iphonecamneg"  "iphoneperunc" 
# [15] "ios"           "htcperpos" 

plot(rfeRC.Results, type = c('g', 'o'))

# Data set with RFE recommended features
iPhoneRC.RFE <- iPhoneRC[,predictors(rfeRC.Results)]
# Add the dependant variable (iphonesentiment)
iPhoneRC.RFE$iphonesentiment <- iPhoneRC$iphonesentiment

str(iPhoneRC.RFE)
#'data.frame':	12973 obs. of  27 variables:

###########################################

#####################################################

# iPhone Large Matrix data set
#####################################################
iPhoneLM$id <- NULL

iPhoneLM$iphonesentiment <- as.factor(iPhoneLM$iphonesentiment)
str(iPhoneLM)
# 'data.frame':	26140 obs. of  59 variables:

###########################

# nearZeroVar() - preprocessing for the large matrix
nzv
# [1]  3  4  6  7  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39 40 41 42 44 45 46
# [37] 47 49 50 51 52 53 54 55 56 57 58

# Data set with nzv features removed
iPhoneLM_NZV <- iPhoneLM[,-nzv]
str(iPhoneLM_NZV)

###########################

#####################################################


######################################################################################



## ----- SAMPLING - not used for this task ----- ##
######################################################################################

#IPHONE

# Create data frames for each data set with 1,000 samples

#iPhoneData1k <- iPhoneData[sample(1:nrow(iPhoneData), 1000, replace = FALSE),]
#str(iPhoneData1k)
#'data.frame':	1000 obs. of  59 variables:


#iPhoneDataCOR1k <- iPhoneDataCOR[sample(1:nrow(iPhoneDataCOR), 1000, replace = FALSE),]
#str(iPhoneDataCOR1k)
#'data.frame':	1000 obs. of  35 variables:


#iPhoneDataNZV1k <- iPhoneDataNZV[sample(1:nrow(iPhoneDataNZV), 1000, replace = FALSE),]
#str(iPhoneDataNZV1k)
#'data.frame':	1000 obs. of  9 variables:


#iPhoneRFE1k <- iPhoneRFE[sample(1:nrow(iPhoneRFE), 1000, replace = FALSE),]
#str(iPhoneRFE1k)
#'data.frame':	1000 obs. of  12 variables:


######################################################################################


## ----- MODEL DEVELOPMENT IPHONE ----- ## 
######################################################################################

# SET 10 FOLD CROSS VALIDATION CONTROL
###########################################
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 1)
###########################################


#iPhoneData
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingFULL <- createDataPartition(iPhoneData$iphonesentiment, 
                                  p = 0.70, 
                                  list = FALSE)
trainSetFULL <- iPhoneData[inTrainingFULL,] 
str(trainSetFULL) # 'data.frame':	9083 obs. of  59 variables:
testSetFULL <- iPhoneData[-inTrainingFULL,]  
str(testSetFULL) # 'data.frame':	3890 obs. of  59 variables:

###########################################

#C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0FULL <- train(iphonesentiment~.,
                              data=trainSetFULL,
                              method = "C5.0",
                              trControl = fitControl,
                              tuneLength = 1))
#   user  system elapsed 
#   2.78    0.08   43.02

c5.0FULL
#C5.0 

#9083 samples
#58 predictor
#6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
#Resampling results across tuning parameters:
  
#  model  winnow  Accuracy   Kappa    
#  rules  FALSE   0.7722112  0.5585117
#  rules   TRUE   0.7712221  0.5565565
#   tree  FALSE   0.7710007  0.5566811
#   tree   TRUE   0.7705623  0.5563522

#Tuning parameter 'trials' was held constant at a value of 1
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = FALSE.

###########################################

#RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfFULL <- train(iphonesentiment~.,
                            data=trainSetFULL, 
                            method = "rf", 
                            trControl = fitControl,
                            tuneLength = 1))
#    user  system elapsed 
#   67.37    0.71  457.72  

rfFULL
#Random Forest 

#9083 samples
#58 predictor
#6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
#Resampling results:
  
#  Accuracy   Kappa    
# 0.7766151  0.5681085

#Tuning parameter 'mtry' was held constant at a value of 19

###########################################

#kNN TRAIN/FIT
###########################################
set.seed(123)
system.time(kknnFULL <- train(iphonesentiment~.,
                              data=trainSetFULL,
                              method = "kknn",
                              trControl = fitControl,
                              tuneLength = 1))
#user  system elapsed 
#8.07    0.09   52.93

kknnFULL
#k-Nearest Neighbors 

#9083 samples
#58 predictor
#6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
#Resampling results:
  
#  Accuracy  Kappa   
#  0.311245  0.154045

#Tuning parameter 'kmax' was held constant at a value of 5
#Tuning parameter 'distance' was held constant at a value of 2
#Tuning parameter 'kernel' was held constant at a value of optimal

###########################################

#SVM (e1071) TRAIN/FIT
###########################################
set.seed(123)
system.time(svmFULL <- train(iphonesentiment~.,
                             data=trainSetFULL,
                             method = "svmLinear2",
                             trControl = fitControl,
                             tuneLength = 1))
#user  system elapsed 
#23.29    0.16  128.12 

svmFULL
#Support Vector Machines with Linear Kernel 

#9083 samples
#58 predictor
#6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
#Resampling results:
  
#  Accuracy  Kappa    
#  0.714194  0.4198418

#Tuning parameter 'cost' was held constant at a value of 0.25

###########################################

#####################################################


#iPhoneDataCOR
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingCOR <- createDataPartition(iPhoneDataCOR$iphonesentiment, 
                                      p = 0.70, 
                                      list = FALSE)
trainSetCOR <- iPhoneDataCOR[inTrainingCOR,] 
str(trainSetCOR) # 'data.frame':	9083 obs. of  35 variables:
testSetCOR <- iPhoneDataCOR[-inTrainingCOR,]  
str(testSetCOR) # 'data.frame':	3890 obs. of  35 variables:

###########################################

#C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0COR <- train(iphonesentiment~.,
                             data=trainSetCOR,
                             method = "C5.0",
                             trControl = fitControl,
                             tuneLength = 1))
# user  system elapsed 
# 1.59    0.17   19.12

c5.0COR
# C5.0 
# 
# 9083 samples
# 34 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7538268  0.5168550
# rules   TRUE   0.7540471  0.5175124
# tree   FALSE   0.7545977  0.5183924
# tree    TRUE   0.7536074  0.5168228
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.

###########################################

#RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfCOR <- train(iphonesentiment~.,
                           data=trainSetCOR,
                           method = "rf",
                           trControl = fitControl,
                           tuneLength = 1))
# user  system elapsed 
# 19.21    0.34  126.01   

rfCOR
# Random Forest 
# 
# 9083 samples
# 34 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7557004  0.5226603
# 
# Tuning parameter 'mtry' was held constant at a value of 11

###########################################

#####################################################


#iPhoneDataNZV
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingNZV <- createDataPartition(iPhoneDataNZV$iphonesentiment, 
                                     p = 0.70, 
                                     list = FALSE)
trainSetNZV <- iPhoneDataNZV[inTrainingNZV,] 
str(trainSetNZV) # 'data.frame':	9083 obs. of  9 variables:
testSetNZV <- iPhoneDataNZV[-inTrainingNZV,]  
str(testSetNZV) # 'data.frame':	3890 obs. of  9 variables:

###########################################

#C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0NZV <- train(iphonesentiment~.,
                             data=trainSetNZV,
                             method = "C5.0",
                             trControl = fitControl,
                             tuneLength = 1))
# user  system elapsed 
# 0.96    0.02    7.87 

c5.0COR
# C5.0 
# 
# 9083 samples
# 34 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7538268  0.5168550
# rules   TRUE   0.7540471  0.5175124
# tree   FALSE   0.7545977  0.5183924
# tree    TRUE   0.7536074  0.5168228
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.

###########################################

#RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfNZV <- train(iphonesentiment~.,
                           data=trainSetNZV,
                           method = "rf",
                           trControl = fitControl,
                           tuneLength = 1))
# user  system elapsed 
# 5.33    0.12   28.19   

rfNZV
# Random Forest 
# 
# 9083 samples
# 11 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7610929  0.5319261
# 
# Tuning parameter 'mtry' was held constant at a value of 3

###########################################

#####################################################


#iPhoneRFE
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingRFE <- createDataPartition(iPhoneRFE$iphonesentiment, 
                                     p = 0.70, 
                                     list = FALSE)
trainSetRFE <- iPhoneRFE[inTrainingRFE,] 
str(trainSetRFE) # 'data.frame':	9083 obs. of  27 variables:
testSetRFE <- iPhoneRFE[-inTrainingRFE,]  
str(testSetRFE) # 'data.frame':	3890 obs. of  27 variables:

###########################################

#C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0RFE <- train(iphonesentiment~.,
                             data=trainSetRFE,
                             method = "C5.0",
                             trControl = fitControl,
                             tuneLength = 1))
# user  system elapsed 
# 1.53    0.10   16.63

c5.0RFE
# C5.0 
# 
# 9083 samples
# 26 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7726525  0.5594310
# rules   TRUE   0.7721022  0.5586905
# tree   FALSE   0.7713317  0.5573449
# tree    TRUE   0.7708917  0.5568612
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.

###########################################

#RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfRFE <- train(iphonesentiment~.,
                           data=trainSetRFE,
                           method = "rf",
                           trControl = fitControl,
                           tuneLength = 1))
# user  system elapsed 
# 16.03    0.15   94.84  

rfRFE
# Random Forest 
# 
# 9083 samples
# 26 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results:
#   
#   Accuracy  Kappa    
# 0.777717  0.5701341
# 
# Tuning parameter 'mtry' was held constant at a value of 8


###########################################

#####################################################


#iPhoneRC
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingRC <- createDataPartition(iPhoneRC$iphonesentiment, 
                                    p = 0.70, 
                                    list = FALSE)
trainSetRC <- iPhoneRC[inTrainingRC,] 
str(trainSetRC) # 'data.frame':	9083 obs. of  59 variables:
testSetRC <- iPhoneRC[-inTrainingRC,]  
str(testSetRC) # 'data.frame':	3890 obs. of  59 variables:

###########################################

# C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0RC <- train(iphonesentiment~.,
                            data=trainSetRC,
                            method = "C5.0",
                            trControl = fitControl,
                            tuneLength = 1))
# user  system elapsed 
# 2.01    0.05   28.61 

c5.0RC
# C5.0 
# 
# 9083 samples
# 58 predictor
# 4 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8176, 8177, 8175, 8174, 8174, 8175, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.8463052  0.6148343
# rules   TRUE   0.8458634  0.6152256
# tree   FALSE   0.8464138  0.6164574
# tree    TRUE   0.8461929  0.6165387
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfRc <- train(iphonesentiment~.,
                          data=trainSetRC,
                          method = "rf",
                          trControl = fitControl,
                          tuneLength = 1))
# user  system elapsed 
# 39.95    0.17  301.20  

rfRc
# Random Forest 
# 
# 9083 samples
# 58 predictor
# 4 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8176, 8177, 8175, 8174, 8174, 8175, ... 
# Resampling results:
#   
#   Accuracy   Kappa   
# 0.8513682  0.628551
# 
# Tuning parameter 'mtry' was held constant at a value of 19


###########################################

#####################################################

# iPhoneRC.RFE
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingRC.RFE <- createDataPartition(iPhoneRC.RFE$iphonesentiment,
                                        p = 0.70,
                                        list = FALSE)
trainSetRC.RFE <- iPhoneRC.RFE[inTrainingRC.RFE,] 
str(trainSetRC.RFE) # 'data.frame':	9083 obs. of  17 variables:
testSetRC.RFE <- iPhoneRC.RFE[-inTrainingRC.RFE,]  
str(testSetRC.RFE) # 'data.frame':	3890 obs. of  17 variables:

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfRC.RFE <- train(iphonesentiment~.,
                              data=trainSetRC.RFE,
                              method = "rf",
                              trControl = fitControl,
                              tuneLength = 1))
# user  system elapsed 
# 9.11    0.22   50.94 

rfRC.RFE
# Random Forest 
# 
# 9083 samples
# 16 predictor
# 4 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8176, 8177, 8175, 8174, 8174, 8175, ... 
# Resampling results:
#   
#   Accuracy   Kappa   
# 0.8472974  0.619405
# 
# Tuning parameter 'mtry' was held constant at a value of 5


###########################################

#####################################################


# iPhone - principal components analysis (iPhonePCA)
#####################################################

# Create train/test usiing iPhone full data set
###########################################

# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(trainSetFULL[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)
# Created from 9083 samples and 58 variables
# 
# Pre-processing:
#   - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)
# 
# PCA needed 24 components to capture 95 percent of the variance


# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, trainSetFULL[,-59])

# add the dependent to training
train.pca$iphonesentiment <- trainSetFULL$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testSetFULL[,-59])

# add the dependent to training
test.pca$iphonesentiment <- testSetFULL$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)

###########################################

#RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfPCA <- train(iphonesentiment~.,
                           data=train.pca,
                           method = "rf",
                           trControl = fitControl,
                           tuneLength = 1))
# user  system elapsed 
# 16.79    0.18   98.19 

rfPCA
# Random Forest 
# 
# 9083 samples
# 24 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8175, 8174, 8173, 8174, 8175, 8176, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7619732  0.5443754
# 
# Tuning parameter 'mtry' was held constant at a value of 8


###########################################

#####################################################

######################################################################################


## ----- EVALUATE MODELS ----- ##
######################################################################################

# iPhoneData
#####################################################

# USE RESAMPLES TO COMPARE MODEL PERFORMANCE
iPhoneFullResults <- resamples(list(CompleteData_C5.0 = c5.0FULL,
                                    CompleteData_RandomForest = rfFULL,
                                    CompleteData_kNN = kknnFULL,
                                    CompleteData_SVM = svmFULL,
                                    NZVariance_C5.0 = c5.0NZV,
                                    NZVariance_RandomForest = rfNZV,
                                    CorrelationFE_C5.0 = c5.0COR,
                                    CorrelationFE_RandomForest = rfCOR,
                                    RFE_C5.0 = c5.0RFE,
                                    RFE_RandomForest = rfRFE,
                                    RCLevelFactors_C5.0 = c5.0RC,
                                    RCLevelFactors_RandomForest = rfRc,
                                    RCLevelFactors.RFE_RandomForest = rfRC.RFE,
                                    PCA_RandomForest = rfPCA))
summary(iPhoneFullResults)
# Call:
#   summary.resamples(object = iPhoneFullResults)
# 
# Models: CompleteData_C5.0, CompleteData_RandomForest, CompleteData_kNN, CompleteData_SVM, NZVariance_C5.0, NZVariance_RandomForest, CorrelationFE_C5.0, CorrelationFE_RandomForest, RFE_C5.0, RFE_RandomForest, RCLevelFactors_C5.0, RCLevelFactors_RandomForest, RCLevelFactors.RFE_RandomForest, PCA_RandomForest 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# CompleteData_C5.0               0.7601760 0.7675385 0.7736784 0.7722112 0.7767386 0.7810781    0
# CompleteData_RandomForest       0.7640573 0.7734652 0.7775330 0.7766151 0.7819642 0.7852423    0
# CompleteData_kNN                0.2871287 0.3042152 0.3095099 0.3112450 0.3196483 0.3292952    0
# CompleteData_SVM                0.6974697 0.7047654 0.7124772 0.7141940 0.7224197 0.7364939    0
# NZVariance_C5.0                 0.7502750 0.7528910 0.7572925 0.7570211 0.7585352 0.7662624    0
# NZVariance_RandomForest         0.7533040 0.7561174 0.7616941 0.7610929 0.7645116 0.7709251    0
# CorrelationFE_C5.0              0.7392739 0.7502072 0.7549559 0.7545977 0.7595733 0.7676211    0
# CorrelationFE_RandomForest      0.7436744 0.7503436 0.7544053 0.7557004 0.7609342 0.7676211    0
# RFE_C5.0                        0.7612761 0.7683635 0.7722745 0.7726525 0.7767984 0.7819383    0
# RFE_RandomForest                0.7673649 0.7749660 0.7772268 0.7777170 0.7821182 0.7863436    0
# RCLevelFactors_C5.0             0.8237885 0.8405837 0.8468261 0.8464138 0.8561606 0.8621830    0
# RCLevelFactors_RandomForest     0.8325991 0.8457317 0.8514028 0.8513682 0.8597360 0.8681319    0
# RCLevelFactors.RFE_RandomForest 0.8259912 0.8403523 0.8454345 0.8472974 0.8571405 0.8643881    0
# PCA_RandomForest                0.7353914 0.7536421 0.7659695 0.7619732 0.7709251 0.7775330    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# CompleteData_C5.0               0.5329749 0.5501121 0.5604326 0.5585117 0.5689805 0.5816579    0
# CompleteData_RandomForest       0.5406454 0.5624131 0.5699775 0.5681085 0.5761466 0.5930385    0
# CompleteData_kNN                0.1300408 0.1458707 0.1525765 0.1540450 0.1661087 0.1725680    0
# CompleteData_SVM                0.3835160 0.3977997 0.4168402 0.4198418 0.4365696 0.4760533    0
# NZVariance_C5.0                 0.5095281 0.5145345 0.5240134 0.5236425 0.5264598 0.5422104    0
# NZVariance_RandomForest         0.5150988 0.5226981 0.5322916 0.5319261 0.5385897 0.5587764    0
# CorrelationFE_C5.0              0.4820438 0.5080847 0.5183510 0.5183924 0.5275413 0.5501056    0
# CorrelationFE_RandomForest      0.4955773 0.5109221 0.5185001 0.5226603 0.5315920 0.5521050    0
# RFE_C5.0                        0.5350570 0.5508501 0.5600565 0.5594310 0.5684592 0.5826266    0
# RFE_RandomForest                0.5462667 0.5654455 0.5707509 0.5701341 0.5766260 0.5947268    0
# RCLevelFactors_C5.0             0.5443997 0.5988205 0.6237488 0.6164574 0.6417232 0.6614388    0
# RCLevelFactors_RandomForest     0.5688873 0.6115564 0.6297600 0.6285510 0.6523175 0.6764933    0
# RCLevelFactors.RFE_RandomForest 0.5590553 0.5976603 0.6148549 0.6194050 0.6486467 0.6649739    0
# PCA_RandomForest                0.4901869 0.5297965 0.5530090 0.5443754 0.5608706 0.5789880    0

#####################################################

######################################################################################


## ----- Predict iPhone test data sets----- ##
######################################################################################

# iPhoneData
#####################################################

# C5.0 prediction
###########################################
c5.0PredFULL <- predict(c5.0FULL, testSetFULL)
postResample(c5.0PredFULL, testSetFULL$iphonesentiment)
# Accuracy     Kappa 
# 0.7696658 0.5523437

# Create a confusion matrix from C5.0 predictions 
CMc5.0FULL <- confusionMatrix(c5.0PredFULL, testSetFULL$iphonesentiment)
CMc5.0FULL
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 1      140                 0                 2             6            16
# Somewhat Negative        0        0                18                 0             0             0
# Somewhat Positive        1        5                 2               230             2             3
# Very Negative            0        2                 1                 2           373            10
# Very Positive          115      284               115               122           207          2233
# 
# Overall Statistics
# 
# Accuracy : 0.7697          
# 95% CI : (0.7561, 0.7828)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5523          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.32483                 0.132353                  0.64607              0.63435               0.9872
# Specificity                  1.00000         0.99277                 1.000000                  0.99632              0.99546               0.4822
# Pos Pred Value                   NaN         0.84848                 1.000000                  0.94650              0.96134               0.7259
# Neg Pred Value               0.96992         0.92188                 0.969525                  0.96545              0.93861               0.9644
# Prevalence                   0.03008         0.11080                 0.034961                  0.09152              0.15116               0.5815
# Detection Rate               0.00000         0.03599                 0.004627                  0.05913              0.09589               0.5740
# Detection Prevalence         0.00000         0.04242                 0.004627                  0.06247              0.09974               0.7907
# Balanced Accuracy            0.50000         0.65880                 0.566176                  0.82119              0.81491               0.7347

###########################################

# RF prediction
###########################################
rfPredFULL <- predict(rfFULL, testSetFULL)
postResample(rfPredFULL, testSetFULL$iphonesentiment)
# Accuracy     Kappa 
# 0.7753213 0.5634136

# Create a confusion matrix from random forest predictions 
CMrfFULL <- confusionMatrix(rfPredFULL, testSetFULL$iphonesentiment) 
CMrfFULL
# Confusion Matrix and Statistics
# 
#                   Reference (Groud truth)
# Prediction        Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 1        1                 0                 0             0             1
# Positive                 0      141                 0                 1             1             3
# Somewhat Negative        0        0                18                 0             0             4
# Somewhat Positive        0        5                 3               234             0             6
# Very Negative            0        2                 2                 1           382             8
# Very Positive          116      282               113               120           205          2240
# 
# Overall Statistics
# 
# Accuracy : 0.7753          
# 95% CI : (0.7619, 0.7884)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5634          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0085470         0.32715                 0.132353                  0.65730               0.6497               0.9903
# Specificity                0.9994699         0.99855                 0.998934                  0.99604               0.9961               0.4865
# Pos Pred Value             0.3333333         0.96575                 0.818182                  0.94355               0.9671               0.7282
# Neg Pred Value             0.9701569         0.92254                 0.969493                  0.96650               0.9411               0.9730
# Prevalence                 0.0300771         0.11080                 0.034961                  0.09152               0.1512               0.5815
# Detection Rate             0.0002571         0.03625                 0.004627                  0.06015               0.0982               0.5758
# Detection Prevalence       0.0007712         0.03753                 0.005656                  0.06375               0.1015               0.7907
# Balanced Accuracy          0.5040085         0.66285                 0.565644                  0.82667               0.8229               0.7384

###########################################

# kNN prediction
###########################################
kknnPredFULL <- predict(kknnFULL, testSetFULL)
postResample(kknnPredFULL, testSetFULL$iphonesentiment)
#Accuracy     Kappa 
#0.3113111 0.1566191

CMkknnFULL <- confusionMatrix(kknnPredFULL, testSetFULL$iphonesentiment)
CMkknnFULL
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 5        5                 1                 1             1            50
# Positive                 5      140                 4                 3             5            77
# Somewhat Negative        3        6                25                 4            13            74
# Somewhat Positive        1        8                 2               234             9            27
# Very Negative           91      223                87               101           527          1754
# Very Positive           12       49                17                13            33           280
# 
# Overall Statistics
# 
# Accuracy : 0.3113          
# 95% CI : (0.2968, 0.3261)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.1566          
# Mcnemar's Test P-Value : <2e-16          
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                 0.042735         0.32483                 0.183824                  0.65730               0.8963              0.12378
# Specificity                 0.984628         0.97282                 0.973362                  0.98670               0.3168              0.92383
# Pos Pred Value              0.079365         0.59829                 0.200000                  0.83274               0.1894              0.69307
# Neg Pred Value              0.970734         0.92040                 0.970518                  0.96620               0.9449              0.43144
# Prevalence                  0.030077         0.11080                 0.034961                  0.09152               0.1512              0.58149
# Detection Rate              0.001285         0.03599                 0.006427                  0.06015               0.1355              0.07198
# Detection Prevalence        0.016195         0.06015                 0.032134                  0.07224               0.7154              0.10386
# Balanced Accuracy           0.513681         0.64883                 0.578593                  0.82200               0.6065              0.52381

###########################################

# SVM prediciton
###########################################
svmPredFULL <- predict(svmFULL, testSetFULL)
postResample(svmPredFULL, testSetFULL$iphonesentiment)
#Accuracy     Kappa 
#0.7133676 0.4184614 

CMsvmFULL <- confusionMatrix(svmPredFULL, testSetFULL$iphonesentiment)
CMsvmFULL
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 0       87                 1                 0             4            12
# Somewhat Negative        0        1                 3                 0             4             2
# Somewhat Positive        1        2                17               103             3             3
# Very Negative            1       10                 1                14           355            18
# Very Positive          115      331               114               239           222          2227
# 
# Overall Statistics
# 
# Accuracy : 0.7134          
# 95% CI : (0.6989, 0.7275)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4185          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.20186                0.0220588                  0.28933              0.60374               0.9845
# Specificity                  1.00000         0.99509                0.9981353                  0.99264              0.98667               0.3729
# Pos Pred Value                   NaN         0.83654                0.3000000                  0.79845              0.88972               0.6857
# Neg Pred Value               0.96992         0.90914                0.9657216                  0.93273              0.93326               0.9455
# Prevalence                   0.03008         0.11080                0.0349614                  0.09152              0.15116               0.5815
# Detection Rate               0.00000         0.02237                0.0007712                  0.02648              0.09126               0.5725
# Detection Prevalence         0.00000         0.02674                0.0025707                  0.03316              0.10257               0.8350
# Balanced Accuracy            0.50000         0.59847                0.5100971                  0.64098              0.79521               0.6787

###########################################

#####################################################


# iPhoneDataCOR
#####################################################

# C5.0 prediction
###########################################
c5.0PredCOR <- predict(c5.0COR, testSetCOR)
postResample(c5.0PredCOR, testSetCOR$iphonesentiment)
# Accuracy     Kappa 
# 0.7485861 0.5060495

# Create a confusion matrix from C5.0 predictions 
CMc5.0COR <- confusionMatrix(c5.0PredCOR, testSetCOR$iphonesentiment)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 0      139                 1                 1             3            19
# Somewhat Negative        0        0                18                 0             0             0
# Somewhat Positive        1        4                 2               168             5            20
# Very Negative            0        3                 0                 4           373             9
# Very Positive          116      285               115               183           207          2214
# 
# Overall Statistics
# 
# Accuracy : 0.7486          
# 95% CI : (0.7346, 0.7622)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.506           
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.32251                 0.132353                  0.47191              0.63435               0.9788
# Specificity                  1.00000         0.99306                 1.000000                  0.99095              0.99515               0.4435
# Pos Pred Value                   NaN         0.85276                 1.000000                  0.84000              0.95887               0.7096
# Neg Pred Value               0.96992         0.92165                 0.969525                  0.94905              0.93859               0.9377
# Prevalence                   0.03008         0.11080                 0.034961                  0.09152              0.15116               0.5815
# Detection Rate               0.00000         0.03573                 0.004627                  0.04319              0.09589               0.5692
# Detection Prevalence         0.00000         0.04190                 0.004627                  0.05141              0.10000               0.8021
# Balanced Accuracy            0.50000         0.65778                 0.566176                  0.73143              0.81475               0.7111

###########################################

# RF prediction
###########################################
rfPredCOR <- predict(rfCOR, testSetCOR)
postResample(rfPredCOR, testSetCOR$iphonesentiment)
# Accuracy     Kappa 
# 0.7514139 0.5119913

# Create a confusion matrix from random forest predictions 
CMrfCOR <- confusionMatrix(rfPredCOR, testSetCOR$iphonesentiment) 
CMrfCOR
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 1        1                 0                 0             0             1
# Positive                 0      137                 0                 1             0            11
# Somewhat Negative        0        0                18                 0             0             0
# Somewhat Positive        2        9                 2               172             4            19
# Very Negative            0        4                 2                 1           376            12
# Very Positive          114      280               114               182           208          2219
# 
# Overall Statistics
# 
# Accuracy : 0.7514          
# 95% CI : (0.7375, 0.7649)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.512           
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0085470         0.31787                 0.132353                  0.48315              0.63946               0.9810
# Specificity                0.9994699         0.99653                 1.000000                  0.98981              0.99425               0.4484
# Pos Pred Value             0.3333333         0.91946                 1.000000                  0.82692              0.95190               0.7119
# Neg Pred Value             0.9701569         0.92141                 0.969525                  0.95003              0.93934               0.9444
# Prevalence                 0.0300771         0.11080                 0.034961                  0.09152              0.15116               0.5815
# Detection Rate             0.0002571         0.03522                 0.004627                  0.04422              0.09666               0.5704
# Detection Prevalence       0.0007712         0.03830                 0.004627                  0.05347              0.10154               0.8013
# Balanced Accuracy          0.5040085         0.65720                 0.566176                  0.73648              0.81685               0.7147

###########################################

#####################################################


# iPhoneDataNZV
#####################################################

# C5.0 prediction
###########################################
c5.0PredNZV <- predict(c5.0NZV, testSetNZV)
postResample(c5.0PredNZV, testSetNZV$iphonesentiment)
# Accuracy     Kappa 
# 0.7573265 0.5238451 

# Create a confusion matrix from C5.0 predictions 
CMc5.0NZV <- confusionMatrix(c5.0PredNZV, testSetNZV$iphonesentiment)
CMc5.0NZV
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 1      147                 2                 1             2            15
# Somewhat Negative        0        0                 0                 0             0             0
# Somewhat Positive        1        4                 1               191             0             6
# Very Negative            0        2                18                 2           377            10
# Very Positive          115      278               115               162           209          2231
# 
# Overall Statistics
# 
# Accuracy : 0.7573          
# 95% CI : (0.7435, 0.7707)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5238          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.34107                  0.00000                  0.53652              0.64116               0.9863
# Specificity                  1.00000         0.99393                  1.00000                  0.99660              0.99031               0.4601
# Pos Pred Value                   NaN         0.87500                      NaN                  0.94089              0.92176               0.7174
# Neg Pred Value               0.96992         0.92370                  0.96504                  0.95525              0.93939               0.9603
# Prevalence                   0.03008         0.11080                  0.03496                  0.09152              0.15116               0.5815
# Detection Rate               0.00000         0.03779                  0.00000                  0.04910              0.09692               0.5735
# Detection Prevalence         0.00000         0.04319                  0.00000                  0.05219              0.10514               0.7995
# Balanced Accuracy            0.50000         0.66750                  0.50000                  0.76656              0.81573               0.7232

###########################################

# RF prediction
###########################################
rfPredNZV <- predict(rfNZV, testSetNZV)
postResample(rfPredNZV, testSetNZV$iphonesentiment)
# Accuracy     Kappa 
# 0.7578406 0.5234571 

# Create a confusion matrix from random forest predictions 
CMrfNZV <- confusionMatrix(rfPredNZV, testSetNZV$iphonesentiment) 
CMrfNZV
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             1
# Positive                 0      140                 0                 3             1             8
# Somewhat Negative        0        0                 0                 0             0             1
# Somewhat Positive        0        2                 1               191             1             4
# Very Negative            0        3                20                 1           380            11
# Very Positive          117      286               115               161           206          2237
# 
# Overall Statistics
# 
# Accuracy : 0.7578          
# 95% CI : (0.7441, 0.7712)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5235          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0000000         0.32483                0.0000000                  0.53652              0.64626               0.9889
# Specificity                0.9997350         0.99653                0.9997336                  0.99774              0.98940               0.4564
# Pos Pred Value             0.0000000         0.92105                0.0000000                  0.95980              0.91566               0.7165
# Neg Pred Value             0.9699151         0.92215                0.9650296                  0.95530              0.94014               0.9674
# Prevalence                 0.0300771         0.11080                0.0349614                  0.09152              0.15116               0.5815
# Detection Rate             0.0000000         0.03599                0.0000000                  0.04910              0.09769               0.5751
# Detection Prevalence       0.0002571         0.03907                0.0002571                  0.05116              0.10668               0.8026
# Balanced Accuracy          0.4998675         0.66068                0.4998668                  0.76713              0.81783               0.7227

###########################################

#####################################################


# iPhoneRFE
#####################################################

# C5.0 prediction
###########################################
c5.0PredRFE <- predict(c5.0RFE, testSetRFE)
postResample(c5.0PredRFE, testSetRFE$iphonesentiment)
# Accuracy     Kappa 
# 0.7691517 0.5516730 

# Create a confusion matrix from C5.0 predictions 
CMc5.0RFE <- confusionMatrix(c5.0PredRFE, testSetRFE$iphonesentiment)
CMc5.0RFE
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 1      139                 0                 3             6            16
# Somewhat Negative        0        1                18                 0             0             0
# Somewhat Positive        1        5                 2               229             4             3
# Very Negative            0        3                 1                 2           373            10
# Very Positive          115      283               115               122           205          2233
# 
# Overall Statistics
# 
# Accuracy : 0.7692          
# 95% CI : (0.7556, 0.7823)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5517          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.32251                 0.132353                  0.64326              0.63435               0.9872
# Specificity                  1.00000         0.99248                 0.999734                  0.99576              0.99515               0.4840
# Pos Pred Value                   NaN         0.84242                 0.947368                  0.93852              0.95887               0.7267
# Neg Pred Value               0.96992         0.92161                 0.969517                  0.96517              0.93859               0.9645
# Prevalence                   0.03008         0.11080                 0.034961                  0.09152              0.15116               0.5815
# Detection Rate               0.00000         0.03573                 0.004627                  0.05887              0.09589               0.5740
# Detection Prevalence         0.00000         0.04242                 0.004884                  0.06272              0.10000               0.7900
# Balanced Accuracy            0.50000         0.65749                 0.566043                  0.81951              0.81475               0.7356

###########################################

# RF prediction
###########################################
rfPredRFE <- predict(rfRFE, testSetRFE)
postResample(rfPredRFE, testSetRFE$iphonesentiment)
# Accuracy     Kappa 
# 0.7753213 0.5639055  

# Create a confusion matrix from random forest predictions 
CMrfRFE <- confusionMatrix(rfPredRFE, testSetRFE$iphonesentiment) 
CMrfRFE
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 1        1                 0                 0             0             1
# Positive                 0      141                 0                 1             1             4
# Somewhat Negative        0        0                18                 0             0             2
# Somewhat Positive        0        5                 2               238             2             6
# Very Negative            0        2                 2                 1           380            11
# Very Positive          116      282               114               116           205          2238
# 
# Overall Statistics
# 
# Accuracy : 0.7753          
# 95% CI : (0.7619, 0.7884)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5639          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0085470         0.32715                 0.132353                  0.66854              0.64626               0.9894
# Specificity                0.9994699         0.99827                 0.999467                  0.99576              0.99515               0.4883
# Pos Pred Value             0.3333333         0.95918                 0.900000                  0.94071              0.95960               0.7288
# Neg Pred Value             0.9701569         0.92252                 0.969509                  0.96756              0.94047               0.9707
# Prevalence                 0.0300771         0.11080                 0.034961                  0.09152              0.15116               0.5815
# Detection Rate             0.0002571         0.03625                 0.004627                  0.06118              0.09769               0.5753
# Detection Prevalence       0.0007712         0.03779                 0.005141                  0.06504              0.10180               0.7895
# Balanced Accuracy          0.5040085         0.66271                 0.565910                  0.83215              0.82071               0.7389

###########################################

#####################################################


# iPhoneRC
#####################################################

# C5.0 prediction
###########################################
c5.0PredRC <- predict(c5.0RC, testSetRC)
postResample(c5.0PredRC, testSetRC$iphonesentiment)
# Accuracy     Kappa 
# 0.8478149 0.6214419

# Create a confusion matrix from C5.0 predictions 
CMc5.0RC <- confusionMatrix(c5.0PredRC, testSetRC$iphonesentiment)
CMc5.0RC
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive
# Negative                 377       15                 2                 3
# Positive                 322     2667               114               119
# Somewhat Negative          0        0                20                 0
# Somewhat Positive          6       11                 0               234
# 
# Overall Statistics
# 
# Accuracy : 0.8478         
# 95% CI : (0.8361, 0.859)
# No Information Rate : 0.6923         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.6214         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive
# Sensitivity                  0.53475          0.9903                 0.147059                  0.65730
# Specificity                  0.99372          0.5363                 1.000000                  0.99519
# Pos Pred Value               0.94962          0.8277                 1.000000                  0.93227
# Neg Pred Value               0.90610          0.9611                 0.970026                  0.96647
# Prevalence                   0.18123          0.6923                 0.034961                  0.09152
# Detection Rate               0.09692          0.6856                 0.005141                  0.06015
# Detection Prevalence         0.10206          0.8283                 0.005141                  0.06452
# Balanced Accuracy            0.76424          0.7633                 0.573529                  0.82625

###########################################

# RF prediction
###########################################
rfPredRC <- predict(rfRc, testSetRC)
postResample(rfPredRC, testSetRC$iphonesentiment)
# Accuracy     Kappa 
# 0.8503856 0.6290911 

# Create a confusion matrix from random forest predictions 
CMrfRC <- confusionMatrix(rfPredRC, testSetRC$iphonesentiment) 
CMrfRC
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive
# Negative                 385       17                 2                 1
# Positive                 318     2664               114               116
# Somewhat Negative          0        0                20                 0
# Somewhat Positive          2       12                 0               239
# 
# Overall Statistics
# 
# Accuracy : 0.8504          
# 95% CI : (0.8388, 0.8615)
# No Information Rate : 0.6923          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.6291          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive
# Sensitivity                  0.54610          0.9892                 0.147059                  0.67135
# Specificity                  0.99372          0.5422                 1.000000                  0.99604
# Pos Pred Value               0.95062          0.8294                 1.000000                  0.94466
# Neg Pred Value               0.90818          0.9572                 0.970026                  0.96783
# Prevalence                   0.18123          0.6923                 0.034961                  0.09152
# Detection Rate               0.09897          0.6848                 0.005141                  0.06144
# Detection Prevalence         0.10411          0.8257                 0.005141                  0.06504
# Balanced Accuracy            0.76991          0.7657                 0.573529                  0.83369



###########################################

#####################################################


# iPhonePCA
#####################################################

# RF prediction
###########################################
rfpredPCA <- predict(rfPCA, test.pca)
postResample(rfpredPCA, test.pca$iphonesentiment)
# Accuracy     Kappa 
# 0.7598972 0.5391556 

# Create a confusion matrix from random forest predictions 
CMrfPCA <- confusionMatrix(rfpredPCA, test.pca$iphonesentiment) 
CMrfPCA
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 1        1                 1                 0             1             9
# Positive                 2      134                 2                 0             3            25
# Somewhat Negative        0        0                19                 0             1             4
# Somewhat Positive        0        2                 2               230             1             9
# Very Negative            0       12                 3                 6           378            21
# Very Positive          114      282               109               120           204          2194
# 
# Overall Statistics
# 
# Accuracy : 0.7599          
# 95% CI : (0.7462, 0.7732)
# No Information Rate : 0.5815          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5392          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0085470         0.31090                 0.139706                  0.64607              0.64286               0.9699
# Specificity                0.9968195         0.99075                 0.998668                  0.99604              0.98728               0.4908
# Pos Pred Value             0.0769231         0.80723                 0.791667                  0.94262              0.90000               0.7258
# Neg Pred Value             0.9700800         0.92025                 0.969736                  0.96544              0.93948               0.9216
# Prevalence                 0.0300771         0.11080                 0.034961                  0.09152              0.15116               0.5815
# Detection Rate             0.0002571         0.03445                 0.004884                  0.05913              0.09717               0.5640
# Detection Prevalence       0.0033419         0.04267                 0.006170                  0.06272              0.10797               0.7771
# Balanced Accuracy          0.5026833         0.65083                 0.569187                  0.82105              0.81507               0.7304

###########################################

#####################################################

######################################################################################


## ----- Predict sentiment in iphone large matrix ----- ##
######################################################################################

## ----- SELECTED DATA ----- ##
iPhoneLM_prediction <- predict(rfRc, iPhoneLM)

str(iPhoneLM_prediction)
summary(iPhoneLM_prediction)
# Negative          Positive Somewhat Negative Somewhat Positive 
#    16474              7837              1410               419 

output <- iPhoneLM
str(output)
output$iphonesentiment <- iPhoneLM_prediction

write.csv(output, file = "C4T3 iPhone Large Matrix.csv", row.names = TRUE)
iPhonePredicted <- read.csv("C4T3 iPhone Large Matrix.csv")


plot_ly(iPhonePredicted, 
        x= ~iPhonePredicted$iphonesentiment, 
        type='histogram')

# NZV - test
##########################

iPhoneLM_prediction_NZV <- predict(rfNZV, iPhoneLM_NZV)
summary(iPhoneLM_prediction_NZV)
# Negative          Positive Somewhat Negative Somewhat Positive     Very Negative     Very Positive 
#        0                 1                 0               444             17880              7815 

##########################

## ----- SELECTED DATA ----- ##
iPhoneLM_prediction_ex <- predict(rfRc, iPhoneLM_ex)

str(iPhoneLM_prediction_ex)
summary(iPhoneLM_prediction_ex)

######################################################################################


## ----- Visual of iPhone, galaxy and large matrix ----- ##
######################################################################################

# Pie charts
#####################################################
# create a data frame for plotting.
# you can add more sentiment levels if needed
# Replace sentiment values 
pieDataiPhoneRC <- data.frame(COM = c("Negative", "Positive", "Somewhat Negative", "Somewhat Positive"),
                            values = c(2352, 8979, 454, 1188))

pieDataiPhoneLM <- data.frame(COM = c("Negative", "Positive", "Somewhat Negative", "Somewhat Positive"),
                              values = c(16474, 7837, 1410, 419))


# create pie chart

# iPhone recoded data set
plot_ly(pieDataiPhoneRC, labels = ~COM, values = ~values, type = "pie",
        textposition = 'inside',
        textinfo = 'percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors, line = list(color = '#FFFFFF', width = 1)),
        showlegend = T) %>%
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# iPhone recoded data set large matrix
plot_ly(pieDataiPhoneLM, labels = ~COM, values = ~values, type = "pie",
        textposition = 'inside',
        textinfo = 'percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors, line = list(color = '#FFFFFF', width = 1)),
        showlegend = T) %>%
  layout(title = 'iPhone Sentiment Large Matrix', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

# plot_ly() %>%
#   add_pie(data = pieDataGalaxyLM, labels = ~COM, values = ~values, name = 'Galaxy') %>%
#   add_pie(data = pieDataiPhoneLM, labels = ~COM, values = ~values, name = 'iPhone') %>%
#   layout(title = "Pie Charts", showlegend = T,
#          xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
#          yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

#####################################################

# Bar charts
#####################################################
plot_ly(galaxyPredicted, x= ~galaxyPredicted$galaxysentiment, type='histogram', name = "Galaxy") %>%
  add_trace(iPhonePredicted, x = ~iPhonePredicted$iphonesentiment, name = 'iPhone') %>%
  layout(
    title = 'Predicted Sentiment Towards Preferred Smart Phones',
    xaxis = list(title = 'Sentimet'),
    yaxis = list(title = 'Count'),
    barmode = 'group')

#####################################################

######################################################################################