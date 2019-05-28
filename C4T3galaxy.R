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


# Import Galaxy data set for training
#####################################################
galaxy <- read.csv("galaxy_smallmatrix_labeled_9d.csv")

#####################################################

# Import Galaxy LargeMatrix data set
#####################################################
galaxyLM <- read.csv("galaxyLargeMatrix.csv")

#####################################################


######################################################################################


## ----- EVALUTE DATA ----- ##
######################################################################################

# Galaxy data set
#####################################################
summary(galaxy)
str(galaxy) #'data.frame':	12911 obs. of  59 variables:


## ------ Evaluate NA values ----------##

# Are there any NAs in df?
?is.na
any(is.na(galaxy)) 
# Use summary to identify where NAs are 
#summary(iPhoneata)

#####################################################

# Galaxy LargeMatrix data set
#####################################################
summary(galaxyLM)
str(galaxyLM)
# 'data.frame':	26140 obs. of  60 variables:
#####################################################

######################################################################################


## ----- PROPROCESS ----- ##
######################################################################################

# Galaxy data set
#####################################################

# Recode values (galaxy)
###########################################
galaxy$galaxysentiment[galaxy$galaxysentiment == 0] <- "Very Negative"
galaxy$galaxysentiment[galaxy$galaxysentiment == 1] <- "Negative"
galaxy$galaxysentiment[galaxy$galaxysentiment == 2] <- "Somewhat Negative"
galaxy$galaxysentiment[galaxy$galaxysentiment == 3] <- "Somewhat Positive"
galaxy$galaxysentiment[galaxy$galaxysentiment == 4] <- "Positive"
galaxy$galaxysentiment[galaxy$galaxysentiment == 5] <- "Very Positive"

# CHANGE DATA TYPES 
galaxy$galaxysentiment <- as.factor(galaxy$galaxysentiment)
str(galaxy)

plot_ly(galaxy, x= ~galaxy$galaxysentiment, type='histogram', name = "Galaxy") %>%
  add_trace(iPhoneData, x = ~iPhoneData$iphonesentiment, name = 'iPhone') %>%
  layout(
    title = 'Sentiment Towards Preferred Smart Phones',
    xaxis = list(title = 'Sentimet'),
    yaxis = list(title = 'Count'),
    barmode = 'group')
  

###########################################

# Examine correlation (galaxyCOR)
###########################################
galaxy_corr <- cor(galaxy[,1:58])
corrplot(galaxy_corr, method = "circle", type="upper", order="hclust")
galaxy_corr
galaxy_corr.high <- findCorrelation(galaxy_corr, cutoff=0.8, names = TRUE, exact = TRUE)
galaxy_corr.high
# [1] "samsungdisneg" "samsungperneg" "samsungdispos" "htcdisneg"     "googleperneg"  "googleperpos" 
# [7] "samsungdisunc" "samsungcamunc" "htcperpos"     "nokiacamunc"   "nokiadisneg"   "nokiadispos"  
# [13] "nokiaperunc"   "nokiacampos"   "nokiadisunc"   "nokiaperneg"   "nokiacamneg"   "iphonedisneg" 
# [19] "iphonedispos"  "sonyperpos"    "iosperunc"     "iosperneg"     "sonydisneg"    "ios"          
# [25] "htcphone"  

# Create dataframe from which the features will be removed from
galaxyCOR <- galaxy
str(galaxyCOR)
galaxyCOR$samsungdisneg <- NULL
galaxyCOR$samsungperneg <- NULL
galaxyCOR$samsungdispos <- NULL
galaxyCOR$htcdisneg <- NULL
galaxyCOR$googleperneg <- NULL
galaxyCOR$googleperpos <- NULL

galaxyCOR$samsungdisunc <- NULL
galaxyCOR$samsungcamunc <- NULL
galaxyCOR$htcperpos <- NULL
galaxyCOR$nokiacamunc <- NULL
galaxyCOR$nokiadisneg <- NULL
galaxyCOR$nokiadispos <- NULL

galaxyCOR$nokiaperunc <- NULL
galaxyCOR$nokiacampos <- NULL
galaxyCOR$nokiadisunc <- NULL
galaxyCOR$nokiaperneg <- NULL
galaxyCOR$nokiacamneg <- NULL
galaxyCOR$iphonedisneg <- NULL

galaxyCOR$iphonedispos <- NULL
galaxyCOR$sonydispos <- NULL
galaxyCOR$iosperunc <- NULL
galaxyCOR$iosperneg <- NULL
galaxyCOR$sonydisneg <- NULL
galaxyCOR$ios <- NULL

galaxyCOR$htcphone <- NULL

str(galaxyCOR)

galaxy.corr2 <- cor(galaxyCOR[,1:33])
corrplot(galaxy.corr2, method = "circle", type="upper", order="hclust")
galaxy.corr2

###########################################

# Examine Feature Variance (galaxyNZV)
###########################################
galaxynzvMetrics <- nearZeroVar(galaxy, saveMetrics = TRUE)
galaxynzvMetrics
# freqRatio percentUnique zeroVar   nzv
# iphone             5.039313    0.20912400   FALSE FALSE
# samsunggalaxy     14.090164    0.05421733   FALSE FALSE
# sonyxperia        44.111888    0.03872667   FALSE  TRUE
# nokialumina      495.500000    0.02323600   FALSE  TRUE
# htcphone          11.427740    0.06970800   FALSE FALSE
# ios               27.662132    0.04647200   FALSE  TRUE
# googleandroid     61.248780    0.04647200   FALSE  TRUE
# iphonecampos      10.526217    0.23236000   FALSE FALSE
# samsungcampos     93.176471    0.08519867   FALSE  TRUE
# sonycampos       347.081081    0.05421733   FALSE  TRUE
# nokiacampos     1841.285714    0.08519867   FALSE  TRUE
# htccampos         79.401274    0.17039734   FALSE  TRUE
# iphonecamneg      19.660473    0.13167067   FALSE  TRUE
# samsungcamneg     99.648438    0.06970800   FALSE  TRUE
# sonycamneg      1842.428571    0.04647200   FALSE  TRUE
# nokiacamneg     2148.500000    0.06196267   FALSE  TRUE
# htccamneg         92.992593    0.11618000   FALSE  TRUE
# iphonecamunc      16.805436    0.16265200   FALSE FALSE
# samsungcamunc     73.953488    0.06970800   FALSE  TRUE
# sonycamunc       585.545455    0.03872667   FALSE  TRUE
# nokiacamunc     2578.800000    0.05421733   FALSE  TRUE
# htccamunc         50.510040    0.12392533   FALSE  TRUE
# iphonedispos       6.797333    0.24785067   FALSE FALSE
# samsungdispos     96.595420    0.13167067   FALSE  TRUE
# sonydispos       329.512821    0.06196267   FALSE  TRUE
# nokiadispos     1431.888889    0.09294400   FALSE  TRUE
# htcdispos         64.383420    0.20137867   FALSE  TRUE
# iphonedisneg      10.104816    0.18588800   FALSE FALSE
# samsungdisneg     98.674419    0.10843467   FALSE  TRUE
# sonydisneg      2149.000000    0.06970800   FALSE  TRUE
# nokiadisneg     1841.285714    0.08519867   FALSE  TRUE
# htcdisneg         88.063380    0.14716134   FALSE  TRUE
# iphonedisunc      11.527865    0.20912400   FALSE FALSE
# samsungdisunc     74.333333    0.09294400   FALSE  TRUE
# sonydisunc       757.941176    0.05421733   FALSE  TRUE
# nokiadisunc     1611.625000    0.04647200   FALSE  TRUE
# htcdisunc         50.757085    0.13941600   FALSE  TRUE
# iphoneperpos       9.299184    0.18588800   FALSE FALSE
# samsungperpos     93.748148    0.10843467   FALSE  TRUE
# sonyperpos       414.903226    0.06196267   FALSE  TRUE
# nokiaperpos     2147.666667    0.08519867   FALSE  TRUE
# htcperpos         74.371257    0.19363334   FALSE  TRUE
# iphoneperneg      11.037910    0.17039734   FALSE FALSE
# samsungperneg    101.158730    0.10068933   FALSE  TRUE
# sonyperneg      2149.333333    0.07745333   FALSE  TRUE
# nokiaperneg     3221.750000    0.09294400   FALSE  TRUE
# htcperneg         93.969925    0.15490667   FALSE  TRUE
# iphoneperunc      13.034602    0.12392533   FALSE FALSE
# samsungperunc     86.087838    0.09294400   FALSE  TRUE
# sonyperunc      3225.000000    0.04647200   FALSE  TRUE
# nokiaperunc     1841.571429    0.06970800   FALSE  TRUE
# htcperunc         50.015936    0.15490667   FALSE  TRUE
# iosperpos        152.626506    0.09294400   FALSE  TRUE
# googleperpos      98.115385    0.06970800   FALSE  TRUE
# iosperneg        141.055556    0.09294400   FALSE  TRUE
# googleperneg      98.922481    0.08519867   FALSE  TRUE
# iosperunc        135.234043    0.07745333   FALSE  TRUE
# googleperunc      95.977444    0.07745333   FALSE  TRUE
# galaxysentiment    4.593750    0.04647200   FALSE FALSE

# nearZeroVar() with saveMetrics = FALSE returns an vector 
str(galaxy)
galaxynzv <- nearZeroVar(galaxy, saveMetrics = FALSE) 
galaxynzv
# [1]  3  4  6  7  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39 40
# [32] 41 42 44 45 46 47 49 50 51 52 53 54 55 56 57 58

# Data set with nzv features removed
galaxyNZV <- galaxy[,-galaxynzv]
str(galaxyNZV)
# 'data.frame':	12911 obs. of  12 variables:
#   $ iphone         : int  1 1 1 0 1 2 1 1 4 1 ...
# $ samsunggalaxy  : int  0 0 1 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 1 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 1 0 0 1 0 0 0 0 ...
# $ iphonecamunc   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonedispos   : int  0 1 0 0 0 0 2 0 0 0 ...
# $ iphonedisneg   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ iphonedisunc   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ iphoneperpos   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperneg   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperunc   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ galaxysentiment: Factor w/ 6 levels "Negative","Positive",..: 6 4 4 5 1 5 4 6 6 6 ...

###########################################

# Recursive Feature Elimination (galaxyRFE)
###########################################
# Sample the data before using RFE
set.seed(123)
galaxySample <- galaxy[sample(1:nrow(galaxy), 1000, replace=FALSE),]

# Use the same control (ctrl) as the ones used in iPhoneRFE


# Use rfe and omit the response variable (attribute 59 iphonesentiment)
galaxyrfeResults <- rfe(galaxySample[,1:58],
                        galaxySample$galaxysentiment,
                        rfeControl=ctrl)
galaxyrfeResults
# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold, repeated 5 times) 
# 
# Resampling performance over subset size:
#   
#   Variables Accuracy  Kappa AccuracySD KappaSD Selected
#           4   0.7163 0.3691    0.03001 0.07542         
#           8   0.7349 0.4261    0.03078 0.07456         
#          16   0.7511 0.4809    0.03099 0.06967        *
#          58   0.7374 0.4225    0.03185 0.08034         

# The top 5 variables (out of 16):
#   iphone, samsunggalaxy, googleandroid, iphonedispos, iphonedisunc

predictors(galaxyrfeResults)
# [1] "iphone"        "samsunggalaxy" "googleandroid" "iphonedispos"  "iphonedisunc"  "iphonedisneg" 
# [7] "htcphone"      "iphonecamunc"  "ios"           "iphoneperpos"  "iphonecamneg"  "htcdispos"    
# [13] "iphoneperunc"  "sonyxperia"    "htccampos"     "iphoneperneg" 

plot(galaxyrfeResults, type = c('g', 'o'))

# Data set with RFE recommended features
galaxyRFE <- galaxy[,predictors(galaxyrfeResults)]
# Add the dependant variable (iphonesentiment)
galaxyRFE$galaxysentiment <- galaxy$galaxysentiment

str(galaxyRFE)
# 'data.frame':	12911 obs. of  17 variables:

###########################################

# Recode and reduce levels to 4 (galaxyRC)
###########################################
## Recode values since some levels have poor sensitivity and balanced accuracy
# create a new dataset that will be used for recoding sentiment
galaxyRC <- galaxy
summary(galaxyRC[,59])
# recode (dplyr package) sentiment to combine factor levels 0 & 1 and 4 & 5
galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, 
                                   'Very Negative' = 'Negative',
                                   'Negative' = 'Negative',
                                   'Somewhat Negative' = 'Somewhat Negative',
                                   'Somewhat Positive' = 'Somewhat Positive',
                                   'Positive' = 'Positive',
                                   'Very Positive' = 'Positive') 
# inspect results
summary(galaxyRC[,59])
str(galaxyRC)
# $ iphonesentiment: Factor w/ 4 levels "Negative","Positive",..: 1 1 1 1 1 2 2 1 1 1 ...

plot_ly(galaxyRC, x= ~galaxyRC$galaxysentiment, type='histogram', name = "Galaxy") %>%
  layout(
    title = 'Sentiment Towards Preferred Smart Phones',
    xaxis = list(title = 'Sentimet'),
    yaxis = list(title = 'Count'),
    barmode = 'group')

## To add two bar charts
# plot_ly(galaxyRC, x= ~galaxyRC$galaxysentiment, type='histogram', name = "Galaxy") %>%
#   add_trace(iPhoneRC, x = ~iPhoneRC$iphonesentiment, name = 'iPhone') %>%
#   layout(
#     title = 'Sentiment Towards Preferred Smart Phones',
#     xaxis = list(title = 'Sentimet'),
#     yaxis = list(title = 'Count'),
#     barmode = 'group')

###########################################

# Recoded w/reduced level=4 and Recursive Feature Elimination (galaxyRC.RFE)
###########################################
# Sample the data before using RFE
set.seed(123)
galaxyRC.Sample <- galaxyRC[sample(1:nrow(galaxyRC), 1000, replace=FALSE),]

# use the same control from the RFE data set (ctrl)

# Use rfe and omit the response variable (attribute 59 iphonesentiment)
galaxyrfeRC.Results <- rfe(galaxyRC.Sample[,1:58],
                           galaxySample$galaxysentiment,
                           rfeControl=ctrl)
galaxyrfeRC.Results
# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold, repeated 5 times) 
# 
# Resampling performance over subset size:
#   
#   Variables Accuracy     Kappa AccuracySD KappaSD Selected
# 4   0.6136 -0.009652    0.01299 0.01571         
# 8   0.6179 -0.003356    0.01045 0.01339        *
#   16   0.6125 -0.005725    0.01307 0.01843         
# 58   0.6174 -0.002489    0.01071 0.01320         
# 
# The top 5 variables (out of 8):
#   htccampos, samsungdispos, iphonedispos, iphonecampos, iosperneg

predictors(galaxyrfeRC.Results)
# [1] "htccampos"     "samsungdispos" "iphonedispos"  "iphonecampos"  "iosperneg"     "htcphone"     
# [7] "iphoneperpos"  "iphonecamunc" 

plot(galaxyrfeRC.Results, type = c('g', 'o'))

# Data set with RFE recommended features
galaxyRC.RFE <- galaxyRC[,predictors(galaxyrfeRC.Results)]
# Add the dependant variable (iphonesentiment)
galaxyRC.RFE$galaxysentiment <- galaxyRC$galaxysentiment

str(galaxyRC.RFE)
# 'data.frame':	12911 obs. of  9 variables:

###########################################

#####################################################

# Galaxy Large Matrix data set
#####################################################
galaxyLM$id <- NULL
galaxyLM$galaxysentiment <- as.factor(galaxyLM$galaxysentiment)


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


## ----- MODEL DEVELOPMENT GALAXY ----- ## 
######################################################################################

# Galaxy
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingFULLgalaxy <- createDataPartition(galaxy$galaxysentiment,
                                            p = 0.70,
                                            list = FALSE)
trainSetFULLgalaxy <- galaxy[inTrainingFULLgalaxy,] 
str(trainSetFULLgalaxy) # 'data.frame':	9040 obs. of  59 variables:
testSetFULLgalaxy <- galaxy[-inTrainingFULLgalaxy,]  
str(testSetFULLgalaxy) # 'data.frame':	3871 obs. of  59 variables:

###########################################

# C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0FULLgalaxy <- train(galaxysentiment~.,
                                    data=trainSetFULLgalaxy,
                                    method = "C5.0",
                                    trControl = fitControl,
                                    tuneLength = 1))
# user  system elapsed 
# 2.73    0.19   31.28

c5.0FULLgalaxy
# C5.0 
# 
# 9040 samples
# 58 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7631661  0.5237177
# rules   TRUE   0.7633875  0.5252765
# tree   FALSE   0.7630557  0.5247198
# tree    TRUE   0.7629449  0.5246037
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfFULLgalaxy <- train(galaxysentiment~.,
                                  data=trainSetFULLgalaxy,
                                  method = "rf",
                                  trControl = fitControl,
                                  tuneLength = 1))
# user  system elapsed 
# 50.13    0.25  316.85 

rfFULLgalaxy
# Random Forest 
# 
# 9040 samples
# 58 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7669247  0.5331563
# 
# Tuning parameter 'mtry' was held constant at a value of 19

###########################################

# kNN TRAIN/FIT
###########################################
set.seed(123)
system.time(kknnFULLgalaxy <- train(galaxysentiment~.,
                                    data=trainSetFULLgalaxy,
                                    method = "kknn",
                                    trControl = fitControl,
                                    tuneLength = 1))
# user  system elapsed 
# 8.61    0.17   51.14 

kknnFULLgalaxy
# k-Nearest Neighbors 
# 
# 9040 samples
# 58 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.6964642  0.4422451
# 
# Tuning parameter 'kmax' was held constant at a value of 5
# Tuning parameter 'distance' was held constant
# at a value of 2
# Tuning parameter 'kernel' was held constant at a value of optimal

###########################################

# SVM (e1071) TRAIN/FIT
###########################################
set.seed(123)
system.time(svmFULLgalaxy <- train(galaxysentiment~.,
                                   data=trainSetFULLgalaxy,
                                   method = "svmLinear2",
                                   trControl = fitControl,
                                   tuneLength = 1))
# user  system elapsed 
# 28.72    0.42  156.55 

svmFULLgalaxy
# Support Vector Machines with Linear Kernel 
# 
# 9040 samples
# 58 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy  Kappa    
# 0.70088   0.3594216
# 
# Tuning parameter 'cost' was held constant at a value of 0.25

###########################################


#####################################################

# galaxyCOR
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingCORgalaxy <- createDataPartition(galaxyCOR$galaxysentiment,
                                           p = 0.70,
                                           list = FALSE)
trainSetCORgalaxy <- galaxyCOR[inTrainingCORgalaxy,] 
str(trainSetCORgalaxy) # 'data.frame':	9040 obs. of  34 variables:
testSetCORgalaxy <- galaxyCOR[-inTrainingCORgalaxy,]  
str(testSetCORgalaxy) # 'data.frame':	3871 obs. of  34 variables:

###########################################

# C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0CORgalaxy <- train(galaxysentiment~.,
                                   data=trainSetCORgalaxy,
                                   method = "C5.0",
                                   trControl = fitControl,
                                   tuneLength = 1))
# user  system elapsed 
# 1.92    0.09   18.11 

c5.0CORgalaxy
# C5.0 
# 
# 9040 samples
# 33 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7475695  0.4869098
# rules   TRUE   0.7474589  0.4860718
# tree   FALSE   0.7474587  0.4871017
# tree    TRUE   0.7470165  0.4858537
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfCORgalaxy <- train(galaxysentiment~.,
                                 data=trainSetCORgalaxy,
                                 method = "rf",
                                 trControl = fitControl,
                                 tuneLength = 1))
# user  system elapsed 
# 22.74    0.23  135.04   

rfCORgalaxy
# Random Forest 
# 
# 9040 samples
# 33 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy  Kappa    
# 0.749669  0.4936488
# 
# Tuning parameter 'mtry' was held constant at a value of 11

###########################################

#####################################################

# galaxyNZV
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingNZVgalaxy <- createDataPartition(galaxyNZV$galaxysentiment,
                                           p = 0.70,
                                           list = FALSE)
trainSetNZVgalaxy <- galaxyNZV[inTrainingNZVgalaxy,] 
str(trainSetNZVgalaxy) # 'data.frame':	9040 obs. of  12 variables:
testSetNZVgalaxy <- galaxyNZV[-inTrainingNZVgalaxy,]  
str(testSetNZVgalaxy) # 'data.frame':	3871 obs. of  12 variables:

###########################################

# C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0NZVgalaxy <- train(galaxysentiment~.,
                                   data=trainSetNZVgalaxy,
                                   method = "C5.0",
                                   trControl = fitControl,
                                   tuneLength = 1))
# user  system elapsed 
# 0.88    0.00   10.12   

c5.0CORgalaxy
# C5.0 
# 
# 9040 samples
# 33 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7475695  0.4869098
# rules   TRUE   0.7474589  0.4860718
# tree   FALSE   0.7474587  0.4871017
# tree    TRUE   0.7470165  0.4858537
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfNZVgalaxy <- train(galaxysentiment~.,
                                 data=trainSetNZVgalaxy,
                                 method = "rf",
                                 trControl = fitControl,
                                 tuneLength = 1))
# user  system elapsed 
# 4.81    0.07   27.17    

rfNZVgalaxy
# Random Forest 
# 
# 9040 samples
# 11 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy   Kappa   
# 0.7529872  0.498193
# 
# Tuning parameter 'mtry' was held constant at a value of 3

###########################################

#####################################################


# galaxyRFE
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingRFEgalaxy <- createDataPartition(galaxyRFE$galaxysentiment, p = 0.70, list = FALSE)
trainSetRFEgalaxy <- galaxyRFE[inTrainingRFEgalaxy,] 
str(trainSetRFEgalaxy) # 'data.frame':	9040 obs. of  17 variables:
testSetRFEgalaxy <- galaxyRFE[-inTrainingRFEgalaxy,]  
str(testSetRFEgalaxy) # 'data.frame':	3871 obs. of  17 variables:

###########################################

#C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0RFEgalaxy <- train(galaxysentiment~.,
                                   data=trainSetRFEgalaxy,
                                   method = "C5.0",
                                   trControl = fitControl,
                                   tuneLength = 1))
# user  system elapsed 
# 1.37    0.05   10.75

c5.0RFEgalaxy
# C5.0 
# 
# 9040 samples
# 16 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  Accuracy   Kappa    
# rules  FALSE   0.7611744  0.5212349
# rules   TRUE   0.7615065  0.5218998
# tree   FALSE   0.7612854  0.5222923
# tree    TRUE   0.7612854  0.5221854
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

###########################################

#RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfRFEgalaxy <- train(galaxysentiment~.,
                                 data=trainSetRFEgalaxy,
                                 method = "rf",
                                 trControl = fitControl,
                                 tuneLength = 1))
# user  system elapsed 
# 10.97    0.29   56.11   

rfRFEgalaxy
# Random Forest 
# 
# 9040 samples
# 16 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.7650456  0.5304036
# 
# Tuning parameter 'mtry' was held constant at a value of 5


###########################################

#####################################################


#galaxyRC
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingRCgalaxy <- createDataPartition(galaxyRC$galaxysentiment, p = 0.70, list = FALSE)

trainSetRCgalaxy <- galaxyRC[inTrainingRCgalaxy,] 
str(trainSetRCgalaxy) # 'data.frame':	9039 obs. of  59 variables:

testSetRCgalaxy <- galaxyRC[-inTrainingRCgalaxy,]  
str(testSetRCgalaxy) # 'data.frame':	3872 obs. of  59 variables:

###########################################

# C5.0 TRAIN/FIT
###########################################
set.seed(123)
system.time(c5.0RCgalaxy <- train(galaxysentiment~.,
                                  data=trainSetRCgalaxy,
                                  method = "C5.0",
                                  trControl = fitControl,
                                  tuneLength = 1))
# user  system elapsed 
# 1.97    0.00   26.08 

c5.0RCgalaxy
# C5.0 
# 
# 9039 samples
# 58 predictor
# 4 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8135, 8136, 8134, 8135, 8134, ... 
# Resampling results across tuning parameters:
#   
# model  winnow  Accuracy   Kappa    
# rules  FALSE   0.8456711  0.5964520
# rules   TRUE   0.8457814  0.5976947 *
# tree   FALSE   0.8442323  0.5946568
# tree    TRUE   0.8443427  0.5955625
# 
# Tuning parameter 'trials' was held constant at a value of 1
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfRcgalaxy <- train(galaxysentiment~.,
                                data=trainSetRCgalaxy,
                                method = "rf",
                                trControl = fitControl,
                                tuneLength = 1))
# user  system elapsed 
# 37.89    0.09  274.17  

rfRcgalaxy
# Random Forest 
# 
# 9039 samples
# 58 predictor
# 4 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8135, 8136, 8134, 8135, 8134, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.8473304  0.6017154
# 
# Tuning parameter 'mtry' was held constant at a value of 19


###########################################

#####################################################


# galaxyRC.RFE
#####################################################

# Data partition
###########################################
set.seed(123) # set random seed
inTrainingRC.RFEgalaxy <- createDataPartition(galaxyRC.RFE$galaxysentiment, p = 0.70, list = FALSE)
trainSetRC.RFEgalaxy <- galaxyRC.RFE[inTrainingRC.RFEgalaxy,] 
str(trainSetRC.RFEgalaxy) # 'data.frame':	9039 obs. of  9 variables:
testSetRC.RFEgalaxy <- galaxyRC.RFE[-inTrainingRC.RFEgalaxy,]  
str(testSetRC.RFEgalaxy) # 'data.frame':	3872 obs. of  9 variables:

###########################################

# RF TRAIN/FIT
###########################################
set.seed(123)
system.time(rfRC.RFEgalaxy <- train(galaxysentiment~.,
                                    data=trainSetRC.RFEgalaxy,
                                    method = "rf",
                                    trControl = fitControl,
                                    tuneLength = 1))
# user  system elapsed 
# 3.00    0.20   14.23

rfRC.RFEgalaxy
# Random Forest 
# 
# 9039 samples
# 8 predictor
# 4 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8135, 8135, 8136, 8134, 8135, 8134, ... 
# Resampling results:
#   
#   Accuracy  Kappa    
# 0.762582  0.2863776
# 
# Tuning parameter 'mtry' was held constant at a value of 2


###########################################

#####################################################


# galaxy - principal components analysis (galaxyPCA)
#####################################################

# Create train/test usiing iPhone full data set
###########################################

# data = training and testing from galaxyDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams.galaxy <- preProcess(trainSetFULLgalaxy[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams.galaxy)
# Created from 9040 samples and 58 variables
# 
# Pre-processing:
#   - centered (58)
# - ignored (0)
# - principal component signal extraction (58)
# - scaled (58)
# 
# PCA needed 23 components to capture 95 percent of the variance


# use predict to apply pca parameters, create training, exclude dependant
train.pca.galaxy <- predict(preprocessParams.galaxy, trainSetFULLgalaxy[,-59])

# add the dependent to training
train.pca.galaxy$galaxysentiment <- trainSetFULLgalaxy$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca.galaxy <- predict(preprocessParams.galaxy, testSetFULLgalaxy[,-59])

# add the dependent to training
test.pca.galaxy$galaxysentiment <- testSetFULLgalaxy$galaxysentiment

# inspect results
str(train.pca.galaxy)
str(test.pca.galaxy)

###########################################

#  TRAIN/FIT
###########################################
set.seed(123)
system.time(rfPCA.galaxy <- train(galaxysentiment~.,
                                  data=train.pca.galaxy,
                                  method = "rf",
                                  trControl = fitControl,
                                  tuneLength = 1))
# user  system elapsed 
# 17.44    0.48  103.24 

rfPCA.galaxy
# Random Forest 
# 
# 9040 samples
# 23 predictor
# 6 classes: 'Negative', 'Positive', 'Somewhat Negative', 'Somewhat Positive', 'Very Negative', 'Very Positive' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 1 times) 
# Summary of sample sizes: 8136, 8137, 8137, 8135, 8137, 8136, ... 
# Resampling results:
#   
#   Accuracy  Kappa    
# 0.752877  0.5066563
# 
# Tuning parameter 'mtry' was held constant at a value of 7


###########################################

#####################################################

######################################################################################


## ----- EVALUATE MODELS ----- ##
######################################################################################

# galaxy
#####################################################

# USE RESAMPLES TO COMPARE MODEL PERFORMANCE
galaxyFullResults <- resamples(list(CompleteData_C5.0 = c5.0FULLgalaxy,
                                    CompleteData_RandomForest = rfFULLgalaxy,
                                    CompleteData_kNN = kknnFULLgalaxy,
                                    CompleteData_SVM = svmFULLgalaxy,
                                    NZVariance_C5.0 = c5.0NZVgalaxy,
                                    NZVariance_RandomForest = rfNZVgalaxy,
                                    CorrelationFE_C5.0 = c5.0CORgalaxy,
                                    CorrelationFE_RandomForest = rfCORgalaxy,
                                    RFE_C5.0 = c5.0RFEgalaxy,
                                    RFE_RandomForest = rfRFEgalaxy,
                                    RCLevelFactors_C5.0 = c5.0RCgalaxy,
                                    RCLevelFactors_RandomForest = rfRcgalaxy,
                                    RCLevelFactors.RFE_RandomForest = rfRC.RFEgalaxy,
                                    PCA_RandomForest = rfPCA.galaxy))
summary(galaxyFullResults)
# Call:
#   summary.resamples(object = galaxyFullResults)
# 
# Models: CompleteData_C5.0, CompleteData_RandomForest, CompleteData_kNN, CompleteData_SVM, NZVariance_C5.0, NZVariance_RandomForest, CorrelationFE_C5.0, CorrelationFE_RandomForest, RFE_C5.0, RFE_RandomForest, RCLevelFactors_C5.0, RCLevelFactors_RandomForest, RCLevelFactors.RFE_RandomForest, PCA_RandomForest 
# Number of resamples: 10 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# CompleteData_C5.0               0.7497231 0.7566298 0.7645101 0.7633875 0.7696567 0.7765487    0
# CompleteData_RandomForest       0.7530454 0.7646409 0.7671439 0.7669247 0.7696567 0.7820796    0
# CompleteData_kNN                0.6386740 0.6871252 0.7011627 0.6964642 0.7134565 0.7215470    0
# CompleteData_SVM                0.6877076 0.6901993 0.6981757 0.7008800 0.7067415 0.7245575    0
# NZVariance_C5.0                 0.7375415 0.7430939 0.7509678 0.7502232 0.7562178 0.7632743    0
# NZVariance_RandomForest         0.7408638 0.7486188 0.7541528 0.7529872 0.7562260 0.7654867    0
# CorrelationFE_C5.0              0.7325967 0.7439227 0.7482023 0.7475695 0.7523531 0.7585825    0
# CorrelationFE_RandomForest      0.7359116 0.7465462 0.7511071 0.7496690 0.7536677 0.7610619    0
# RFE_C5.0                        0.7508306 0.7569061 0.7624585 0.7615065 0.7645758 0.7740864    0
# RFE_RandomForest                0.7541528 0.7604972 0.7670174 0.7650456 0.7692315 0.7743363    0
# RCLevelFactors_C5.0             0.8241150 0.8392394 0.8501943 0.8457814 0.8536650 0.8571429    0
# RCLevelFactors_RandomForest     0.8263274 0.8403462 0.8490894 0.8473304 0.8566293 0.8585635    0  *
# RCLevelFactors.RFE_RandomForest 0.7444690 0.7586493 0.7636952 0.7625820 0.7686142 0.7745856    0
# PCA_RandomForest                0.7436464 0.7490309 0.7523499 0.7528770 0.7556659 0.7654867    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# CompleteData_C5.0               0.4898139 0.5077090 0.5274540 0.5252765 0.5409615 0.5611250    0
# CompleteData_RandomForest       0.4976112 0.5240720 0.5346649 0.5331563 0.5404891 0.5743513    0
# CompleteData_kNN                0.3676711 0.4165824 0.4600882 0.4422451 0.4655360 0.4777232    0
# CompleteData_SVM                0.3202513 0.3385644 0.3541687 0.3594216 0.3707742 0.4182754    0
# NZVariance_C5.0                 0.4545267 0.4725947 0.4993344 0.4923946 0.5065534 0.5295537    0
# NZVariance_RandomForest         0.4639155 0.4847060 0.5035605 0.4981930 0.5068978 0.5340002    0
# CorrelationFE_C5.0              0.4458361 0.4771866 0.4867560 0.4869098 0.5015568 0.5219103    0
# CorrelationFE_RandomForest      0.4533477 0.4829938 0.5001809 0.4936488 0.5023040 0.5286461    0
# RFE_C5.0                        0.4949062 0.5070919 0.5246707 0.5218998 0.5298331 0.5506056    0
# RFE_RandomForest                0.5005332 0.5180549 0.5323317 0.5304036 0.5406146 0.5602526    0
# RCLevelFactors_C5.0             0.5317283 0.5814518 0.6118155 0.5976947 0.6216257 0.6290530    0
# RCLevelFactors_RandomForest     0.5345020 0.5847429 0.6098737 0.6017154 0.6258275 0.6366151    0  *
# RCLevelFactors.RFE_RandomForest 0.2168170 0.2732231 0.2966682 0.2863776 0.3080222 0.3218135    0
# PCA_RandomForest                0.4802781 0.4935787 0.5059562 0.5066563 0.5171544 0.5428637    0

#####################################################

######################################################################################


## ----- Predict galaxy test data sets----- ##
######################################################################################

# galaxy
#####################################################

# C5.0 prediction
###########################################
c5.0PredFULLgalaxy <- predict(c5.0FULLgalaxy, testSetFULLgalaxy)
postResample(c5.0PredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7700852 0.5403752

# Create a confusion matrix from C5.0 predictions 
CMc5.0FULLgalaxy <- confusionMatrix(c5.0PredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
CMc5.0FULLgalaxy
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 1      141                 1                 3             6            15
# Somewhat Negative        0        0                15                 0             1             3
# Somewhat Positive        0        4                 2               204             6            20
# Very Negative            2        8                 1                 3           353            31
# Very Positive          111      272               116               142           142          2268
# 
# Overall Statistics
# 
# Accuracy : 0.7701          
# 95% CI : (0.7565, 0.7833)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5404          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.33176                 0.111111                  0.57955              0.69488               0.9705
# Specificity                  1.00000         0.99246                 0.998929                  0.99091              0.98662               0.4896
# Pos Pred Value                   NaN         0.84431                 0.789474                  0.86441              0.88693               0.7434
# Neg Pred Value               0.97055         0.92333                 0.968847                  0.95928              0.95537               0.9159
# Prevalence                   0.02945         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate               0.00000         0.03642                 0.003875                  0.05270              0.09119               0.5859
# Detection Prevalence         0.00000         0.04314                 0.004908                  0.06097              0.10282               0.7882
# Balanced Accuracy            0.50000         0.66211                 0.555020                  0.78523              0.84075               0.7300

###########################################

# RF prediction
###########################################
rfPredFULLgalaxy <- predict(rfFULLgalaxy, testSetFULLgalaxy)
postResample(rfPredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7716352 0.5426724 

# Create a confusion matrix from random forest predictions 
CMrfFULLgalaxy <- confusionMatrix(rfPredFULLgalaxy, testSetFULLgalaxy$galaxysentiment) 
CMrfFULLgalaxy
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        1                 1                 0             0             0
# Positive                 1      139                 0                 2             6            10
# Somewhat Negative        0        0                15                 0             1             2
# Somewhat Positive        0        3                 1               209             4            19
# Very Negative            2        7                 2                 2           352            34
# Very Positive          111      275               116               139           145          2272
# 
# Overall Statistics
# 
# Accuracy : 0.7716          
# 95% CI : (0.7581, 0.7848)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5427          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0000000         0.32706                 0.111111                  0.59375              0.69291               0.9722
# Specificity                0.9994677         0.99449                 0.999197                  0.99233              0.98602               0.4876
# Pos Pred Value             0.0000000         0.87975                 0.833333                  0.88559              0.88221               0.7430
# Neg Pred Value             0.9705350         0.92297                 0.968855                  0.96066              0.95507               0.9200
# Prevalence                 0.0294498         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate             0.0000000         0.03591                 0.003875                  0.05399              0.09093               0.5869
# Detection Prevalence       0.0005167         0.04082                 0.004650                  0.06097              0.10307               0.7900
# Balanced Accuracy          0.4997338         0.66077                 0.555154                  0.79304              0.83947               0.7299

###########################################

# kNN prediction
###########################################
kknnPredFULLgalaxy <- predict(kknnFULLgalaxy, testSetFULLgalaxy)
postResample(kknnPredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7016275 0.4541651

CMkknnFULLgalaxy <- confusionMatrix(kknnPredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
CMkknnFULLgalaxy
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 4        9                 6                 7             5            58
# Positive                 5      139                 5                 9            17           109
# Somewhat Negative        1        5                15                 0             2            24
# Somewhat Positive        2       12                 3               210             7            73
# Very Negative            4       13                 5                 4           343            68
# Very Positive           98      247               101               122           134          2005
# 
# Overall Statistics
# 
# Accuracy : 0.7016         
# 95% CI : (0.6869, 0.716)
# No Information Rate : 0.6037         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.4542         
# Mcnemar's Test P-Value : < 2.2e-16      
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                 0.035088         0.32706                 0.111111                  0.59659              0.67520               0.8579
# Specificity                 0.977376         0.95792                 0.991435                  0.97244              0.97205               0.5424
# Pos Pred Value              0.044944         0.48944                 0.319149                  0.68404              0.78490               0.7407
# Neg Pred Value              0.970915         0.92027                 0.968619                  0.96016              0.95195               0.7148
# Prevalence                  0.029450         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate              0.001033         0.03591                 0.003875                  0.05425              0.08861               0.5180
# Detection Prevalence        0.022991         0.07337                 0.012142                  0.07931              0.11289               0.6993
# Balanced Accuracy           0.506232         0.64249                 0.551273                  0.78451              0.82362               0.7002

###########################################

# SVM prediciton
###########################################
svmPredFULLgalaxy <- predict(svmFULLgalaxy, testSetFULLgalaxy)
postResample(svmPredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7005942 0.3556399

CMsvmFULLgalaxy <- confusionMatrix(svmPredFULLgalaxy, testSetFULLgalaxy$galaxysentiment)
CMsvmFULLgalaxy
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 0       82                 1                 1             4             2
# Somewhat Negative        1        0                 3                 0             1             3
# Somewhat Positive        1        2                13                87            43            12
# Very Negative            2        9                 1                 5           244            24
# Very Positive          110      332               117               259           216          2296
# 
# Overall Statistics
# 
# Accuracy : 0.7006         
# 95% CI : (0.6859, 0.715)
# No Information Rate : 0.6037         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.3556         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.19294                 0.022222                  0.24716              0.48031               0.9825
# Specificity                  1.00000         0.99768                 0.998662                  0.97982              0.98781               0.3259
# Pos Pred Value                   NaN         0.91111                 0.375000                  0.55063              0.85614               0.6895
# Neg Pred Value               0.97055         0.90928                 0.965830                  0.92863              0.92638               0.9242
# Prevalence                   0.02945         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate               0.00000         0.02118                 0.000775                  0.02247              0.06303               0.5931
# Detection Prevalence         0.00000         0.02325                 0.002067                  0.04082              0.07362               0.8602
# Balanced Accuracy            0.50000         0.59531                 0.510442                  0.61349              0.73406               0.6542

###########################################

#####################################################


# galaxyCOR
#####################################################

# C5.0 prediction
###########################################
c5.0PredCORgalaxy <- predict(c5.0CORgalaxy, testSetCORgalaxy)
postResample(c5.0PredCORgalaxy, testSetCORgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7501937 0.4932975 

# Create a confusion matrix from C5.0 predictions 
CMc5.0CORgalaxy <- confusionMatrix(c5.0PredCORgalaxy, testSetCORgalaxy$galaxysentiment)
CMc5.0CORgalaxy
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 0      129                 1                 4             5            10
# Somewhat Negative        0        0                15                 0             1             3
# Somewhat Positive        2        4                 3               147             7            34
# Very Negative            3        8                 2                 2           354            31
# Very Positive          109      284               114               199           141          2259
# 
# Overall Statistics
# 
# Accuracy : 0.7502          
# 95% CI : (0.7362, 0.7638)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4933          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.30353                 0.111111                  0.41761              0.69685               0.9666
# Specificity                  1.00000         0.99420                 0.998929                  0.98579              0.98632               0.4478
# Pos Pred Value                   NaN         0.86577                 0.789474                  0.74619              0.88500               0.7273
# Neg Pred Value               0.97055         0.92047                 0.968847                  0.94420              0.95563               0.8980
# Prevalence                   0.02945         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate               0.00000         0.03332                 0.003875                  0.03797              0.09145               0.5836
# Detection Prevalence         0.00000         0.03849                 0.004908                  0.05089              0.10333               0.8024
# Balanced Accuracy            0.50000         0.64886                 0.555020                  0.70170              0.84159               0.7072

###########################################

# RF prediction
###########################################
rfPredCORgalaxy <- predict(rfCORgalaxy, testSetCORgalaxy)
postResample(rfPredCORgalaxy, testSetCORgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7507104 0.4975297

# Create a confusion matrix from random forest predictions 
CMrfCORgalaxy <- confusionMatrix(rfPredCORgalaxy, testSetCORgalaxy$galaxysentiment) 
CMrfCORgalaxy
# Confusion Matrix and Statistics
# 
#                    Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 0      136                 0                 3             6            14
# Somewhat Negative        1        0                15                 0             1             3
# Somewhat Positive        2        3                 3               152             7            41
# Very Negative            2        8                 2                 2           356            32
# Very Positive          109      278               115               195           138          2247
# 
# Overall Statistics
# 
# Accuracy : 0.7507          
# 95% CI : (0.7368, 0.7643)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.4975          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.32000                 0.111111                  0.43182              0.70079               0.9615
# Specificity                  1.00000         0.99333                 0.998662                  0.98409              0.98632               0.4557
# Pos Pred Value                   NaN         0.85535                 0.750000                  0.73077              0.88557               0.7291
# Neg Pred Value               0.97055         0.92214                 0.968839                  0.94540              0.95618               0.8859
# Prevalence                   0.02945         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate               0.00000         0.03513                 0.003875                  0.03927              0.09197               0.5805
# Detection Prevalence         0.00000         0.04107                 0.005167                  0.05373              0.10385               0.7962
# Balanced Accuracy            0.50000         0.65666                 0.554886                  0.70795              0.84355               0.7086

###########################################

#####################################################


# galaxyNZV
#####################################################

# C5.0 prediction
###########################################
c5.0PredNZVgalaxy <- predict(c5.0NZVgalaxy, testSetNZVgalaxy)
postResample(c5.0PredNZVgalaxy, testSetNZVgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7571687 0.5082188 

# Create a confusion matrix from C5.0 predictions 
CMc5.0NZVgalaxy <- confusionMatrix(c5.0PredNZVgalaxy, testSetNZVgalaxy$galaxysentiment)
CMc5.0NZVgalaxy
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 1      134                 0                 5             5             9
# Somewhat Negative        0        0                 0                 0             0             0
# Somewhat Positive        0        5                 0               169             3            20
# Very Negative            3        9                17                 5           355            35
# Very Positive          110      277               118               173           145          2273
# 
# Overall Statistics
# 
# Accuracy : 0.7572          
# 95% CI : (0.7433, 0.7706)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5082          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.31529                  0.00000                  0.48011              0.69882               0.9726
# Specificity                  1.00000         0.99420                  1.00000                  0.99204              0.97948               0.4635
# Pos Pred Value                   NaN         0.87013                      NaN                  0.85787              0.83726               0.7342
# Neg Pred Value               0.97055         0.92171                  0.96513                  0.95019              0.95561               0.9174
# Prevalence                   0.02945         0.10979                  0.03487                  0.09093              0.13123               0.6037
# Detection Rate               0.00000         0.03462                  0.00000                  0.04366              0.09171               0.5872
# Detection Prevalence         0.00000         0.03978                  0.00000                  0.05089              0.10953               0.7998
# Balanced Accuracy            0.50000         0.65475                  0.50000                  0.73608              0.83915               0.7181

###########################################

# RF prediction
###########################################
rfPredNZVgalaxy <- predict(rfNZVgalaxy, testSetNZVgalaxy)
postResample(rfPredNZVgalaxy, testSetNZVgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7618187 0.5178267 

# Create a confusion matrix from random forest predictions 
CMrfNZVgalaxy <- confusionMatrix(rfPredNZVgalaxy, testSetNZVgalaxy$galaxysentiment) 
CMrfNZVgalaxy
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 1                 0             0             0
# Positive                 1      141                 1                 2             4             9
# Somewhat Negative        0        0                 0                 0             0             1
# Somewhat Positive        0        4                 0               176             3            14
# Very Negative            2       10                16                 2           354            35
# Very Positive          111      270               117               172           147          2278
# 
# Overall Statistics
# 
# Accuracy : 0.7618          
# 95% CI : (0.7481, 0.7752)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5178          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                0.0000000         0.33176                0.0000000                  0.50000              0.69685               0.9748
# Specificity                0.9997338         0.99507                0.9997323                  0.99403              0.98067               0.4674
# Pos Pred Value             0.0000000         0.89241                0.0000000                  0.89340              0.84487               0.7360
# Neg Pred Value             0.9705426         0.92351                0.9651163                  0.95210              0.95539               0.9240
# Prevalence                 0.0294498         0.10979                0.0348747                  0.09093              0.13123               0.6037
# Detection Rate             0.0000000         0.03642                0.0000000                  0.04547              0.09145               0.5885
# Detection Prevalence       0.0002583         0.04082                0.0002583                  0.05089              0.10824               0.7995
# Balanced Accuracy          0.4998669         0.66342                0.4998662                  0.74702              0.83876               0.7211

###########################################

#####################################################

# galaxyRFE
#####################################################

# C5.0 prediction
###########################################
c5.0PredRFEgalaxy <- predict(c5.0RFEgalaxy, testSetRFEgalaxy)
postResample(c5.0PredRFEgalaxy, testSetRFEgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7675019 0.5379182

# Create a confusion matrix from C5.0 predictions 
CMc5.0RFEgalaxy <- confusionMatrix(c5.0PredRFEgalaxy, testSetRFEgalaxy$galaxysentiment)
CMc5.0RFEgalaxy
# Confusion Matrix and Statistics
# 
# Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 0                 0             0             0
# Positive                 2      142                 3                 5             8            18
# Somewhat Negative        0        0                15                 0             1             3
# Somewhat Positive        0        3                 0               205             4            30
# Very Negative            3        9                 1                 2           356            33
# Very Positive          109      271               116               140           139          2253
# 
# Overall Statistics
# 
# Accuracy : 0.7675          
# 95% CI : (0.7539, 0.7807)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5379          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
# Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                  0.00000         0.33412                 0.111111                  0.58239              0.70079               0.9641
# Specificity                  1.00000         0.98955                 0.998929                  0.98949              0.98573               0.4948
# Pos Pred Value                   NaN         0.79775                 0.789474                  0.84711              0.88119               0.7441
# Neg Pred Value               0.97055         0.92337                 0.968847                  0.95949              0.95616               0.9004
# Prevalence                   0.02945         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate               0.00000         0.03668                 0.003875                  0.05296              0.09197               0.5820
# Detection Prevalence         0.00000         0.04598                 0.004908                  0.06252              0.10437               0.7822
# Balanced Accuracy            0.50000         0.66184                 0.555020                  0.78594              0.84326               0.7294

###########################################

# RF prediction
###########################################
rfPredRFEgalaxy <- predict(rfRFEgalaxy, testSetRFEgalaxy)
postResample(rfPredRFEgalaxy, testSetRFEgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7700852 0.5405174

# Create a confusion matrix from random forest predictions 
CMrfRFEgalaxy <- confusionMatrix(rfPredRFEgalaxy, testSetRFEgalaxy$galaxysentiment) 
CMrfRFEgalaxy
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        1                 1                 0             1             0
# Positive                 1      135                 0                 2             5             7
# Somewhat Negative        0        0                15                 0             1             2
# Somewhat Positive        0        4                 0               208             4            27
# Very Negative            2        9                 2                 2           357            35
# Very Positive          111      276               117               140           140          2266
# 
# Overall Statistics
# 
# Accuracy : 0.7701          
# 95% CI : (0.7565, 0.7833)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5405          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                 0.000000         0.31765                 0.111111                  0.59091              0.70276               0.9696
# Specificity                 0.999201         0.99565                 0.999197                  0.99005              0.98513               0.4889
# Pos Pred Value              0.000000         0.90000                 0.833333                  0.85597              0.87715               0.7430
# Neg Pred Value              0.970527         0.92206                 0.968855                  0.96031              0.95641               0.9135
# Prevalence                  0.029450         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate              0.000000         0.03487                 0.003875                  0.05373              0.09222               0.5854
# Detection Prevalence        0.000775         0.03875                 0.004650                  0.06277              0.10514               0.7879
# Balanced Accuracy           0.499601         0.65665                 0.555154                  0.79048              0.84394               0.7293

###########################################

#####################################################


# galaxyRC
#####################################################

# C5.0 prediction
###########################################
c5.0PredRCgalaxy <- predict(c5.0RCgalaxy, testSetRCgalaxy)
postResample(c5.0PredRCgalaxy, testSetRCgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.8401343 0.5855297

# Create a confusion matrix from C5.0 predictions 
CMc5.0RCgalaxy <- confusionMatrix(c5.0PredRCgalaxy, testSetRCgalaxy$galaxysentiment)
CMc5.0RCgalaxy
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive
# Negative               351       44                 1                12
# Positive               269     2687               112               144
# Somewhat Negative        0        1                20                 1
# Somewhat Positive        3       30                 2               195
# 
# Overall Statistics
# 
# Accuracy : 0.8401          
# 95% CI : (0.8282, 0.8515)
# No Information Rate : 0.7133          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5855          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive
# Sensitivity                  0.56340          0.9728                 0.148148                  0.55398
# Specificity                  0.98246          0.5270                 0.999465                  0.99006
# Pos Pred Value               0.86029          0.8366                 0.909091                  0.84783
# Neg Pred Value               0.92148          0.8864                 0.970130                  0.95689
# Prevalence                   0.16090          0.7133                 0.034866                  0.09091
# Detection Rate               0.09065          0.6940                 0.005165                  0.05036
# Detection Prevalence         0.10537          0.8295                 0.005682                  0.05940
# Balanced Accuracy            0.77293          0.7499                 0.573806                  0.77202

###########################################

# RF prediction
###########################################
rfPredRCgalaxy <- predict(rfRcgalaxy, testSetRCgalaxy)
postResample(rfPredRCgalaxy, testSetRCgalaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.8434917 0.5957018 

# Create a confusion matrix from random forest predictions 
CMrfRCgalaxy <- confusionMatrix(rfPredRCgalaxy, testSetRCgalaxy$galaxysentiment) 
CMrfRCgalaxy
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive
# Negative               352       40                 1                 3
# Positive               269     2686               110               140
# Somewhat Negative        1        4                20                 1
# Somewhat Positive        1       32                 4               208
# 
# Overall Statistics
# 
# Accuracy : 0.8435          
# 95% CI : (0.8317, 0.8548)
# No Information Rate : 0.7133          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5957          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive
# Sensitivity                  0.56501          0.9725                 0.148148                  0.59091
# Specificity                  0.98646          0.5324                 0.998394                  0.98949
# Pos Pred Value               0.88889          0.8381                 0.769231                  0.84898
# Neg Pred Value               0.92204          0.8861                 0.970099                  0.96030
# Prevalence                   0.16090          0.7133                 0.034866                  0.09091
# Detection Rate               0.09091          0.6937                 0.005165                  0.05372
# Detection Prevalence         0.10227          0.8277                 0.006715                  0.06327
# Balanced Accuracy            0.77573          0.7525                 0.573271                  0.79020

###########################################

#####################################################


# galaxyPCA
#####################################################

# RF prediction
###########################################
rfpredPCAgalaxy <- predict(rfPCA.galaxy, test.pca.galaxy)
postResample(rfpredPCAgalaxy, test.pca.galaxy$galaxysentiment)
# Accuracy     Kappa 
# 0.7540687 0.5130775 

# Create a confusion matrix from random forest predictions 
CMrfPCAgalaxy <- confusionMatrix(rfpredPCAgalaxy, test.pca.galaxy$galaxysentiment) 
CMrfPCAgalaxy
# Confusion Matrix and Statistics
# 
#                     Reference
# Prediction          Negative Positive Somewhat Negative Somewhat Positive Very Negative Very Positive
# Negative                 0        0                 1                 1             1             2
# Positive                 2      133                 2                 3             8            32
# Somewhat Negative        0        0                15                 0             2            11
# Somewhat Positive        0        4                 0               198             3            22
# Very Negative            5        9                 2                 2           351            48
# Very Positive          107      279               115               148           143          2222
# 
# Overall Statistics
# 
# Accuracy : 0.7541          
# 95% CI : (0.7402, 0.7676)
# No Information Rate : 0.6037          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5131          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Negative Class: Positive Class: Somewhat Negative Class: Somewhat Positive Class: Very Negative Class: Very Positive
# Sensitivity                 0.000000         0.31294                 0.111111                  0.56250              0.69094               0.9508
# Specificity                 0.998669         0.98636                 0.996520                  0.99176              0.98037               0.4837
# Pos Pred Value              0.000000         0.73889                 0.535714                  0.87225              0.84173               0.7372
# Neg Pred Value              0.970512         0.92089                 0.968774                  0.95774              0.95455               0.8658
# Prevalence                  0.029450         0.10979                 0.034875                  0.09093              0.13123               0.6037
# Detection Rate              0.000000         0.03436                 0.003875                  0.05115              0.09067               0.5740
# Detection Prevalence        0.001292         0.04650                 0.007233                  0.05864              0.10772               0.7786
# Balanced Accuracy           0.499335         0.64965                 0.553816                  0.77713              0.83566               0.7172

###########################################

#####################################################

######################################################################################


## ----- Predict sentiment in galaxy large matrix ----- ##
######################################################################################

galaxyLM_prediction <- predict(rfRcgalaxy, galaxyLM)
summary(galaxyLM_prediction)
# Negative          Positive Somewhat Negative Somewhat Positive 
#    16466              7848              1410               416

outputGalaxy <- galaxyLM
str(outputGalaxy)
outputGalaxy$galaxysentiment <- galaxyLM_prediction

write.csv(outputGalaxy, file = "C4T3 Galaxy Large Matrix.csv", row.names = TRUE)
galaxyPredicted <- read.csv("C4T3 Galaxy Large Matrix.csv")


plot_ly(galaxyPredicted, 
        x= ~galaxyPredicted$galaxysentiment, 
        type='histogram')

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
pieDataGalaxyRC <- data.frame(COM = c("Negative", "Positive", "Somewhat Negative", "Somewhat Positive"),
                            values = c(2078, 9208, 450, 1175))
pieDataiPhoneLM <- data.frame(COM = c("Negative", "Positive", "Somewhat Negative", "Somewhat Positive"),
                              values = c(16474, 7837, 1410, 419))
pieDataGalaxyLM <- data.frame(COM = c("Negative", "Positive", "Somewhat Negative", "Somewhat Positive"),
                              values = c(16466, 7848, 1410, 416))


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

# Galaxy recoded data set
plot_ly(pieDataGalaxyRC, labels = ~COM, values = ~values, type = "pie",
        textposition = 'inside',
        textinfo = 'percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors, line = list(color = '#FFFFFF', width = 1)),
        showlegend = T) %>%
  layout(title = 'Galaxy Sentiment', 
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

# galaxy recoded data set large matrix
plot_ly(pieDataGalaxyLM, labels = ~COM, values = ~values, type = "pie",
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

plot_ly() %>%
  add_pie(data = pieDataGalaxyLM, labels = ~COM, values = ~values, name = 'Galaxy') %>%
  add_pie(data = pieDataiPhoneLM, labels = ~COM, values = ~values, name = 'iPhone') %>%
  layout(title = "Pie Charts", showlegend = T,
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))

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