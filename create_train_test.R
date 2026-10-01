##### LOAD required packages
library(caret)

##### SET WORKING DIRECTORY TO MMFRAMEWORK #####
## setwd("D:/OneDrive/Projects/FF66/LCMS-Net/npy data")

# --------------------------------------------------------
# 1. CREATE Train-test sets
# --------------------------------------------------------
set.seed(14)
dt = read.csv("D:/OneDrive/Projects/FF66/LCMS-Net/856_1024/Default_raw/NorPD/meta_NorESKD.csv", stringsAsFactors = T) #load input data

## 1.1 Create training sets by Resample 70/30, 10 times
TrainList_biclass = createDataPartition(dt$Class, p=0.7, times=10) #training sets

## or 1.1 Create training sets by repeated 3-fold cross-validation 5 times
#TrainList_biclass = createMultiFolds(dt$Class, k=3, times=5) #training sets

## 1.2 Create test sets
TestList_biclass = lapply(TrainList_biclass, function(idx) dt$Index[-idx]) #test sets
lapply(TrainList_biclass, print) #preview Train-test lists

# --------------------------------------------------------
# 2. EXPORT Train-test sets
# --------------------------------------------------------
## 2.1 Export training sets
lines1 = sapply(names(TrainList_biclass), function(nm) {
  value <- TrainList_biclass[[nm]]
  paste(nm, ":", toString(value))
})
writeLines(lines1, "D:/OneDrive/Projects/FF66/LCMS-Net/856_1024/Default_raw/NorPD/TrainList_biclass.txt") #training sets

## 2.2 Export test sets
lines2 = sapply(names(TestList_biclass), function(nm) {
  value <- TestList_biclass[[nm]]
  paste(nm, ":", toString(value))
})
writeLines(lines2, "D:/OneDrive/Projects/FF66/LCMS-Net/856_1024/Default_raw/NorPD/TestList_biclass.txt") #test sets
