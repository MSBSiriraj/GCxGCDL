library(metabox2)
library(caret)
library(pROC)
library(dplyr)
library(optparse)



# -----------------------------
# 01. Apply arguments

# -----------------------------
setwd(wkpath)
method = proc
if (!dir.exists(outpath)) {
  dir.create(outpath, recursive = TRUE)
}

# -----------------------------
# 0. Initialize
# -----------------------------
source("MMFramework_balanced.R") #general functions
filenamepath = paste0(outpath,"/",method,"_pls_");
################## Setup input ##################
dt = read.csv(input,stringsAsFactors = T) #load input data Index|Class|FeaturesXx
#training sets
trainlines <- readLines(train)
TrainList <- lapply(trainlines, function(x) {
  parts <- strsplit(x, " : ")[[1]]
  as.numeric(strsplit(parts[2], ",\\s*")[[1]])
})
names(TrainList) <- sub(" :.*", "", trainlines)
#test sets
testlines <- readLines(test)
TestList <- lapply(testlines, function(x) {
  parts <- strsplit(x, " : ")[[1]]
  as.numeric(strsplit(parts[2], ",\\s*")[[1]])
})
names(TestList) <- sub(" :.*", "", testlines)

idCol=1; classCol=2; xCol=3; #input data Index|Class|FeaturesXx
binary = ifelse(nlevels(dt[,2]) > 2, FALSE, TRUE) #classification type

## 1. RF model
set.seed(14)
fitControl = trainControl(method = "repeatedcv", number = 3, repeats = 5, savePredictions = T) #Repeated 3-fold CV; 70/30 for model optimization
out_model = list(); out_perform = data.frame(); out_imp = data.frame(); imp_list = list(); stat_list = list();
for(i in 1:length(TrainList)){
  cat("Running ..", names(TrainList)[i], "..")
  #setup data
  inTraining_raw = dt[dt$Index %in% TrainList[[i]],]
  inTraining_raw[] = lapply(inTraining_raw, function(x) if(is.factor(x)) factor(x) else x)
  inTesting_raw = dt[dt$Index %in% TestList[[i]],]
  inTesting_raw[] = lapply(inTesting_raw, function(x) if(is.factor(x)) factor(x) else x)
  inTraining = process.data(inTraining_raw, method=method,idCol,classCol,xCol) #process train
  inTraining[is.na(inTraining)] = 0 #manage NaN
  inTraining[,classCol] = factor(make.names(inTraining[,classCol])) #make valid names
  inTesting = process.data(inTesting_raw, method=method,idCol,classCol,xCol) #process test
  inTesting[is.na(inTesting)] = 0 #manage NaN
  inTesting[,classCol] = factor(make.names(inTesting[,classCol])) #make valid names
  mbobj_train = set_input_obj(inTraining,idCol,classCol,xCol) #Training
  mbobj_test = set_input_obj(inTesting,idCol,classCol,xCol) #Testing
  #calculate
  inModeling = data.frame(Class=mbobj_train$Y,mbobj_train$X, check.names = FALSE)
  predmodel = train(Class ~ ., data = inModeling, method = "pls", trControl = fitControl, preProc = NULL, importance = TRUE) #rf
  train.pred = predict(predmodel, mbobj_train$X) #Training prediction
  train.acc = mean(train.pred == inModeling$Class) #Training accuracy
  test.pred = predict(predmodel, mbobj_test$X) #Testing prediction
  test.prob = predict(predmodel, mbobj_test$X, type = "prob") #Testing prediction prob
  out.cf = confusionMatrix(test.pred, mbobj_test$Y) #cf - the first level will be used as the positive
  multi.auc = multiclass.roc(mbobj_test$Y, as.matrix(test.prob)) #auc
  if(nlevels(mbobj_train$Y) > 2){#multi-class
    class.stats = as.data.frame(out.cf$byClass)
    class.stats = class.stats %>% mutate(Class = gsub("Class: ","",rownames(out.cf$byClass))) %>%
      select(Class, Sensitivity, Specificity, F1, 'Balanced Accuracy')
    class.stats = class.stats %>% rename(Balanced.Accuracy = 'Balanced Accuracy')
    class.avg = cbind(Class="Average",as.data.frame(t(apply(class.stats[,-1], 2, function(x) {mean(x,na.rm=T)})))) #avg across all classes
    #one-vs-rest AUCs
    auc_class = sapply(levels(mbobj_test$Y), function(cls) {
      #convert to binary (1 = this class, 0 = not this)
      true_y = ifelse(mbobj_test$Y == cls, 1, 0)
      roc_obj = roc(response = true_y, predictor = test.prob[[cls]])
      auc(roc_obj)
    })
    auc_df = data.frame(Class=names(auc_class),AUC=unlist(auc_class))
    #imp_class = predmodel$finalModel$importance[, 1:nlevels(mbobj_train$Y)]
    #imp_list[[names(TrainList)[i]]] = data.frame(Class=colnames(imp_class),t(apply(imp_class, 2, function(x) 100 * x / max(x)))) #important features
    stat_list[[names(TrainList)[i]]] = merge(class.stats,auc_df,by="Class") #performance per class
  }else{#binary class
    class.avg = as.data.frame(t(out.cf$byClass))
    class.avg = class.avg %>% mutate(Class = out.cf$positive) %>%
      select(Class, Sensitivity, Specificity, F1, 'Balanced Accuracy')
    class.avg = class.avg %>% rename(Balanced.Accuracy = 'Balanced Accuracy')
    #imp_list[[names(TrainList)[i]]] = data.frame()
    stat_list[[names(TrainList)[i]]] = data.frame()
  }
  ############## ----------------------------------------------------- ##############
  wtmat = compute_weighted_metrics(out.cf)
  #overall outputs
  out_perform = rbind(out_perform,c(class.avg,
                                    data.frame(Train.Accuracy=train.acc, Test.Accuracy=out.cf$overall["Accuracy"], Overall.AUC=multi.auc$auc,
                                               Weighted.Sensitivity=wtmat$weighted_sensitivity, Weighted.F1=wtmat$weighted_f1)))
  ############## ----------------------------------------------------- ##############
  
  #imp_raw = predmodel$finalModel$importance[, "MeanDecreaseAccuracy"] #MeanDecreaseAccuracy; MeanDecreaseGini
  #out_scaled = data.frame(t(100 * imp_raw / max(imp_raw)))
  #out_imp = rbind(out_imp,out_scaled) #scaled importance scores
  #out_model[[names(TrainList)[i]]] = predmodel
}
#row.names(out_perform) = row.names(out_imp) = names(TrainList)
write.csv(out_perform,file = paste0(outpath,"raw_pls_summary.csv"))
## 2. Export outputs
#export_model_outputs(binary=binary,rf=TRUE,out_pf=out_perform,out_imt=out_imp,stat_ls=stat_list,imp_ls=imp_list,
                    # filename=filenamepath,width=9.5,height=(8+ncol(out_imp)*0.03))
cat("#### DONE!! ####\n")