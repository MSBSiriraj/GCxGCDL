# =============================
# Batch runner for Pls
# =============================

rm(list = ls())
gc()

library(metabox2)
library(caret)
library(ropls)
library(pROC)
library(dplyr)
library(optparse)


# Folder containing MMFramework.R and pls.R

wkpath <- "D:/OneDrive/Projects/FF66/LCMS-Net/MMFramework/MMFramework"

# PLS script to run
pls_script <- file.path(wkpath, "Pls_run_edit.R")

# Define datasets
datasets <- list(
  list(
    name    = "Elution_PT_NorESKD_Detect10_bin05_cutoff30",
    proc    = "raw",
    input   = "D:/OneDrive/Projects/FF66/LCMS-Net/Peaktables/xcomp_rt/Min_detect_10/All_HD_PD/All_HD_PD_bin05/RT1_direct/PT_All_HD_PD_RT1only_bin05_detect10a.csv",
    outpath = "D:/OneDrive/Projects/FF66/LCMS-Net/Peaktables/xcomp_rt/Min_detect_10/All_HD_PD/All_HD_PD_bin05/RT1_direct/raw_pls",
    train   = "D:/OneDrive/Projects/FF66/LCMS-Net/856_1024/Default_raw/xcomp_RT/All_HD_PD/TrainList_biclass.txt",
    test    = "D:/OneDrive/Projects/FF66/LCMS-Net/856_1024/Default_raw/xcomp_RT/All_HD_PD/TestList_biclass.txt"
  )
)

# Run loop
for (ds in datasets) {
  
  cat("\n=============================\n")
  cat("Running:", ds$name, "\n")
  cat("=============================\n")
  
  proc    <- ds$proc
  input   <- ds$input
  outpath <- ds$outpath
  train   <- ds$train
  test    <- ds$test
  
  setwd(wkpath)
  
  tryCatch({
    source(pls_script, local = FALSE)
    cat("DONE:", ds$name, "\n")
  }, error = function(e) {
    cat("ERROR in:", ds$name, "\n")
    cat(e$message, "\n")
  })
}