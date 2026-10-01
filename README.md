# GC×GC DL
This study develops and evaluates a workflow for applying the end-to-end deep learning model LCMS-Net to GC×GC-TOFMS metabolomics data using condensed retention-time representations. LCMS-Net performance was benchmarked against random forest (RF) and partial least squares discriminant analysis (PLS-DA) using tile-based and conventional peak table-based workflows, with additional evaluation of input optimization, data preprocessing, and instrumental batch effects. This repository provides the source code for implementing LCMS-Net and peak table-based classification workflows.

![workflow](https://github.com/MSBSiriraj/GCxGCDL/blob/main/GCGC_DL_workflow.png)

How to use
==========
### LCMS-Net classification

To generate input data for LCMS-Net, RT1 and RT2 are combined into a single retention-time axis. Two RT composite approaches are available: (1) SumRT, which combines RT1 and RT2 by summation, and (2) CompRT, which generates a composite RT representation. LCMS-Net is then trained and evaluated using the default matrix dimensions and aggregation method.

Generate .npy input files 
   * CompRT: bin_data_CompRT.py 
   * SumRT: bin_data_SumRT.py
2. Generate training and test sample lists
   * create_train_test.R
3. Train LCMS-Net across resamples
   * training_batch_resamples.py
4. Evaluate the trained models
   * evaluation_batch_resamples.py

### Peak table-based classification
ASCII peak tables exported from the instrument software are used as input. The peak tables are first converted into feature matrices using either the CompRT or SumRT representation, followed by RF or PLS-DA classification.
Prerequisite: RF and PLS-DA modeling is implemented using the caret R package through the MMFramework platform (MMFramework_balanced.R). 
MMFramework and its required dependencies should therefore be installed before running the RF and PLS-DA scripts.

Generate input matrices
   * CompRT: combine_peaktable_CompRT.py 
   * SumRT: combine_peaktable_SumRT.py
2. RF classification
   * Batch execution: Rf_run_batch.R 
   * Model training and evaluation: Rf_run.R
3. PLS-DA classification 
   * Batch execution: Pls_run_batch.R 
   * Model training and evaluation: Pls_run.R


References
=========
- Menacher LM, Ward LJ, Heintz F, Green H, Sysoev O. LCMS-Net: Deep Learning for Raw High Resolution Mass Spectrometry Data Applied to Forensic Cause-of-Death Screening. Analytical Chemistry. 2026 Feb 27;98(9):6589.: [5c05404](https://doi.org/10.1021/acs.analchem.5c05404)
- Wanichthanarak K, Duangkumpha K, Kleebkomut N, Saiviroonporn P, Tongdee T, Sirivatanauksorn Y, Kitiyakara C, Khoomrung S: When Clinical and Metabolomics Data Work Together: A Comparative Framework for Multimodal Disease Classification across Machine and Deep Learning. ACS Meas. Sci. Au 2026: [6c00108](https://doi.org/10.1021/acsmeasuresciau.6c00108)
