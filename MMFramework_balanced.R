library(ComplexHeatmap)
library(circlize)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tidyverse)
library(forcats)
library(RColorBrewer)
library(ggtext)
library(qpdf)
library(pROC)
library(tibble)
library(tidytext)

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
##### Data processing train/test data before modeling #####
process.data <- function(indata, method='auto',idCol,classCol,xCol,featCol=NULL)
{
  ##Data processing
  if (method == "raw"){#Raw
    outdata = indata
  }
  if (method == "log2"){#Log2
    indata_scaled = transform_input_data(set_input_obj(indata,idCol,classCol,xCol),method = 'log2')
    outdata = data.frame(indata_scaled$inputdata[,1:(indata_scaled$xCol-1)],indata_scaled$X)
    outdata[] <- lapply(outdata, function(x) if(is.factor(x)) factor(x) else x) #log2
  }
  if (method == "auto"){#Auto
    indata_scaled = scale_input_data(set_input_obj(indata,idCol,classCol,xCol),method = 'auto')
    outdata = data.frame(indata_scaled$inputdata[,1:(indata_scaled$xCol-1)],indata_scaled$X)
    outdata[] <- lapply(outdata, function(x) if(is.factor(x)) factor(x) else x) #auto
  }
  if (method == "minmax"){#Min-Max
    indata_scaled = set_input_obj(indata,idCol,classCol,xCol)
    scaler = preProcess(indata_scaled$X, method = "range")
    scaled_df = predict(scaler, indata_scaled$X)
    outdata = data.frame(indata_scaled$inputdata[,1:(indata_scaled$xCol-1)], scaled_df)
    outdata[] <- lapply(outdata, function(x) if(is.factor(x)) factor(x) else x) #min-max
  }
  if (method == "mb"){#MB
    indata_scaled = set_input_obj(indata,idCol,classCol,xCol = featCol) #set featCol=16
    meta_dt = data.frame(indata_scaled$inputdata[,xCol:(indata_scaled$xCol-1)]) #clinic data
    feat_dt = data.frame(indata_scaled$X) #feature data
    areaX = ktab.list.df(list(meta_dat = meta_dt, feat_dat = feat_dt)) #input data
    disjonctif = (disjunctive(data.frame(indata_scaled$Y))) #input class
    dudiY   = dudi.pca(disjonctif, center = FALSE, scale = FALSE, scannf = FALSE) #format input class
    rownames(dudiY$tab) = rownames(areaX[[1]])
    area_norm = mbplsda(dudiY, areaX, scale = TRUE, option = "uniform", scannf = FALSE, nf = ncol(indata_scaled$X))
    outdata = data.frame(indata_scaled$inputdata[,1:(xCol-1)], area_norm$tabX)
    outdata[] <- lapply(outdata, function(x) if(is.factor(x)) factor(x) else x) #mb
  }
  if (method == "mb2"){#MB2
    indata_scaled = set_input_obj(indata,idCol,classCol,xCol = featCol) #set featCol=16
    meta_dt = data.frame(indata_scaled$inputdata[,xCol:(indata_scaled$xCol-1)]) #clinic data
    feat_dt = data.frame(indata_scaled$X) #feature data
    areaX = ktab.list.df(list(meta_dat = meta_dt, feat_dat = feat_dt)) #input data
    disjonctif = (disjunctive(data.frame(indata_scaled$Y))) #input class
    dudiY   = dudi.pca(disjonctif, center = FALSE, scale = FALSE, scannf = FALSE) #format input class
    rownames(dudiY$tab) = rownames(areaX[[1]])
    area_norm = mbplsda(dudiY, areaX, scale = TRUE, option = "none", scannf = FALSE, nf = ncol(indata_scaled$X))
    outdata = data.frame(indata_scaled$inputdata[,1:(xCol-1)], area_norm$tabX)
    outdata[] <- lapply(outdata, function(x) if(is.factor(x)) factor(x) else x) #mb
  }

  return(outdata)
}

# -------------------------------------------------------
# Main framework functions
# -------------------------------------------------------
##### Compute weighted metrics for class imbalance #####
############## ----------------------------------------------------- ##############
compute_weighted_metrics <- function(cfm){
  tab = cfm$table #confusion matrix table
  classes = rownames(tab)
  recall_per_class = numeric(length(classes))
  f1_per_class = numeric(length(classes))
  support = rowSums(tab) #class frequencies

  for(i in seq_along(classes)) {
    TP <- tab[i, i]
    FP <- sum(tab[, i]) - TP
    FN <- sum(tab[i, ]) - TP

    #Recall or Sensitivity
    recall <- TP/(TP + FN)
    #Precision
    precision <- TP/(TP + FP)
    #F1
    f1 <- 2*((precision * recall)/(precision + recall))

    recall_per_class[i] = recall
    f1_per_class[i] = f1
  }

  #weighted sensitivity
  weighted_sensitivity = weighted.mean(recall_per_class,w = support,na.rm = TRUE)

  #weighted F1
  weighted_f1 = weighted.mean(f1_per_class,w = support,na.rm = TRUE)

  return(list(weighted_sensitivity = weighted_sensitivity, weighted_f1 = weighted_f1))
}
############## ----------------------------------------------------- ##############

##### Compute mean and SD of important score per feature #####
compute_vip_summary <- function(vip_df) {
  #Mean and SD VIP per feature
  vip_mean = colMeans(vip_df, na.rm = TRUE)
  vip_sd   = apply(vip_df, 2, sd, na.rm = TRUE)
  #Rank features per row (highest VIP = rank 1)
  vip_ranks = t(apply(vip_df, 1, function(x) rank(-x, ties.method = "average")))
  #Median and IQR rank per feature
  rank_median = apply(vip_ranks, 2, median, na.rm = TRUE)
  rank_iqr   = apply(vip_ranks, 2, IQR, na.rm = TRUE)

  vip_summary = data.frame(
    Feature = colnames(vip_df),VIP_mean = vip_mean,
    VIP_sd = vip_sd,Rank_median = rank_median,Rank_IQR = rank_iqr
  )
  return(vip_summary)
}

##### Combine list of performances per class #####
combine_stat_list <- function(stat_ls) {
  stat_summary = bind_rows(lapply(names(stat_ls), function(fold) {
      stat_ls[[fold]] %>% mutate(Fold = fold)
    }), .id = NULL) %>%
    pivot_longer(cols = -c(Fold, Class), names_to = "Metric",values_to = "Value")
  stat_summary$Class = as.factor(stat_summary$Class)
  return(stat_summary)
}

##### Combine list of importance per class #####
combine_imp_list <- function(imp_ls){
  df_imp = bind_rows(imp_ls, .id = "FoldRep")
  imp_summary = df_imp %>% pivot_longer(cols = -c(FoldRep, Class), names_to = "Feature", values_to = "Importance")
  imp_summary$Class = as.factor(imp_summary$Class)
  return(imp_summary)
}

##### Export outputs and plots #####
export_model_outputs <- function(binary=TRUE,rf=FALSE,out_pf,out_imt,stat_ls,imp_ls,filename="model",width=8,height=8){
  pdf(file = paste0(filename,"temp1.pdf"), width=7, height=7) #Overall model performance
  # Overall
  print(plot_model_eval(out_pf, title="Accuracy Curves Across Training Iterations"))
  print(plot_barplot_model(out_pf[,c(-1)]))
  dev.off()
  pdf(file = paste0(filename,"temp2.pdf"), width=width, height=height) #Feature importance
  print(plot_barplot_importance(out_imt))
  plot_heatmap_importance(out_imt)
  imp_sum = compute_vip_summary(out_imt)
  print(plot_rank_stability(imp_sum))
  dev.off()

  # Per class
  if(!binary){#multiclass
    if(length(imp_ls[[1]])>0) {#if have imp list
      model_summary_perclass = combine_stat_list(stat_ls)
      feature_summary_perclass = combine_imp_list(imp_ls)
      pdf(file = paste0(filename,"temp3.pdf"), width=width, height=8) #Feature importance
      print(plot_barplot_model_perclass(model_summary_perclass))
      dev.off()
      pdf(file = paste0(filename,"temp4.pdf"), width=width, height=height) #Feature importance
      if(rf){
        print(plot_stackplot_feature_perclass_rf(feature_summary_perclass))}
      else{
        print(plot_stackplot_feature_perclass(feature_summary_perclass))
      }
      rank_list = list() #Collect median ranks
      for(cls in levels(feature_summary_perclass$Class)) {
        df_class <- feature_summary_perclass %>% filter(Class == cls) %>%
          select(FoldRep, Feature, Importance) %>%
          pivot_wider(names_from = Feature, values_from = Importance) %>% as.data.frame()
        rownames(df_class) = df_class$FoldRep
        df_class = df_class[, -1]
        plot_heatmap_importance(df_class, title = paste("Feature Importance Map -", cls))
        print(plot_barplot_importance(df_class, title = paste("Average Feature Importance -", cls)))
        vip_summary = compute_vip_summary(df_class)
        rank_list[[cls]] <- vip_summary %>% select(Feature, Rank_median) %>% rename(!!cls := Rank_median)
        print(plot_rank_stability(vip_summary, title = paste("Rank Stability Plot -", cls)))
      }
      dev.off()
      rank_feature_summary_perclass <- Reduce(function(x, y) full_join(x, y, by = "Feature"), rank_list)
      write.csv(model_summary_perclass,file = paste0(filename,"model_perclass.csv"), row.names = FALSE) #model performance
      #write.csv(feature_summary_perclass,file = paste0(filename,"feature_perclass.csv"), row.names = FALSE) #feature importance
      feature_summary_perclass_wide <- feature_summary_perclass %>% rename(Fold.Rep = FoldRep) %>%
        pivot_wider(id_cols = c(Fold.Rep, Feature),names_from = Class,values_from = Importance)
      write.csv(feature_summary_perclass_wide,file = paste0(filename,"feature_perclass.csv"), row.names = FALSE) #feature importance
      write.csv(rank_feature_summary_perclass,file = paste0(filename,"feature_rank_perclass.csv"), row.names = FALSE) #ranked feature
      pdf_files = paste0(filename,c("temp1.pdf", "temp2.pdf", "temp3.pdf","temp4.pdf"))
    }else{#no imp list
      write.csv(data.frame(),file = paste0(filename,"model_perclass.csv"), row.names = FALSE) #empty model performance
      write.csv(data.frame(),file = paste0(filename,"feature_perclass.csv"), row.names = FALSE) #empty feature importance
      write.csv(data.frame(),file = paste0(filename,"feature_rank_perclass.csv")) #empty ranked feature
      pdf_files = paste0(filename,c("temp1.pdf", "temp2.pdf"))
    }
      pdf_combine(input = pdf_files, output = paste0(filename,"summary.pdf"))
  }else{#binary
    pdf_files <- paste0(filename,c("temp1.pdf", "temp2.pdf"))
    pdf_combine(input = pdf_files, output = paste0(filename,"summary.pdf"))
  }
  # Write to files
  write.csv(out_pf,file = paste0(filename,"model.csv")) #model performance
  #write.csv(out_imt,file = paste0(filename,"feature.csv")) #feature importance
  out_imt_long <- out_imt %>%
    rownames_to_column(var = "Fold.Rep") %>%
    pivot_longer(cols = -Fold.Rep,names_to = "Feature",values_to = "Feature_Importance")
  write.csv(out_imt_long,file = paste0(filename,"feature.csv"), row.names = FALSE) #feature importance
  write.csv(subset(imp_sum, select = -c(VIP_mean, VIP_sd)),file = paste0(filename,"feature_rank.csv"), row.names = FALSE) #ranked feature

  # Delete temporary pdf files
  temp_pdfs <- list.files(
    path = dirname(filename),
    pattern = paste0(basename(filename), "temp.*\\.pdf$"),
    full.names = TRUE
  )
  if (length(temp_pdfs) > 0) {
    unlink(temp_pdfs, force = TRUE)
  }
}

##### Plot model evaluation plot #####
plot_model_eval <- function(model_df,title=""){
  avg_micro = mean(model_df$Balanced.Accuracy, na.rm = TRUE)
  avg_test = mean(model_df$Test.Accuracy, na.rm = TRUE)
  avg_train = mean(model_df$Train.Accuracy, na.rm = TRUE)

  # Define caption with colors
  caption_text <- paste0(
    "<span style='color:#D11E00;'>Test Accuracy (mean): ", round(avg_test, 3), "</span><br>",
    "<span style='color:#00429D;'>Training Accuracy (mean): ", round(avg_train, 3), "</span><br>",
    "<span style='color:#383838;'>Balanced Accuracy (mean): ", round(avg_micro, 3), "</span><br>"
  )

  # Plot
  ggplot(data = model_df, aes(x=1:nrow(model_df), y=Train.Accuracy)) +
    geom_line(color = "#00429D", linewidth = 0.9, alpha=0.6) +  #train
    geom_hline(yintercept = avg_train, color = "#00429D", linetype = 5, linewidth = 1.5, alpha=0.8) + #train acc
    geom_hline(yintercept = avg_test, color = "#D11E00", linetype = 5, linewidth = 1.5, alpha=0.8) + #test acc
    theme_bw() +
    theme(legend.position = "none", plot.title = element_text(size = 18), title = element_text(size = 18),
      axis.title = element_text(size = 18), axis.text  = element_text(size = 16),
      plot.caption = element_markdown(size = 18, hjust = 0)) +
    labs(title = title,x = "Round",y = "Accuracy", caption = caption_text)
}

##### Plot model summary barplot #####
plot_barplot_model <- function(results_df, title = "Overall Model Performance (Mean ± SD)") {
  df_long = results_df %>% mutate(Fold = rownames(results_df)) %>%
    pivot_longer(cols = -Fold, names_to = "Metric", values_to = "Value")

  # Compute mean and SD per feature
  summary_df = df_long %>% group_by(Metric) %>%
    summarise(
      mean_importance = mean(Value, na.rm = TRUE),
      sd_importance = sd(Value, na.rm = TRUE)
    ) %>% arrange(desc(mean_importance))
  df_long$Metric = factor(df_long$Metric, levels = summary_df$Metric)

  # Barplot with mean ± SD
  max_x = max(summary_df$mean_importance + summary_df$sd_importance)
  ggplot(summary_df, aes(x=mean_importance, y=Metric)) + geom_col(color="#858585",fill = "#32A5D9", width = 0.75) +
    geom_errorbar(aes(xmin = mean_importance - sd_importance,
                      xmax = mean_importance + sd_importance), width = 0.1) +
    geom_text(aes(label = sprintf("%.3f±%.2f", mean_importance, sd_importance),
                  x = mean_importance + sd_importance + 0.005 * max_x),
              hjust = 0,size = 4,lineheight = 0.9) +
    scale_x_continuous(
      breaks = seq(0, 1.0, by = 0.2)
    ) +
    coord_cartesian(xlim = c(0, max_x * 1.15))+
    theme_bw() +
    labs(x = "", y = "", title = title) +
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 16),
          axis.text = element_text(size = 14),text =  element_text(size = 14))
}

##### Plot importance summary heatmap #####
plot_heatmap_importance <- function(results_df, title = "Overall Feature Importance Map") {
  # Heatmap
  heatmap_data_t = t(as.matrix(results_df))
  if(all(heatmap_data_t==0)){
    ht = Heatmap(heatmap_data_t, col = c("0" = "grey90"))
  }else{
    ht=Heatmap(heatmap_data_t,name = "Feature Importance",column_title = title,
          show_row_dend = FALSE,
          show_row_names = TRUE, show_column_names = TRUE,
          row_names_gp = gpar(fontsize = 14), column_names_gp = gpar(fontsize = 12), column_names_rot = 70,
          cluster_rows = T, cluster_columns = FALSE,
          rect_gp = gpar(col = "grey", lwd = 1),col=colorRampPalette(brewer.pal(9, "PuRd"))(256),
          column_title_gp = gpar(fontsize = 18),
          heatmap_legend_param = list(title_gp = gpar(fontsize = 14, fontface = "bold"),labels_gp = gpar(fontsize = 14),legend_direction="horizontal"))
    }
  draw(ht, heatmap_legend_side = "bottom")
}

##### Plot importance summary barplot #####
plot_barplot_importance <- function(results_df,cutoff=NULL, title = "Average Feature Importance") {
  df_long = results_df %>% mutate(Fold = rownames(results_df)) %>%
    pivot_longer(cols = -Fold, names_to = "Feature", values_to = "Importance")

  # Compute mean and SD per feature
  summary_df = df_long %>% group_by(Feature) %>%
    summarise(
      mean_importance = mean(Importance, na.rm = TRUE),
      sd_importance = sd(Importance, na.rm = TRUE)
    ) %>% arrange(desc(mean_importance))
  df_long$Feature = factor(df_long$Feature, levels = summary_df$Feature)

  # Barplot with mean ± SD
  max_x = max(summary_df$mean_importance + summary_df$sd_importance)
  median_rank = median(summary_df$mean_importance)
  rank_cutoff = ifelse(is.null(cutoff),median_rank,cutoff)
  # Define caption with colors
  caption_text <- paste0(
    "<span style='color:#D35400;'>Median Importance = ", round(median_rank, 3), "</span>"
  )
  ggplot(summary_df, aes(x = mean_importance, y = fct_rev(factor(Feature, levels = Feature)))) +
    geom_col(color="#858585",fill = "#7F90E3", width = 0.75) +
    geom_errorbar(aes(xmin = mean_importance - sd_importance,
                      xmax = mean_importance + sd_importance), width = 0.1) +
    geom_vline(xintercept = rank_cutoff, linetype = "dashed", color = "#D35400", linewidth = 1, alpha = 0.8) +
    geom_text(data = summary_df,
      aes(label = sprintf("%.3f±%.2f", mean_importance, sd_importance),
        x = mean_importance + sd_importance + 0.005 * max_x),
      hjust = 0,size = 4.5,lineheight = 0.9
    ) + scale_x_continuous(limits = c(0, max_x * 1.15)) +
    theme_bw() +
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 16),
                      axis.text = element_text(size = 16),text =  element_text(size = 12),
                      plot.caption = element_markdown(size = 14, hjust = 0)) +
    labs(x = "Feature Importance", y = NULL, title = title, caption = caption_text)
}

##### Plot rank stability with mean±std error bars #####
plot_rank_stability <- function(rank_df,cutoff = NULL, title = "Rank Stability Plot") {
  if (is.null(cutoff)) {cutoff = median(rank_df$Rank_median, na.rm = TRUE)}
  rank_df = rank_df %>% mutate(highlight = Rank_median <= cutoff)
  rank_df = rank_df %>% arrange(Rank_median) %>% mutate(Feature = factor(Feature, levels = rev(Feature)))  #reverse for ggplot y-axis

  # Define caption with colors
  caption_text = paste0(
    "<span style='color:#D35400;'> Median Rank = ",cutoff,"</span>"
  )
  # Plot
  ggplot(rank_df, aes(x = Rank_median, y = Feature)) +
    geom_errorbar(aes(xmin = Rank_median - (Rank_IQR/2), xmax = Rank_median + (Rank_IQR/2)),
                  orientation = "y",width = 0.2, color = "black") +
    geom_point(aes(fill = Rank_median > cutoff),size = 5,alpha = 0.9,shape = 21,
               color = "black", stroke = 0.4) +
    geom_text(aes(label = round(Rank_median, 2)),nudge_y = -0.4,size = 4,show.legend = FALSE) +
    geom_vline(xintercept = cutoff, linetype = "dashed", color = "#D35400", linewidth = 1, alpha = 0.8) +
    scale_size_continuous(name = "Rank",trans = "reverse") +
    scale_fill_manual(values = c("#D65FA1", "#007BFF"), labels = c(" > Median Rank","< Median Rank")) +
    labs(x = "Rank", y = "", title = title, subtitle = "Point: Median Rank; Bar: IQR",
         fill = NULL, caption = caption_text) + theme_bw() +
    theme(plot.title = element_text(size = 18), plot.subtitle = element_text(size = 12),
          axis.title = element_text(size = 18), axis.text = element_text(size = 16),
          text = element_text(size = 16),
      plot.caption = element_markdown(size = 14, hjust = 0) #ggtext for colored caption
    )
}

##### Plot model summary barplot per class #####
plot_barplot_model_perclass <- function(results_df, title = "Overall Class-wise Performance Metrics") {
  # Compute mean and SD per class
  summary_df <- results_df %>%  group_by(Class, Metric) %>%
    summarise(mean_importance = mean(Value, na.rm = TRUE),
              sd_importance   = sd(Value, na.rm = TRUE), .groups = "drop"
    )
  # Barplot with mean ± SD
  ggplot(summary_df, aes(x = Metric, y = mean_importance, fill = Class)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7, color = "black", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.2f", mean_importance)), position = position_dodge(width = 0.75),
      vjust = -0.3,size = 4.5) +
    scale_fill_brewer(palette = "Dark2") + scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "", y = "Performance", title = title) + theme_bw() +
    theme(plot.title = element_text(size = 18), legend.position = "top",
          axis.title = element_text(size = 18), axis.text.x = element_text(angle = 25, hjust = 1),
          axis.text = element_text(size = 16),text =  element_text(size = 16))
}

##### Plot importance summary stackplot per class #####
plot_stackplot_feature_perclass <- function(results_df, title = "Class-wise Feature Importance Distribution") {
  # Compute mean and SD per class
  summary_df <- results_df %>% group_by(Feature, Class) %>%
    summarise(mean_imp = mean(Importance, na.rm = TRUE), .groups = "drop") %>%
    group_by(Feature) %>% mutate(pct = mean_imp / sum(mean_imp) * 100)

  ggplot(summary_df, aes(x = pct, y = Feature, fill = Class)) +
    geom_col(color = "black", alpha = 0.8) +
    geom_text(data = subset(summary_df, pct > 0), aes(label = sprintf("%.1f", pct)),
              position = position_stack(vjust = 0.5), color = "black", size = 5) +
    scale_fill_brewer(palette = "Dark2") +
    labs(title = title,x = "Percentage of Importance (%)",y = "") +
    theme_bw() + theme(plot.title = element_text(size = 18), legend.position = "top",
                           axis.title = element_text(size = 18),
                           axis.text = element_text(size = 16),text =  element_text(size = 16))
}

##### Plot importance summary stackplot per class #####
plot_stackplot_feature_perclass_rf <- function(results_df, title = "Class-wise Feature Importance Distribution") {
  # Compute mean and SD per class
  summary_df <- results_df %>% group_by(Feature, Class) %>%
    summarise(mean_imp = mean(Importance, na.rm = TRUE), .groups = "drop") %>%
    group_by(Feature) %>% mutate(pct = mean_imp / sum(mean_imp) * 100)

  ggplot(summary_df, aes(x = mean_imp, y = Feature, fill = Class)) +
    geom_col(color = "black", alpha = 0.8) +
    scale_fill_brewer(palette = "Dark2") +
    labs(title = title,x = "Percentage of Importance (%)",y = "") +
    theme_bw() + theme(plot.title = element_text(size = 18), legend.position = "top",
                       axis.title = element_text(size = 18),
                       axis.text = element_text(size = 16),text =  element_text(size = 16))
}

##### Plot model evaluation plot #####
plot_evaluation_plot <- function(eval_df,model_df,title=""){
  #Setup data
  eval_tb = lapply(eval_df, function(df) {
    df %>% mutate(Round = row_number(), Eval = "Eval")
  })
  eval_ls = bind_rows(eval_tb, .id = "dataset") #Validation accuracy per-fold
  avg_eval = mean(eval_ls$Accuracy, na.rm = TRUE)
  avg_test = mean(model_df$acc, na.rm = TRUE)
  avg_train = mean(model_df$train_acc, na.rm = TRUE)

  # Define caption with colors
  caption_text <- paste0(
    "<span style='color:#D11E00;'>Test Accuracy (mean): ", round(avg_test, 3), "</span><br>",
    "<span style='color:#00429D;'>Training Accuracy (mean): ", round(avg_train, 3), "</span><br>",
    "<span style='color:#383838;'>Validation Accuracy (mean): ", round(avg_eval, 3), "</span><br>"
  )

  # Plot
  ggplot(data = model_df, aes(1:20, train_acc)) +
    geom_line(color = "#00429D", linewidth = 4) +  #train
    geom_line(data = eval_ls, aes(x=Round, y=Accuracy, color=dataset),
              linetype=2, linewidth=1, alpha=0.5) +  #per resample
    scale_color_manual(name="Legend", values=rep("#ADADAD", length(unique(eval_ls$dataset)))) +
    geom_line(data = model_df, aes(1:20, eval_acc), color = "#383838", linewidth = 1.5) +  #validation
    geom_hline(yintercept = avg_train, linetype = 5, color = "#00429D", linewidth = 1, alpha=0.6) + #train acc
    geom_hline(yintercept = avg_eval, linetype = 5, color = "#383838", linewidth = 1, alpha=0.6) + #validation acc
    geom_hline(yintercept = avg_test, linetype = 5, color = "#D11E00", linewidth = 1, alpha=0.6) + #test acc
    theme_bw() +
    theme(
      legend.position = "none",
      axis.title = element_text(size = 20),
      axis.text  = element_text(size = 18),
      plot.caption = element_markdown(size = 18, hjust = 0), #ggtext for colored caption
    ) +
    labs(title = title,x = "Round",y = "Accuracy", caption = caption_text)
}

# -------------------------------------------------------
# Functions to combine model outputs
# -------------------------------------------------------
##### Compute overall model performance #####
#' folder_path = "LN/clinical"
############## ----------------------------------------------------- ##############
compute_model_performance <- function(folder_path){
  # Set folder path
  folder_path = folder_path
  # List all model CSV files
  files = list.files(folder_path, pattern = "_model\\.csv$", full.names = TRUE)
  # Compute mean, SD
  get_model_stats <- function(file) {
    stat_df = read.csv(file)
    # Return first existing column from a candidate list
    pick_col <- function(df, cols) {
      cols[cols %in% names(df)][1]
    }
    acc_col = pick_col(stat_df, c("Accuracy", "Test.Accuracy"))
    f1_col  = pick_col(stat_df, c("F1.score", "F1"))
    auc_col = pick_col(stat_df, c("AUC", "Overall.AUC"))
    stat_df %>%
      summarise(
        Model = gsub("_model$", "", tools::file_path_sans_ext(basename(file))),
        n = n(), t_value = qt(0.975, df = n - 1),
        Accuracy_mean = mean(.data[[acc_col]], na.rm = TRUE),
        Accuracy_sd = sd(.data[[acc_col]], na.rm = TRUE),
        Accuracy_lowerCI = Accuracy_mean - t_value * Accuracy_sd / sqrt(n),
        Accuracy_upperCI = Accuracy_mean + t_value * Accuracy_sd / sqrt(n),
        Sensitivity_mean = mean(Sensitivity, na.rm = TRUE),
        Sensitivity_sd = sd(Sensitivity, na.rm = TRUE),
        Sensitivity_lowerCI = Sensitivity_mean - t_value * Sensitivity_sd / sqrt(n),
        Sensitivity_upperCI = Sensitivity_mean + t_value * Sensitivity_sd / sqrt(n),
        Specificity_mean = mean(Specificity, na.rm = TRUE),
        Specificity_sd = sd(Specificity, na.rm = TRUE),
        Specificity_lowerCI = Specificity_mean - t_value * Specificity_sd / sqrt(n),
        Specificity_upperCI = Specificity_mean + t_value * Specificity_sd / sqrt(n),
        F1_mean = mean(.data[[f1_col]], na.rm = TRUE),
        F1_sd = sd(.data[[f1_col]], na.rm = TRUE),
        F1_lowerCI = F1_mean - t_value * F1_sd / sqrt(n),
        F1_upperCI = F1_mean + t_value * F1_sd / sqrt(n),
        AUC_mean = mean(.data[[auc_col]], na.rm = TRUE),
        AUC_sd = sd(.data[[auc_col]], na.rm = TRUE),
        AUC_lowerCI = AUC_mean - t_value * AUC_sd / sqrt(n),
        AUC_upperCI = AUC_mean + t_value * AUC_sd / sqrt(n),
        Balanced.Accuracy_mean = mean(Balanced.Accuracy, na.rm = TRUE),
        Balanced.Accuracy_sd = sd(Balanced.Accuracy, na.rm = TRUE),
        Balanced.Accuracy_lowerCI = Balanced.Accuracy_mean - t_value * Balanced.Accuracy_sd / sqrt(n),
        Balanced.Accuracy_upperCI = Balanced.Accuracy_mean + t_value * Balanced.Accuracy_sd / sqrt(n),
        Weighted.Sensitivity_mean = mean(Weighted.Sensitivity, na.rm = TRUE),
        Weighted.Sensitivity_sd = sd(Weighted.Sensitivity, na.rm = TRUE),
        Weighted.Sensitivity_lowerCI = Weighted.Sensitivity_mean - t_value * Weighted.Sensitivity_sd / sqrt(n),
        Weighted.Sensitivity_upperCI = Weighted.Sensitivity_mean + t_value * Weighted.Sensitivity_sd / sqrt(n),
        Weighted.F1_mean = mean(Weighted.F1, na.rm = TRUE),
        Weighted.F1_sd = sd(Weighted.F1, na.rm = TRUE),
        Weighted.F1_lowerCI = Weighted.F1_mean - t_value * Weighted.F1_sd / sqrt(n),
        Weighted.F1_upperCI = Weighted.F1_mean + t_value * Weighted.F1_sd / sqrt(n)
      )
  }
  # Collect all files
  out_stat = bind_rows(lapply(files, get_model_stats))
  # Export the output
  write.csv(out_stat, paste0(folder_path,"/ModelPerformance.csv"), row.names = FALSE)
  cat("#### Exporting ModelPerformance.csv ... DONE!! ####\n")
}
############## ----------------------------------------------------- ##############

##### Plot comparative heatmap #####
#' input_list = c("Int"="LN/integrated","Cli"="LN/clinical","Met"="LN/metabolomics")
#' title = "Control vs LN"
plot_comparative_heatmap <- function(input_list, title="Comparative heatmap"){
  df_list = list()
  # Collect ModelPerformance.csv from each folder
  for (prefix in names(input_list)) {
    file_path = file.path(input_list[[prefix]], "ModelPerformance.csv")
    if (file.exists(file_path)) {
      tmp = read.csv(file_path, stringsAsFactors = TRUE)
      # Rename Model column
      tmp$Model = paste0(prefix, "_", tmp$Model)
      df_list[[prefix]] <- tmp
    }
  }
  # Combine all dataframes
  combined_df = do.call(rbind, df_list)
  #combined_df = combined_df[grepl("_log2_(gradcam|shap|rf)|_auto_pls", combined_df$Model, ignore.case = TRUE), ] #filtered models
  combined_df = combined_df[grepl("gradcam|shap|plsda|rf|elastic", combined_df$Model, ignore.case = TRUE), ] #filtered models
  combined_df$Type = ifelse(grepl("plsda|rf|elastic", combined_df$Model, ignore.case = TRUE),
    "ML", ifelse(grepl("gradcam|shap", combined_df$Model, ignore.case = TRUE),
      "DL","ND")) #set model types
  mat = (combined_df[, c("Type","Accuracy_mean", "Sensitivity_mean", "Specificity_mean", "F1_mean", "AUC_mean")])
  colnames(mat) = gsub("_mean$", "", colnames(mat))
  rownames(mat) = combined_df$Model
  # Set color
  inpcolors = c("#d95f02", "#5A9BD5", "#1b9e77", "#7570b3", "#e7298a", "#a6761d")
  col_fun = colorRamp2(
    breaks = c(0.1, 0.4, 0.55, 0.7, 1.0),
    colors = c("#FFF176","#D9F99D","#4ADE80","#22C55E","#8B5CF6")
  )
  row_col_fun <- function(rn) {
    prefix = sub("_.*", "", rn)
    color_map = setNames(inpcolors[seq_along(input_list)], names(input_list))
    cols = color_map[prefix]
    cols[is.na(cols)] = "black" #default
    cols
  }
  type_col = c(DL = "#B12A90",ML = "#9C9C9C",ND = "black")
  # Set matrix
  #mt = mat[grepl("_log2_(gradcam|shap|rf)|_auto_plsda", rownames(mat), ignore.case = TRUE), ] #filtered models
  mt = mat
  # Row annotation
  row_anno_df = data.frame(Type = mt$Type)
  rownames(row_anno_df) = rownames(mt)
  hmmt = as.matrix(mt[, -which(names(mt) == "Type")])
  row_anno = rowAnnotation(Model = row_anno_df$Type, col = list(Model = type_col),
    annotation_name_gp = gpar(fontsize = 16,fontface = "bold"),
    annotation_legend_param = list(title = "Model Type",
                                   title_gp = gpar(fontsize = 14,fontface = "bold"),
                                   labels_gp = gpar(fontsize = 14))
  )
  # Plot
  Heatmap(hmmt,name = "Performance",
          col = col_fun, cluster_rows = TRUE, cluster_columns = FALSE,
          show_row_names = TRUE, column_names_rot = 90,
          left_annotation = row_anno, column_title = title,
          row_names_gp = gpar(col = row_col_fun(rownames(hmmt)), fontsize = 16),
          column_names_gp = gpar(fontsize = 16),
          heatmap_legend_param = list(
            title_gp = gpar(fontsize = 14,fontface = "bold"),
            labels_gp = gpar(fontsize = 14)
          ),
          rect_gp = gpar(col = "black", lwd = 1.2),
          cell_fun = function(j, i, x, y, width, height, fill) {
            grid.text(sprintf("%.2f", hmmt[i, j]), x, y, gp = gpar(fontsize = 14))
          }
  )#7x7
}

##### Plot cross-model rank comparison #####
#' input_list = c("I"="LN/integrated","C"="LN/clinical","M"="LN/metabolomics")
#' featureFile = "auto_plsda_feature.csv"
#' title = "Control vs LN"
plot_cross_modal_rank <- function(input_list, featureFile, title="Cross-modal rank comparison"){
  ### Internal parameters ###
  # Compute mean per feature
  comp_imp_avg <- function(in_df, input_type){
    summary_df = in_df %>% group_by(Feature) %>%
      summarise(
        Importance = mean(Feature_Importance, na.rm = TRUE),
      ) %>% arrange(desc(Importance)) %>%
    mutate(
      rank = row_number(),
      input_type = input_type,
      tag = paste0("[", prefix, "]") #edit tag if needed
    )
    return(summary_df)
  }
  input_name = c("Integrated","Clinical","Metabolomics","DataX","DataY","DataZ")
  inpcolors = c("#d95f02", "#5A9BD5", "#1b9e77", "#7570b3", "#e7298a", "#a6761d")
  ### ###### ###### ###
  df_list = list()
  # Collect featureFile from each folder
  for (i in seq_along(input_list)) {
    prefix = names(input_list)[i]
    modality = input_name[i]
    file_path = file.path(input_list[[prefix]], featureFile)
    if (file.exists(file_path)) {
      tmp = read.csv(file_path, stringsAsFactors = TRUE)
      df_list[[prefix]] = comp_imp_avg(tmp, modality)
    }
  }
  df_plot = bind_rows(df_list, .id = "prefix")
  # Build y-axis label
  tag_lookup = df_plot %>% filter(prefix != names(input_list)[1]) %>%
    select(Feature, tag) %>% distinct() #for mapping
  df_plot = df_plot %>% select(-tag) %>% left_join(tag_lookup, by = "Feature")
  df_plot$Feature_label = paste0(df_plot$Feature, " ", df_plot$tag) #create label tag
  df_plot$input_type = factor(df_plot$input_type, levels = input_name[seq_along(input_list)])
  df_plot$Feature_label = reorder_within(df_plot$Feature_label,-df_plot$rank, df_plot$input_type) #order each facet
  # Plot
  ggplot(df_plot, aes(x=Importance, y=Feature_label, fill=input_type)) +
    geom_col(color = "black", width = 0.8
             #linewidth = 0.5
             ) +
    geom_text(aes(label = round(Importance,2)), hjust = -0.07,size = 4.5) +
    facet_wrap(~input_type, scales = "free_y") + scale_y_reordered() +
    scale_fill_manual(values = inpcolors) +
    labs(title = title,x = "Average Feature Importance",y = NULL,fill="Input type") +
    theme_test(base_size = 14) +
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 14, face = "bold"),
          axis.text = element_text(size = 15, color="black"),text = element_text(size = 16),
          strip.text = element_text(size = 16),legend.text = element_text(size = 14),legend.position = "top")+
    scale_x_continuous(limits = c(ifelse(min(df_plot$Importance) >=0,0,min(df_plot$Importance)), max(df_plot$Importance)*1.13)) #15.5x6
}

##### Plot modality ranking comparison #####
#' input_list = c("I"="LN/integrated","C"="LN/clinical","M"="LN/metabolomics")
#' featureFile = "auto_plsda_feature.csv"
#' title = "Control vs LN"
plot_modality_rank_comparison <- function(input_list, featureFile, title="Cross-modal rank comparison"){
  ### Internal parameters ###
  # Compute mean per feature
  comp_imp_avg <- function(in_df){
    summary_df = in_df %>% group_by(Feature) %>%
      summarise(
        Importance = mean(Feature_Importance, na.rm = TRUE),
      ) %>% arrange(desc(Importance))
    in_df$Feature = factor(in_df$Feature, levels = summary_df$Feature)
    return(summary_df)
  }
  rank_domain <- function(df, prefix, model_name) {
    if (length(unique(df$Importance)) == 1) {#can't find important feature
      summary_df = df %>% mutate(
        domain_rank = 1,
        rank_label  = paste0(prefix, domain_rank),
        input_type  = model_name
      ) %>% select(Feature, rank_label, input_type)
    } else {
      summary_df = df %>% mutate(
        domain_rank = rank(-Importance, ties.method = "first"),
        rank_label  = paste0(prefix, domain_rank),
        input_type  = model_name
      ) %>% select(Feature, rank_label, input_type)
    }
    return(summary_df)
  }
  input_name = c("Integrated","Clinical","Metabolomics","DataX","DataY","DataZ")
  inpcolors = c("Integrated"="#d95f02", "Clinical"="#5A9BD5", "Metabolomics"="#1b9e77",
                "DataX"="#7570b3", "DataY"="#e7298a", "DataZ"="#a6761d")

  ### ###### ###### ###
  df_list = list()
  # Collect featureFile from each folder
  for (prefix in names(input_list)) {
    file_path = file.path(input_list[[prefix]], featureFile)
    if (file.exists(file_path)) {
      tmp = read.csv(file_path, stringsAsFactors = TRUE)
      df_list[[prefix]] = comp_imp_avg(tmp)
    }
  }
  ## Ranks from single-domain models
  rank_list = list()
  for (i in 1:length(df_list)) {
    if(i == 1){#set integrated rank
      intg_rank = df_list[[i]] %>%
        mutate(intg_rank = rank(-Importance, ties.method = "first"))
    }else{#set other rank
      prefix = names(df_list)[i]
      rank_list[[prefix]] = rank_domain(
        df_list[[i]], prefix = prefix, model_name = input_name[i]
      )
    }
  }
  # Merge with integrated ranking
  rank_df = bind_rows(rank_list)
  df_plot = intg_rank %>% left_join(rank_df, by = "Feature")

  # Assign modality label
  df_plot$label_color = df_plot$input_type
  df_plot$label_color[is.na(df_plot$label_color)] = "Int"

  # Attach modality tag to feature name
  df_plot$Feature = paste0(df_plot$Feature, " [", substr(df_plot$label_color,1,1), "]")

  # Build y-axis label with integrated rank
  df_plot = df_plot %>%
    mutate(y_label = paste0(Feature, " (Int", intg_rank, ")")) %>%
    arrange(intg_rank) %>%
    mutate(y_label = factor(y_label, levels = rev(y_label)))

  # Plot
  ggplot(df_plot, aes(x = Importance, y = y_label)) +
    geom_col(fill = "#d95f02",color = "black",width = 0.8,linewidth = 0.5) +
    geom_text(aes(x=max(Importance)* 1.13,label = paste0("(",round(Importance, 2),")")),size = 5) +
    geom_text(aes(label = rank_label, color = input_type),hjust = -0.15,fontface = "bold",size = 5,na.rm = TRUE) +
    scale_color_manual(values = inpcolors) +
    labs(title = title,x = "Average Feature Importance",y = NULL) +
    theme_test(base_size = 14) +
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 16),
          axis.text = element_text(size = 14, color="black"),text = element_text(size = 16),legend.position = "top")+
    scale_x_continuous(limits = c(ifelse(min(df_plot$Importance) >=0,0,min(df_plot$Importance)), max(df_plot$Importance)*1.14)) #8x7
}

##### Plot combined ranking #####
#' input_list = c("I"="LN/integrated","C"="LN/clinical","M"="LN/metabolomics")
#' featureFile = "auto_plsda_feature.csv"
#' n_perm = 1000
#' out_folder = "LN"
combine_ranking <- function(input_list, featureFile, n_perm=1000, out_folder=""){
  #Read input Fold.Rep|Feature|Feature_Importance
  df_list = list()
  # Collect featureFile from each folder
  for (prefix in names(input_list)) {
    file_path = file.path(input_list[[prefix]], featureFile)
    if (file.exists(file_path)) {
      tmp = read.csv(file_path, stringsAsFactors = FALSE)
      df_list[[prefix]] = tmp
    }
  }
  df_list_scaled = lapply(df_list, function(df) {
    df$Feature_Importance = ave(df$Feature_Importance,df$Fold.Rep,
                                FUN = function(x) x / max(x, na.rm = TRUE) * 100) # Scale each input to 100
    df
  })

  combined_df = do.call(rbind, df_list_scaled)
  # Compute rank
  observed_rp_df = combined_df %>% group_by(Fold.Rep) %>% mutate(r = rank(-Feature_Importance, ties.method = "average")) %>%
    ungroup() %>% group_by(Feature) %>%
    summarise(
      RankProduct  = exp(mean(log(r))), # Rank product
      Rank_median  = median(r), # Median rank
      Rank_IQR     = IQR(r), # IQR
      .groups = "drop"
    )
  # Permutation Test
  set.seed(14)
  null_rps = matrix(NA, nrow = nrow(observed_rp_df), ncol = n_perm)
  for (i in 1:n_perm) {
    # Shuffle features within each fold
    perm_data <- combined_df %>% group_by(Fold.Rep) %>%
      mutate(Feature = sample(Feature)) %>% ungroup()
    # Compute RP
    null_rps[, i] <- perm_data %>% group_by(Fold.Rep) %>%
      mutate(r = rank(-Feature_Importance, ties.method = "average")) %>%
      arrange(Feature) %>% group_by(Feature) %>%
      summarise(RP = exp(mean(log(r))), .groups = 'drop') %>%
      pull(RP)
  }
  # Calculate P-values and FDR
  ## feature-specific distribution
  # result_rp <- observed_rp_df %>%
  #   mutate(
  #     p_value = sapply(1:nrow(observed_rp_df), function(j) {
  #       sum(null_rps[j, ] <= RankProduct[j]) / n_perm
  #     }),
  #     FDR = p.adjust(p_value, method = "fdr")
  #   ) %>% arrange(RankProduct)
  ## global distribution
  result_rp <- observed_rp_df %>%
    mutate(
      p_value = sapply(RankProduct, function(x) {mean(null_rps <= x, na.rm = TRUE)}),
      FDR = p.adjust(p_value, method = "fdr")
    ) %>% arrange(RankProduct)
  # Plot RP and export
  cat("Exporting rankvalues_outputs.csv to",out_folder,"..\n")
  write.csv(result_rp,file = paste0(out_folder,"/rankvalues_outputs.csv"), row.names = FALSE)

  cat("Ploting 2 plots of feature ranking on screen ..\n")
  RP_plot = ggplot(result_rp, aes(x = RankProduct, y = reorder(Feature, -RankProduct))) +
    geom_point(aes(fill = FDR < 0.05),size = 5,alpha = 0.9,shape = 21, color = "black", stroke = 0.4) +
    geom_text(aes(label = round(RankProduct, 2)),nudge_y = -0.4,size = 4.5,show.legend = FALSE) +
    scale_size_continuous(name = "Rank Product",trans = "reverse") +
    scale_fill_manual(values = c("#007BFF", "#D65FA1"), labels = c("FDR > 0.05", "FDR < 0.05")) +
    labs(title = "Rank Product-based Ranking with Significance",
         subtitle = paste(n_perm, "Permutations with FDR correction"),fill = NULL,
         x = "Rank Product", y = "", color = "Significance"
    ) + theme_bw() +
    theme(plot.title = element_text(size = 18), plot.subtitle = element_text(size = 12),
          axis.title = element_text(size = 16), axis.text = element_text(size = 16),
          text = element_text(size = 16)
    )

  # Plot feature median rank
  cutoff = median(result_rp$Rank_median, na.rm = TRUE)
  caption_text = paste0("<span style='color:#C90000;'> Median Rank = ",cutoff,"</span>")
  MedianRank_Plot = ggplot(result_rp, aes(x = Rank_median, y = reorder(Feature, -Rank_median))) +
    geom_errorbar(aes(xmin = Rank_median - (Rank_IQR/2), xmax = Rank_median + (Rank_IQR/2)),
      orientation = "y",width = 0.2, color = "black") +
    geom_point(aes(fill = Rank_median > cutoff),size = 5,alpha = 0.9,shape = 21,
               color = "black", stroke = 0.4) +
    geom_text(aes(label = round(Rank_median, 2)),nudge_y = -0.4,size = 4.5,show.legend = FALSE) +
    geom_vline(xintercept = cutoff, linetype = "dashed", color = "#D35400", linewidth = 1, alpha = 0.8) +
    scale_size_continuous(name = "Rank",trans = "reverse") +
    scale_fill_manual(values = c("#D65FA1", "#007BFF"), labels = c(" > Median Rank","< Median Rank")) +
    labs(x = "Rank", y = "", color = "Color",
         title = "Feature Median Rank", subtitle = "Point: Median Rank; Bar: IQR", fill = NULL,
         caption = caption_text) + theme_bw() +
    theme(plot.title = element_text(size = 18), plot.subtitle = element_text(size = 12),
          axis.title = element_text(size = 16), axis.text = element_text(size = 16),
          text = element_text(size = 16),
          plot.caption = element_markdown(size = 14, hjust = 0)
    )
  print(RP_plot)
  print(MedianRank_Plot)
}

# -------------------------------------------------------
# Functions to check modeling performance across models
# -------------------------------------------------------
##### Plot accuracy curves across training rounds #####
#' folder_path = "LN/clinical"
#' title = "Control vs LN (Clinical modal)"
plot_cross_model_accuracy <- function(folder_path, title="Accuracy Curves Across Training Rounds"){
  # Set folder path
  folder_path = folder_path
  # List all model CSV files
  files = list.files(folder_path, pattern = "_(model|eval)\\.csv$", full.names = TRUE)
  # ML input files
  mlinputfiles = files[grepl("(plsda|rf|elastic)_model\\.csv$", files)]
  # DL input files
  dlinputfiles = files[grepl("(gradcam|shap)_model\\.csv$", files)]
  # DL evaluation files
  evalinputfiles = files[grepl("(gradcam|shap)_eval\\.csv$", files)]
  # Read inputs
  ml_list = lapply(mlinputfiles, function(f) { #ML
    read.csv(f, stringsAsFactors = FALSE, row.names = 1) #read input
  })
  dl_list = lapply(dlinputfiles, function(f) { #DL
    read.csv(f, stringsAsFactors = FALSE, row.names = 1) #read input
  })
  eval_list = lapply(evalinputfiles, function(f) { #DL eval
    read.csv(f, stringsAsFactors = FALSE) #read input
  })
  eval_df = bind_rows( #DL eval
    map2_df(eval_list,toupper(sub(".*(gradcam|shap).*", "\\1", (evalinputfiles))),
            ~ tibble(Resample = .x$Fold.Rep,Accuracy = .x$accuracy,Model = .y))
  )
  dl_train_list = eval_df %>%
    group_by(Model,Resample) %>% summarise(Accuracy = mean(Accuracy, na.rm = TRUE), .groups = "drop")
  #ML
  ml_acc_df = map2_df(ml_list,toupper(sub(".*(plsda|rf|elastic).*", "\\1", (mlinputfiles))),
                      ~ tibble(Resample = rownames(.x),Accuracy = .x$Train.Accuracy,Model = .y))
  #DL
  dl_acc_df = dl_train_list %>% transmute(Resample, Accuracy, Model)
  acc_df = bind_rows(ml_acc_df, dl_acc_df) #Training acc
  acc_df$Type = ifelse(acc_df$Model %in% c("PLSDA", "RF", "ELASTIC"), "ML", "DL")
  acc_df = acc_df %>% group_by(Model) %>% mutate(Resample = row_number()) %>% ungroup()
  acc_df$Model = factor(acc_df$Model,levels = c("PLSDA", "RF", "GRADCAM", "SHAP", "ELASTIC"))

  ml_summary <- map2_df( #ML stat summary
    ml_list,toupper(sub(".*(plsda|rf|elastic).*", "\\1", (mlinputfiles))),
    ~ tibble(Model = .y,Train = mean(.x$Train.Accuracy, na.rm = TRUE),Test  = mean(.x$Test.Accuracy, na.rm = TRUE),
             Train_sd = sd(.x$Train.Accuracy, na.rm = TRUE),Test_sd  = sd(.x$Test.Accuracy, na.rm = TRUE),
             Train_median = median(.x$Train.Accuracy, na.rm = TRUE),Test_median  = median(.x$Test.Accuracy, na.rm = TRUE),
             Train_iqr = IQR(.x$Train.Accuracy, na.rm = TRUE),Test_iqr  = IQR(.x$Test.Accuracy, na.rm = TRUE))
  )
  dl_train <- dl_acc_df %>%
    group_by(Model) %>% summarise(Train = mean(Accuracy, na.rm = TRUE),Train_sd = sd(Accuracy, na.rm = TRUE),
                                  Train_median = median(Accuracy, na.rm = TRUE),Train_iqr = IQR(Accuracy, na.rm = TRUE),.groups = "drop")
  dl_test <- map2_df(
    dl_list,toupper(sub(".*(gradcam|shap).*", "\\1", (dlinputfiles))),
    ~ tibble(Model = .y,Test = mean(.x$Accuracy, na.rm = TRUE), Test_sd = sd(.x$Accuracy, na.rm = TRUE),
             Train_iqr = IQR(.x$Accuracy, na.rm = TRUE),Test_iqr  = IQR(.x$Accuracy, na.rm = TRUE))
  )
  dl_summary = left_join(dl_train, dl_test, by = "Model")
  acc_summary = bind_rows(ml_summary, dl_summary)
  caption_text <- acc_summary %>%
    mutate(txt = paste0(
      Model, ": ","Train = ", sprintf("%.2f", Train),"(±",sprintf("%.3f", Train_sd),
      "), Test = ",  sprintf("%.2f", Test),"(±",sprintf("%.3f", Test_sd),")"
    )) %>%
    summarise(line = paste(txt, collapse = " \n "))

  cat("Ploting 2 plots of accuracy across resamplings on screen ..\n")
  # barplot
  accolors = c("PLSDA"="#6A7FDB","RF"= "#FF0000","GRADCAM"="#00C96B","SHAP"= "#E39AD6","ELASTIC"= "#E69F00")
  bplot = ggplot(acc_df, aes(x=factor(Resample), y=Accuracy, fill=Model)) +
    geom_col(position=position_dodge(width = 0.8), width=0.7, alpha=0.7, color="black", linewidth=0.3) +
    geom_line(aes(group = 1), color="black", linewidth=1) +
    geom_point(color="black", size=1.5) +
    labs(title=title, x="Resample", y="Accuracy", caption=caption_text) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.1), expand = expansion(mult = c(0.02, 0.02))) +
    scale_fill_manual(values = accolors) + facet_wrap(~ Model, nrow = 2) +
    theme_test(base_size = 14) +
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 16),
          plot.caption = element_text(size = 14), axis.text = element_text(size = 12, color = "black"),
          text = element_text(size = 16), legend.position = "none", strip.text = element_text(face = "bold", size = 14)) #8x6

  # Line plot
  accolors = c("PLSDA"="#6A7FDB","RF"= "#D55E00","GRADCAM"="#00C96B","SHAP"= "#E39AD6","ELASTIC"="#00A9CF")
  lplot = ggplot(acc_df, aes(x = Resample, y = Accuracy, group = Model, color = Model)) +
    geom_line(linewidth = 1.2, alpha=0.8) + geom_point(size = 0.1) +
    labs(title=title,x = "Resample",y = "Accuracy",caption = caption_text) +
    scale_x_continuous(breaks = seq(0, max(acc_df$Resample), by = 2),limits = c(1, max(acc_df$Resample)),
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.1),limits = c(round(min(acc_df$Accuracy)-0.1,1), 1),
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_color_manual(values = accolors) +
    theme_test(base_size = 14) +
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 16),
          plot.caption  = element_text(size = 14),axis.text = element_text(size = 14, color="black"),
          text = element_text(size = 16),legend.position = "top") #8x7
  print(bplot)
  print(lplot)
}

##### Plot overall model performance (Mean ± SD) #####
#' folder_path = "LN/clinical"
#' title = "Control vs LN (Clinical modal)"
plot_cross_model_performance <- function(folder_path, title="Overall Model Performance (Mean ± SD)"){
  # Set folder path
  folder_path = folder_path
  # List all model CSV files
  files = list.files(folder_path, pattern = "_model\\.csv$", full.names = TRUE)
  # ML input files
  mlinputfiles = files[grepl("(plsda|rf|elastic)_model\\.csv$", files)]
  # DL input files
  dlinputfiles = files[grepl("(gradcam|shap)_model\\.csv$", files)]
  # Read inputs
  ml_list = lapply(mlinputfiles, function(f) {
    read.csv(f, stringsAsFactors = FALSE, row.names = 1) #read input
  })
  dl_list = lapply(dlinputfiles, function(f) {
    read.csv(f, stringsAsFactors = FALSE, row.names = 1) #read input
  })
  # Compute mean ± SD (ML)
  ml_stats = map2_df(ml_list, toupper(sub(".*(plsda|rf|elastic).*", "\\1", (mlinputfiles))),
                     ~ tibble(Model = .y,
                       Accuracy    = mean(.x$Test.Accuracy),Accuracy_sd = sd(.x$Test.Accuracy),
                       Sensitivity    = mean(.x$Sensitivity),Sensitivity_sd = sd(.x$Sensitivity),
                       Specificity    = mean(.x$Specificity),Specificity_sd = sd(.x$Specificity),
                       F1    = mean(.x$F1),F1_sd = sd(.x$F1),
                       AUC    = mean(.x$Overall.AUC),AUC_sd = sd(.x$Overall.AUC)
                     )
  )

  # Compute mean ± SD (DL)
  dl_stats = map2_df(dl_list, toupper(sub(".*(gradcam|shap).*", "\\1", (dlinputfiles))),
                     ~ tibble(Model = .y,
                       Accuracy    = mean(.x$Accuracy),Accuracy_sd = sd(.x$Accuracy),
                       Sensitivity    = mean(.x$Sensitivity),Sensitivity_sd = sd(.x$Sensitivity),
                       Specificity    = mean(.x$Specificity),Specificity_sd = sd(.x$Specificity),
                       F1    = mean(.x$F1.score),F1_sd = sd(.x$F1.score),
                       AUC    = mean(.x$AUC),AUC_sd = sd(.x$AUC)
                     )
  )
  # Combine
  metrics = c("Accuracy", "Sensitivity", "Specificity", "F1", "AUC")
  stat_long = bind_rows(ml_stats, dl_stats) %>%
    pivot_longer(cols = -Model,names_to = "Metric",values_to = "Value") %>%
    mutate(
      Type = ifelse(grepl("_sd$", Metric), "SD", "Mean"),
      Metric = gsub("_sd$", "", Metric)
    ) %>%
    pivot_wider(names_from = Type,values_from = Value)
  stat_long$Model = factor(stat_long$Model,levels = c("PLSDA", "RF", "GRADCAM", "SHAP", "ELASTIC"))
  stat_long$Metric = factor(stat_long$Metric,levels = metrics)

  # Set color shades per metric
  accolors = c("PLSDA"="#3B4992","RF"="#EE0000","GRADCAM"="#008B45","SHAP"="#CC79A7","ELASTIC"="#E69F00")
  metric_shades = lapply(accolors, function(col) {
    shades = rev(colorRampPalette(c("#E0E0E0",col))(length(metrics) + 1)[-1])
    setNames(shades, metrics)
  })
  stat_long$FillColor = mapply(function(m, met) metric_shades[[as.character(m)]][[as.character(met)]],
    stat_long$Model, stat_long$Metric)

  # Grouped bar plot
  ggplot(stat_long,aes(x = Model, y = Mean, fill = FillColor, group = Metric)) +
    geom_col(position = position_dodge(width = 0.85),width = 0.75,color = "black") +
    geom_errorbar(
      aes(ymin = Mean - SD, ymax = Mean + SD),
      position = position_dodge(width = 0.85),
      width = 0.2, linewidth = 0.5
    ) +
    geom_text(aes(y = Mean + SD + 0.02,label = sprintf("%.3f", Mean)),
              position = position_dodge(width = 0.85), size = 5, angle = 90, hjust = 0.05) +
    scale_fill_identity() +
    scale_y_continuous(limits = c(0, 1.1)) +
    labs(title=title, x = NULL, y = "Performance",
         caption = "Color intensity (from darker to lighter): Accuracy, Sensitivity, Specificity, F1, and AUC") +
    theme_classic(base_size = 14)+
    theme(plot.title = element_text(size = 16), axis.title = element_text(size = 16),
          plot.caption  = element_text(size = 14),axis.text = element_text(size = 14, color="black"),
          text = element_text(size = 16),legend.position = "top") #6x7
}
