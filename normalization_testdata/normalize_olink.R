###
#normalize test data - OLINK
###

count_data_long <- read_tsv("../raw_prediction/2022BD_plasma_cytokine_concentration.tsv")
df_subject_specimen <- read_tsv('../raw_prediction/2022BD_specimen.tsv')
df_subject_specimen$dataset <- "2022"

cytokine_median_D0 <- count_data_long %>%
  group_by(protein_id)  %>%
  summarise(median = median(protein_expression, na.rm = T))

cytokine_long_normalized_pre <- count_data_long %>%  left_join(df_subject_specimen[c("specimen_id", "subject_id")]) %>% group_by(subject_id, protein_id) %>% summarise(protein_expression = median(protein_expression))

cytokine_long_normalized_pre <-  cytokine_long_normalized_pre %>%
    left_join(cytokine_median_D0) %>%
    mutate(protein_expression_normalized = if_else(is.na(protein_expression) == T, NA, protein_expression/median))

cytokine_wide_normalized_pre <- cytokine_long_normalized_pre  %>%
  dplyr::select(protein_id, subject_id, protein_expression_normalized) %>%
  pivot_wider(names_from = "protein_id", values_from = protein_expression_normalized) %>%
  column_to_rownames("subject_id")%>%
  t() 

cytokineFreq_normalized_imputed <- cytokine_wide_normalized_pre[rowMeans(is.na(cytokine_wide_normalized_pre)) < 1, ] %>%
  as.matrix() %>%
  impute.knn() %>%
  .$data
median_vals <- setNames(data.frame(c(apply(cytokineFreq_normalized_imputed, MARGIN=1, FUN=median))),"97")
missing <- cbind(cytokineFreq_normalized_imputed, median_vals)
write.table(missing %>% t(), "cytokine_frequency_test.tsv", sep="\t")

