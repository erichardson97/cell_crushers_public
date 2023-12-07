count_data_long <- read_tsv("../raw_prediction/2022BD_pbmc_cell_frequency.tsv")
df_subject_specimen <- read_tsv('../raw_prediction/2022BD_specimen.tsv')


## Perform median normalization
cytof_median_D0 <- count_data_long %>%
  group_by(cell_type_name)  %>%
  summarise(median = median(percent_live_cell, na.rm = T))

count_data_long <- count_data_long %>% left_join(df_subject_specimen[c("specimen_id", "subject_id")]) %>% group_by(subject_id, cell_type_name) %>% summarise(percent_live_cell = median(percent_live_cell))

cell_long_normalized_pre <-  count_data_long  %>%
  left_join(cytof_median_D0) %>%
  mutate(percent_live_cell_normalized = if_else(is.na(percent_live_cell) == T, NA, percent_live_cell/median))

## Reshape dataframe in wide format
cell_wide_normalized_pre <- cell_long_normalized_pre  %>%
  dplyr::select(cell_type_name, subject_id, percent_live_cell_normalized) %>%
  pivot_wider(names_from = "cell_type_name", values_from = percent_live_cell_normalized) %>%
  column_to_rownames("subject_id")%>%
  t() 

cellFreq_normalized_imputed = cell_wide_normalized_pre[rowMeans(is.na(cell_wide_normalized_pre)) < 1, ] %>%
  as.matrix() %>%
  impute.knn() %>%
  .$data

write.table(cellFreq_normalized_imputed, "2022_cell_frequency_normalized.tsv", sep = "\t")
