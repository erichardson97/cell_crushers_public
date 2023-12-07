count_data_long <- read_tsv("../raw_prediction/2022BD_plasma_ab_titer.tsv")
df_subject_specimen <- read_tsv('../raw_prediction/2022BD_specimen.tsv')
median_vals <- count_data_long %>% left_join(df_subject_specimen[c('specimen_id','subject_id')])  %>% mutate(isotype_antigen = paste0(isotype,"_", antigen)) %>% group_by(subject_id, isotype_antigen) %>% summarise(MFI_normalised = median(MFI_normalised)) 
median_flat <- median_vals  %>%
  dplyr::select(isotype_antigen, subject_id, MFI_normalised) %>%
  pivot_wider(names_from = "isotype_antigen", values_from = MFI_normalised) %>%
  column_to_rownames("subject_id")%>%
  t() 

write.table(median_flat, "IgG_titres_test.tsv",sep='\t')