#!/usr/bin/env Rscript
#install.packages('httr')
library('httr')
source("scripts/codebase.R")


args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  dir_rds_objects <- "./"
} else {
  dir_rds_objects <- args[1]
}


batch.factors = c("timepoint","infancy_vac","biological_sex","dataset","subject_id")

download_file <- function(file_name){
  final_url <- paste(base_url, file_name, sep="")
  response <- httr::GET(final_url)
  content <- httr::content(response, "text")
  cat(content, file=file_name)
}

base_url <- 'https://www.cmi-pb.org/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/training_data/'
filename_list <- c('2020LD_pbmc_cell_frequency.tsv', '2020LD_pbmc_gene_expression.tsv', '2020LD_plasma_ab_titer.tsv', '2020LD_plasma_cytokine_concentration.tsv',
                   '2020LD_subject.tsv', '2021LD_pbmc_cell_frequency.tsv', '2021LD_plasma_ab_titer.tsv', '2021LD_plasma_cytokine_concentration.tsv',
                   '2021LD_specimen.tsv', '2021LD_subject.tsv', '2021LD_pbmc_gene_expression.tsv', '2020LD_specimen.tsv')
for (filename in filename_list){
  download_file(filename)
  r <- read.csv(filename, sep = '\t')
  assign(str_replace(paste('d', strsplit(filename, '.',fixed=TRUE)[[1]][[1]], sep = ''), "LD", ""), r)
}

base_url <- 'https://www.cmi-pb.org/downloads/cmipb_challenge_datasets/current/2nd_challenge/raw_datasets/prediction_data/'
filename_list <- c('2022BD_pbmc_cell_frequency.tsv', '2022BD_pbmc_gene_expression.tsv', '2022BD_plasma_ab_titer.tsv',
                   '2022BD_plasma_cytokine_concentration.tsv', '2022BD_specimen.tsv', '2022BD_subject.tsv')
for (filename in filename_list){
  download_file(filename)
  r <- read.csv(filename, sep = '\t')
  cat(str_replace(paste('d', strsplit(filename, '.',fixed=TRUE)[[1]][[1]], sep = ''), "BD", ""))
  assign(str_replace(paste('d', strsplit(filename, '.',fixed=TRUE)[[1]][[1]], sep = ''), "BD", ""), r)
}


subject <-bind_rows(d2020_subject, d2021_subject, d2022_subject)
specimen <-bind_rows(d2020_specimen, d2021_specimen, d2022_specimen)
subject_specimen_total <- specimen %>%
  left_join(subject) %>%
  mutate(timepoint = planned_day_relative_to_boost)


### Reading plasma cytokines.
plasma_cytokine_concentrations_common_features <- Reduce(intersect, list(unique(d2020_plasma_cytokine_concentration$protein_id), unique(d2021_plasma_cytokine_concentration$protein_id), unique(d2022_plasma_cytokine_concentration$protein_id)))
plasma_cytokine_concentrations_long <-bind_rows(d2020_plasma_cytokine_concentration, d2021_plasma_cytokine_concentration, d2022_plasma_cytokine_concentration) %>%
  filter(protein_id %in% plasma_cytokine_concentrations_common_features)
plasma_cytokine_concentrations_wide <- plasma_cytokine_concentrations_long %>%
  dplyr::select(specimen_id, protein_id, protein_expression) %>%
  pivot_wider(names_from = protein_id, values_from = protein_expression)


### Reading ab titer.
d2020_plasma_antibody_levels <- d2020_plasma_ab_titer %>%
  mutate(isotype_antigen = paste0(isotype,"_", antigen))  %>%
  dplyr::select(-antigen, -isotype)
d2021_plasma_antibody_levels <- d2021_plasma_ab_titer %>%
  mutate(isotype_antigen = paste0(isotype,"_", antigen))  %>%
  dplyr::select(-antigen, -isotype)
d2022_plasma_antibody_levels <- d2022_plasma_ab_titer %>%
  mutate(isotype_antigen = paste0(isotype,"_", antigen))  %>%
  dplyr::select(-antigen, -isotype)
plasma_antibody_levels_common_features <- Reduce(intersect, list(unique(d2020_plasma_antibody_levels$isotype_antigen), unique(d2021_plasma_antibody_levels$isotype_antigen), unique(d2022_plasma_antibody_levels$isotype_antigen)))
plasma_antibody_levels_long <- bind_rows(d2020_plasma_antibody_levels, d2021_plasma_antibody_levels, d2022_plasma_antibody_levels) %>%
  filter(isotype_antigen %in% plasma_antibody_levels_common_features)
plasma_antibody_levels_wide <- plasma_antibody_levels_long %>%
  dplyr::select(specimen_id, isotype_antigen, MFI_normalised) %>%
  pivot_wider(names_from = isotype_antigen, values_from = MFI_normalised)



### Reading Cytof.
pbmc_cell_frequency_common_features <- Reduce(intersect, list(unique(d2020_pbmc_cell_frequency$cell_type_name), unique(d2021_pbmc_cell_frequency$cell_type_name), unique(d2022_pbmc_cell_frequency$cell_type_name)))
pbmc_cell_frequency_long <-bind_rows(d2020_pbmc_cell_frequency, d2021_pbmc_cell_frequency,d2022_pbmc_cell_frequency) %>%
  filter(cell_type_name %in% pbmc_cell_frequency_common_features)
pbmc_cell_frequency_wide <- pbmc_cell_frequency_long %>%
  pivot_wider(names_from = cell_type_name, values_from = percent_live_cell)


### Reading GEX.
colnames(d2020_pbmc_gene_expression) = c('versioned_ensembl_gene_id','specimen_id','tpm','raw_count')
colnames(d2021_pbmc_gene_expression) = c('versioned_ensembl_gene_id','specimen_id','tpm','raw_count')
colnames(d2022_pbmc_gene_expression) = c('versioned_ensembl_gene_id','specimen_id','tpm','raw_count')
pbmc_gene_expression_common_features <- Reduce(intersect, list(unique(d2020_pbmc_gene_expression$versioned_ensembl_gene_id), unique(d2021_pbmc_gene_expression$versioned_ensembl_gene_id), unique(d2022_pbmc_gene_expression$versioned_ensembl_gene_id)))
pbmc_gene_expression_long <-bind_rows(d2020_pbmc_gene_expression, d2021_pbmc_gene_expression,d2022_pbmc_gene_expression) %>%
  filter(versioned_ensembl_gene_id %in% pbmc_gene_expression_common_features)
pbmc_gene_expression_wide <- pbmc_gene_expression_long %>%
  dplyr::select(specimen_id, versioned_ensembl_gene_id, raw_count) %>%
  pivot_wider(names_from = versioned_ensembl_gene_id, values_from = raw_count)


##Combining into data_obj.
master_database_data <- list(
  
  subject_specimen = subject_specimen_total,
  plasma_antibody_levels = list(
    
    wide = plasma_antibody_levels_wide,
    long = plasma_antibody_levels_long
  ),
  plasma_cytokine_concentrations = list(
    
    wide = plasma_cytokine_concentrations_wide,
    long = plasma_cytokine_concentrations_long
  ),
  pbmc_cell_frequency_wide = list(
    
    wide = pbmc_cell_frequency_wide,
    long = pbmc_cell_frequency_long
  ),
  pbmc_gene_expression_wide = list(
    
    wide = pbmc_gene_expression_wide,
    long = pbmc_gene_expression_long
  )
)



##Specifying training vs. test.
training_dataset <- subset_dataset(master_database_data, c("2020_dataset", "2021_dataset"))
test_dataset <- subset_dataset(master_database_data, c("2022_dataset"))
subject_specimen <- master_database_data$subject_specimen  %>%
  mutate(timepoint = planned_day_relative_to_boost)


############OLINK.

## Before batch correction
data_obj <- master_database_data
olink_wide_before <- data_obj$plasma_cytokine_concentrations$wide  %>%
  column_to_rownames("specimen_id")%>%
  t() 
#pvca_analysis(olink_wide_before, data_obj$subject_specimen, batch.factors, plot_title = "Cytokine concetrations:  Raw data")

## Apply data normalization and batch correction
olink_data_processed = processOlink(data_obj, BatchCorrection = TRUE)
#pvca_analysis(olink_data_processed$normalized_data, data_obj$subject_specimen, batch.factors, plot_title = "Cytokine concetrations:  Normalized")
#pvca_analysis(olink_data_processed$batchCorrected_data, data_obj$subject_specimen, batch.factors, plot_title = "Cytokine concetrations:  Normalized + Batch corrected")


######CYTOF.

cell_wide_before <- data_obj$pbmc_cell_frequency$wide %>%
  column_to_rownames("specimen_id")%>%
  t() 

#pvca_analysis(cell_wide_before, data_obj$subject_specimen, batch.factors, plot_title = "Cell frequency:  Raw data")

## Apply data normalization and batch correction
cytof_data_processed = processCellFreq(data_obj, BatchCorrection = TRUE)
#pvca_analysis(cytof_data_processed$normalized_data, data_obj$subject_specimen, batch.factors, plot_title = "Cell frequency: Normalization")
#pvca_analysis(cytof_data_processed$batchCorrected_data, data_obj$subject_specimen, batch.factors, plot_title = "Cell frequency:  Normalization and batch correction")


abtiter_wide_before <- data_obj$plasma_antibody_levels$long %>%
  dplyr::select(isotype_antigen, specimen_id, MFI) %>%
  pivot_wider(names_from = "isotype_antigen", values_from = MFI) %>%
  column_to_rownames("specimen_id")%>%
  t() 
#pvca_analysis(abtiter_wide_before, data_obj$subject_specimen, batch.factors, plot_title = "Antibody titer:  Raw data")



####AB TITER
## Apply data normalization and batch correction
abtiter_data_processed = processAbtiter(data_obj, BatchCorrection = TRUE)
#pvca_analysis(abtiter_data_processed$normalized_data, data_obj$subject_specimen, batch.factors, plot_title = "Antibody titer: Normalization")
#pvca_analysis(abtiter_data_processed$batchCorrected_data, data_obj$subject_specimen, batch.factors, plot_title = "Antibody titer:  Normalization and batch correction")



#######RNASEQ.

##First matrix including everything.
rnaseq_countData <- master_database_data$pbmc_gene_expression_wide$wide %>%
  column_to_rownames("specimen_id") %>%
  t()  %>%
  as.data.frame()
colnames(rnaseq_countData) = as.integer(colnames(rnaseq_countData))
rnaseq_metaData <- master_database_data$subject_specimen %>%
  filter(specimen_id %in% colnames(rnaseq_countData)) %>%
  mutate(specimen_id1 = specimen_id) %>%
  column_to_rownames("specimen_id1")

##Matrix ONLY including training. This exists for the MAD calculation filtering etc
##that was used to select the Ensembl IDs.
rnaseq_countData_training <- training_dataset$pbmc_gene_expression_wide$wide %>%
  column_to_rownames("specimen_id") %>%
  t()  %>%
  as.data.frame()
colnames(rnaseq_countData_training) = as.integer(colnames(rnaseq_countData_training))

rnaseq_metaData_training <- training_dataset$subject_specimen %>%
  filter(specimen_id %in% colnames(rnaseq_countData_training)) %>%
  mutate(specimen_id1 = specimen_id) %>%
  column_to_rownames("specimen_id1")

##Download gene_90_30_export.tsv

cat(httr::content(httr::GET('https://raw.githubusercontent.com/CMI-PB/second-challenge-train-dataset-preparation/main/data/gene_90_38_export.tsv'), 'text'), file = 'gene_90_38_export.tsv')


##Other filtering step - this must be done on training ONLY,
threshold_proportion_greater_than_1 = 0.8
tpm_sum_infancy_subgroup <- rnaseq_countData_training %>%
  rownames_to_column("versioned_ensembl_gene_id") %>%
  pivot_longer(!versioned_ensembl_gene_id, values_to = "tpm", names_to = "specimen_id") %>%
  mutate(specimen_id = as.integer(specimen_id)) %>%
  left_join(subject_specimen) %>%
  group_by(dataset, versioned_ensembl_gene_id, infancy_vac) %>%
  #group_by(versioned_ensembl_gene_id, infancy_vac) %>%
  summarise(proportion_greater_than_1 = mean(tpm >= 1)) %>%
  pivot_wider(names_from = infancy_vac, values_from = proportion_greater_than_1)  %>%
  mutate(gene_meets_criterion_aP = aP >= threshold_proportion_greater_than_1 & wP <= (1 - threshold_proportion_greater_than_1),
         gene_meets_criterion_wP = wP >= threshold_proportion_greater_than_1 & aP <= (1 - threshold_proportion_greater_than_1)
  )  %>%
  filter((gene_meets_criterion_aP == TRUE & gene_meets_criterion_wP == FALSE) || (gene_meets_criterion_aP == FALSE & gene_meets_criterion_wP == TRUE))

gene_90_38_export <- read.csv('gene_90_38_export.tsv',sep='\t')
mito_genes <- gene_90_38_export %>%
  filter(substr(display_label, 1,3) == "MT-")

gene_90_38_shortlist <- gene_90_38_export %>%
  filter(biotype == "protein_coding") %>%
  filter(!versioned_ensembl_gene_id %in% mito_genes$versioned_ensembl_gene_id)

##TPM is average - have to make it just training data to match processed datat. 
tpm_shortlist <- rnaseq_countData_training %>%
  rownames_to_column("versioned_ensembl_gene_id") %>%
  filter(versioned_ensembl_gene_id %in% gene_90_38_shortlist$versioned_ensembl_gene_id) %>%
  pivot_longer(!versioned_ensembl_gene_id, values_to = "tpm", names_to = "specimen_id") %>%
  mutate(specimen_id = as.integer(specimen_id)) %>%
  left_join(subject_specimen) %>%
  group_by(versioned_ensembl_gene_id) %>%
  #group_by(versioned_ensembl_gene_id, infancy_vac) %>%
  summarise(proportion = mean(tpm >= 1))  %>%
  filter(proportion >= 0.3)


## Before batch correction - including 2022.
rnaseq_countData_v2 <- rnaseq_countData %>%
  rownames_to_column("versioned_ensembl_gene_id") %>%
  filter(versioned_ensembl_gene_id %in% gene_90_38_shortlist$versioned_ensembl_gene_id) %>%
  filter(!versioned_ensembl_gene_id %in% tpm_sum_infancy_subgroup$versioned_ensembl_gene_id) %>%
  filter(versioned_ensembl_gene_id %in% tpm_shortlist$versioned_ensembl_gene_id) %>%
  column_to_rownames("versioned_ensembl_gene_id")

test_subject_specimen_baseline <- subject_specimen %>% 
  filter(dataset %in% c("2022_dataset")) %>% 
  filter(timepoint %in% c(-30, -15, 0))
subject_specimen_baseline <- subject_specimen %>% 
  filter(timepoint %in% c(-30, -15, 0))

data_obj <- training_dataset
#mad calculations are ok on the full data as just using the specified subject specimens.
mad_2020 <- mad_calculations(rnaseq_countData_v2, data_obj$subject_specimen, c("2020_dataset"))
mad_2021 <- mad_calculations(rnaseq_countData_v2, data_obj$subject_specimen, c("2021_dataset"))
mad_shotlisted_genes = intersect(mad_2020$gene_id, mad_2021$gene_id)

rnaseq_countData_v3 <- rnaseq_countData_v2 %>%
  rownames_to_column("versioned_ensembl_gene_id") %>%
  filter(versioned_ensembl_gene_id %in% mad_shotlisted_genes) %>%
  column_to_rownames("versioned_ensembl_gene_id")

data_obj <- master_database_data
## Do the batch correction on EVERYTHING, 2022 included.
#pvca_analysis_rnaseq(rnaseq_countData_v3, data_obj$subject_specimen, batch.factors, plot_title = "RNASeq: Raw data")
batch_lebels = as.data.frame(colnames(rnaseq_countData_v3)) %>%
  rename(specimen_id = starts_with("colnames")) %>%
  mutate(specimen_id = as.integer(specimen_id)) %>%
  left_join(rnaseq_metaData) %>%
  dplyr::select(dataset)

rnaseq_batchCorrected = sva::ComBat_seq(as.matrix(rnaseq_countData_v3), batch = batch_lebels$dataset)
#pvca_analysis_rnaseq(rnaseq_batchCorrected, data_obj$subject_specimen, batch.factors, plot_title = "RNASeq: Batch correction")
rnaseq_normalised_data = list(
  
  metadata = rnaseq_metaData,
  raw_data = as.matrix(rnaseq_countData_v3),
  batchCorrected_data = rnaseq_batchCorrected
)

master_normalized_data <- list(
  
  subject_specimen = subject_specimen_total,
  abtiter = abtiter_data_processed,
  plasma_cytokine_concentrations = olink_data_processed,
  pbmc_cell_frequency = cytof_data_processed,
  pbmc_gene_expression = rnaseq_normalised_data
  
)

dir_rds_objects <- '/Users/erichardson/Documents/CMIPB_Processing/processed_combined'
save_dataframes_to_tsv(master_normalized_data)
