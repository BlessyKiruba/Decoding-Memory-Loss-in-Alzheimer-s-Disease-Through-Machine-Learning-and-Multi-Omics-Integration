metadata <- Supplementary_Table
head(metadata)

GSE5281$Gene -> rownames(GSE5281)
library(dplyr)

GSE5281 <- GSE5281 %>%
  filter(!is.na(Gene), Gene != "")
GSE5281 <- GSE5281 %>%
  group_by(Gene) %>%
  summarise(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )


rownames(GSE5281) <- GSE5281$Gene
GSE5281$Gene <- NULL


head(GSE5281)



GSE5281 <- as.data.frame(GSE5281)

write.csv(sampleInfo, "SampleInfoGSE48350.csv")

metadata <- metadata[match(colnames(GSE5281), metadata$ID), ]

stopifnot(all(metadata$ID == colnames(GSE5281)))



table(metadata$Region, metadata$Type)

metadata$Region[
  metadata$Region == "post-central gyrus"
] <- "postcentral gyrus"


library(limma)

regions <- c(
  "Medial Temporal Gyrus",
  "Posterior Cingulate",
  "Primary Visual Cortex"
)


regions <- c("postcentral gyrus")

metadata$Region <- gsub("\u00A0", " ", metadata$Region)
metadata$Region <- trimws(metadata$Region)

metadata$Type <- gsub("\u00A0", " ", metadata$Type)
metadata$Type <- trimws(metadata$Type)

metadata$Type <- ifelse(
  metadata$Type == "Alzheimer's Disease",
  "AD",
  metadata$Type
)

metadata$Type <- factor(
  metadata$Type,
  levels = c("normal", "AD")
)


expr <- as.matrix(GSE5281)

expr_log2 <- log2(expr + 1)
summary(as.numeric(expr_log2))

quantile(
  as.numeric(expr_log2),
  probs = c(0, 0.01, 0.25, 0.5, 0.75, 0.99, 1),
  na.rm = TRUE
)

library(limma)

expr_norm <- normalizeBetweenArrays(
  expr_log2,
  method = "quantile"
)

rownames(expr_norm) <- rownames(GSE5281)
colnames(expr_norm) <- colnames(GSE5281)

GSE5281 <- expr_norm


GSE5281 <- GSE5281
GSE48350 <- GSE5281

GSE5281

GSE48350



deg_results <- list()

for (reg in regions) {
  
  # samples from this region
  meta_reg <- metadata[metadata$Region == reg, ]
  
  # expression matrix for these samples
  expr_reg <- GSE5281[, meta_reg$ID, drop = FALSE]
  
  # make sure ordering is identical
  expr_reg <- expr_reg[, meta_reg$ID, drop = FALSE]
  
  # diagnosis factor
  meta_reg$Type <- factor(
    meta_reg$Type,
    levels = c("normal", "AD")
  )
  
  # design matrix
  design <- model.matrix(~ Type, data = meta_reg)
  
  # limma
  fit <- lmFit(expr_reg, design)
  fit <- eBayes(fit)
  
  # AD vs normal
  deg <- topTable(
    fit,
    coef = "TypeAD",
    number = Inf,
    adjust.method = "BH"
  )
  
  deg$Gene <- rownames(deg)
  deg$Region <- reg
  
  deg_results[[reg]] <- deg
}

head(deg_results[["Medial Temporal Gyrus"]])
head(deg_results[["Posterior Cingulate"]])
head(deg_results[["Primary Visual Cortex"]])





deg_results[[]]
MTG <-deg_results[["Medial Temporal Gyrus"]]
PC <- deg_results[["Posterior Cingulate"]]
PVC <- deg_results[["Primary Visual Cortex"]]
PCG <- deg_results[["postcentral gyrus"]]
