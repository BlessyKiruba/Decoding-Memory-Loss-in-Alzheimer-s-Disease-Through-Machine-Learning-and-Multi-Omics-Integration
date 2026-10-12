# ============================================================
# REGION-AWARE DIFFERENTIAL EXPRESSION: GSE48350 + GSE5281
# ============================================================
# 
exp_annotated       # GSE5281 expression matrix
metadata            # GSE5281 metadata

exp_GSE48350        # GSE48350 expression matrix
meta_gse48350       # GSE48350 metadata

exp_annotated <- GSE5281


exp_GSE48350 <- GSE48350


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

meta_gse48350$Region <- gsub(
  "\u00A0", " ",
  meta_gse48350$Region
)

meta_gse48350$Region <- trimws(
  meta_gse48350$Region
)

meta_gse48350$Type <- trimws(
  meta_gse48350$Type
)

meta_gse48350$Region[
  meta_gse48350$Region == "post-central gyrus"
] <- "postcentral gyrus"

meta_gse48350$Region[
  meta_gse48350$Region == "postcentral gyrus"
] <- "Postcentral gyrus"

meta_gse48350$Region[
  meta_gse48350$Region == "hippocampus"
] <- "Hippocampus"

meta_gse48350$Type <- factor(
  meta_gse48350$Type,
  levels = c("normal", "AD")
)


table(meta_gse48350$Region, meta_gse48350$Type)


metadata$Study <- "GSE5281"
meta_gse48350$Study <- "GSE48350"

stopifnot(
  all(metadata$ID == colnames(exp_annotated))
)

library(limma)

expr_5281_norm <- as.matrix(exp_annotated)
expr_48350_norm <- as.matrix(exp_GSE48350)


common_genes <- intersect(
  rownames(expr_5281_norm),
  rownames(expr_48350_norm)
)

length(common_genes)

expr_5281_common <- expr_5281_norm[
  common_genes,
  ,
  drop = FALSE
]

expr_48350_common <- expr_48350_norm[
  common_genes,
  ,
  drop = FALSE
]

expr_5281_common <- expr_5281_norm[
  common_genes,
  ,
  drop = FALSE
]

expr_48350_common <- expr_48350_norm[
  common_genes,
  ,
  drop = FALSE
]



expr_merged <- cbind(
  expr_5281_common,
  expr_48350_common
)



meta_5281 <- metadata
meta_48350 <- meta_gse48350

meta_merged <- rbind(
  meta_5281,
  meta_48350
)

rownames(meta_merged) <- meta_merged$ID



dim(expr_merged)

dim(meta_merged)

stopifnot(
  all(colnames(expr_merged) == rownames(meta_merged))
)


table(
  meta_merged$Study,
  meta_merged$Region,
  meta_merged$Type
)


table(
  meta_merged$Study,
  meta_merged$Region,
  meta_merged$Type
)























library(ggplot2)

pca_before <- prcomp(
  t(expr_merged),
  scale. = FALSE
)

pca_before_df <- data.frame(
  PC1 = pca_before$x[, 1],
  PC2 = pca_before$x[, 2],
  Study = meta_merged$Study,
  Diagnosis = meta_merged$Type,
  Region = meta_merged$Region
)

ggplot(
  pca_before_df,
  aes(
    x = PC1,
    y = PC2,
    colour = Study,
    shape = Diagnosis
  )
) +
  geom_point(size = 3, alpha = 0.8) +
  theme_classic() +
  labs(
    title = "PCA Before Batch Correction",
    colour = "Study",
    shape = "Diagnosis"
  )







library(ggplot2)

pca_before <- prcomp(
  t(expr_merged),
  scale. = FALSE
)

pca_before_df <- data.frame(
  PC1 = pca_before$x[, 1],
  PC2 = pca_before$x[, 2],
  Study = meta_merged$Study,
  Diagnosis = meta_merged$Type,
  Region = meta_merged$Region
)

ggplot(
  pca_before_df,
  aes(
    x = PC1,
    y = PC2,
    colour = Study,
    shape = Diagnosis
  )
) +
  geom_point(size = 3, alpha = 0.8) +
  theme_classic() +
  labs(
    title = "PCA Before Batch Correction",
    colour = "Study",
    shape = "Diagnosis"
  )


library(sva)

combat_mod <- model.matrix(
  ~ Type + Region,
  data = meta_merged
)
expr_combat <- ComBat(
  dat = as.matrix(expr_merged),
  batch = meta_merged$Study,
  mod = combat_mod,
  par.prior = TRUE,
  prior.plots = FALSE
)


pca_after <- prcomp(
  t(expr_combat),
  scale. = FALSE
)

pca_after_df <- data.frame(
  PC1 = pca_after$x[, 1],
  PC2 = pca_after$x[, 2],
  Study = meta_merged$Study,
  Diagnosis = meta_merged$Type,
  Region = meta_merged$Region
)

ggplot(
  pca_after_df,
  aes(
    x = PC1,
    y = PC2,
    colour = Study,
    shape = Diagnosis
  )
) +
  geom_point(size = 3, alpha = 0.8) +
  theme_classic() +
  labs(
    title = "PCA After Batch Correction",
    colour = "Study",
    shape = "Diagnosis"
  )

ggplot(
  pca_after_df,
  aes(
    x = PC1,
    y = PC2,
    colour = Diagnosis,
    shape = Study
  )
) +
  geom_point(size = 3, alpha = 0.8) +
  theme_classic() +
  labs(
    title = "PCA After Batch Correction",
    colour = "Diagnosis",
    shape = "Study"
  )





shared_regions <- c(
  "Entorhinal Cortex",
  "Hippocampus",
  "Superior Frontal Gyrus"
)



# GSE5281
metadata$Region <- trimws(metadata$Region)

metadata$Region[
  tolower(metadata$Region) == "hippocampus"
] <- "Hippocampus"

metadata$Region[
  tolower(metadata$Region) == "entorhinal cortex"
] <- "Entorhinal Cortex"

metadata$Region[
  tolower(metadata$Region) == "superior frontal gyrus"
] <- "Superior Frontal Gyrus"


# GSE48350
meta_gse48350$Region <- trimws(meta_gse48350$Region)

meta_gse48350$Region[
  tolower(meta_gse48350$Region) == "hippocampus"
] <- "Hippocampus"

meta_gse48350$Region[
  tolower(meta_gse48350$Region) == "entorhinal cortex"
] <- "Entorhinal Cortex"

meta_gse48350$Region[
  tolower(meta_gse48350$Region) == "superior frontal gyrus"
] <- "Superior Frontal Gyrus"

meta_gse48350$Region[
  tolower(meta_gse48350$Region) == "post-central gyrus"
] <- "Postcentral gyrus"

meta_gse48350$Region[
  tolower(meta_gse48350$Region) == "postcentral gyrus"
] <- "Postcentral gyrus"


metadata$Study <- "GSE5281"
meta_gse48350$Study <- "GSE48350"

meta_merged <- rbind(
  metadata,
  meta_gse48350
)

rownames(meta_merged) <- meta_merged$ID


stopifnot(
  all(colnames(expr_merged) == rownames(meta_merged))
)

table(
  meta_merged$Region,
  meta_merged$Study,
  meta_merged$Type
)

shared_regions <- c(
  "Entorhinal Cortex",
  "Hippocampus",
  "Superior Frontal Gyrus"
)

for (reg in shared_regions) {
  
  cat("\n============================\n")
  cat(reg, "\n")
  cat("============================\n")
  
  print(
    table(
      meta_merged$Study[
        meta_merged$Region == reg
      ],
      meta_merged$Type[
        meta_merged$Region == reg
      ]
    )
  )
}


deg_shared <- list()

for (reg in shared_regions) {
  
  cat("\n============================\n")
  cat("Region:", reg, "\n")
  cat("============================\n")
  
  meta_reg <- meta_merged[
    meta_merged$Region == reg,
    ,
    drop = FALSE
  ]
  
  # Check that samples exist
  keep <- meta_reg$ID %in% colnames(expr_merged)
  
  meta_reg <- meta_reg[keep, , drop = FALSE]
  
  expr_reg <- expr_merged[
    ,
    meta_reg$ID,
    drop = FALSE
  ]
  
  # Ensure exact order
  expr_reg <- expr_reg[
    ,
    match(meta_reg$ID, colnames(expr_reg)),
    drop = FALSE
  ]
  
  stopifnot(
    all(colnames(expr_reg) == meta_reg$ID)
  )
  
  meta_reg$Study <- factor(
    meta_reg$Study
  )
  
  meta_reg$Type <- factor(
    meta_reg$Type,
    levels = c("normal", "AD")
  )
  
  print(
    table(
      meta_reg$Study,
      meta_reg$Type
    )
  )
  
  design <- model.matrix(
    ~ Study + Type,
    data = meta_reg
  )
  
  print(colnames(design))
  
  fit <- lmFit(
    expr_reg,
    design
  )
  
  fit <- eBayes(fit)
  
  deg <- topTable(
    fit,
    coef = "TypeAD",
    number = Inf,
    adjust.method = "BH"
  )
  
  deg$Gene <- rownames(deg)
  deg$Region <- reg
  
  deg_shared[[reg]] <- deg
}




head(deg_shared[["Entorhinal Cortex"]])
head(deg_shared[["Hippocampus"]])
head(deg_shared[["Superior Frontal Gyrus"]])


EC <- deg_shared[["Entorhinal Cortex"]]
Hippo <- deg_shared[["Hippocampus"]]
SFG <-deg_shared[["Superior Frontal Gyrus"]]
