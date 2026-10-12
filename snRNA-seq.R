
# ============================================================
# 1. LOAD PACKAGES
# ============================================================

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

if (!requireNamespace("Seurat", quietly = TRUE)) {
  install.packages("Seurat")
}

if (!requireNamespace("harmony", quietly = TRUE)) {
  install.packages("harmony")
}

library(Seurat)
library(harmony)
library(ggplot2)
library(patchwork)
library(cowplot)

# ============================================================
# 2. DEFINE A FUNCTION TO PROCESS EACH SAMPLE
# ============================================================

process_sample <- function(path, sample_name, diagnosis,
                           max_features, max_mt,
                           file_type = "10X") {

  message("Processing: ", sample_name)

  if (!dir.exists(path) && !file.exists(path)) {
    stop("Path does not exist: ", path)
  }

  if (file_type == "H5") {
    counts <- Read10X_h5(path)
  } else {
    counts <- Read10X(path)
  }

  # Create Seurat object
  obj <- CreateSeuratObject(
    counts = counts,
    project = sample_name,
    min.cells = 3,
    min.features = 200
  )

  # Sample metadata
  obj$Type <- diagnosis
  obj$Region <- sub("[0-9]+$", "", sample_name)
  obj$Sample <- sample_name

  # Mitochondrial percentage
  obj[["percent.mt"]] <- PercentageFeatureSet(
    obj,
    pattern = "^MT-"
  )

  # QC plots before filtering
  p <- VlnPlot(
    obj,
    features = c(
      "nFeature_RNA",
      "nCount_RNA",
      "percent.mt"
    ),
    ncol = 3
  ) + plot_annotation(title = sample_name)

  print(p)

  # Filter cells
  obj <- subset(
    obj,
    subset =
      nFeature_RNA > 200 &
      nFeature_RNA < max_features &
      percent.mt < max_mt
  )

  message(
    sample_name, ": ",
    ncol(obj), " cells retained after QC"
  )

  return(obj)
}

# ============================================================
# 3. SFG SAMPLES - GSE147528
# ============================================================

sfg_dir <- "C:/Users/Admin/Desktop/GSE147528_RAW"

sfg_names <- c(
  "SFG2", "SFG1", "SFG3", "SFG4", "SFG6",
  "SFG7", "SFG5", "SFG9", "SFG8", "SFG10"
)

sfg_gsm <- 4432635:4432644

sfg_files <- file.path(
  sfg_dir,
  paste0(
    "GSM", sfg_gsm, "_",
    sfg_names,
    "_raw_gene_bc_matrices_h5.h5"
  )
)

sfg_max_features <- rep(2500, 10)
sfg_max_mt <- rep(10, 10)

# Preserve the original SFG6 threshold
sfg_max_features[6] <- 500
sfg_max_mt[6] <- 1.5

sfg_objects <- list()

for (i in seq_along(sfg_names)) {

  sfg_objects[[sfg_names[i]]] <- process_sample(
    path = sfg_files[i],
    sample_name = sfg_names[i],
    diagnosis = "AD",
    max_features = sfg_max_features[i],
    max_mt = sfg_max_mt[i],
    file_type = "H5"
  )
}

# ============================================================
# 4. EC SAMPLES - GSE160936
# ============================================================

ec_paths <- c(#Path to the files)

ec_names <- paste0("EC", 1:12)

ec_type <- c(
  "C", "AD", "AD", "AD", "C", "AD",
  "AD", "C", "C", "AD", "C", "C"
)

ec_objects <- list()

for (i in seq_along(ec_names)) {

  ec_objects[[ec_names[i]]] <- process_sample(
    path = ec_paths[i],
    sample_name = ec_names[i],
    diagnosis = ec_type[i],
    max_features = ec_max_features[i],
    max_mt = ec_max_mt[i]
  )
}

# ============================================================
# 5. MTG SAMPLES - GSE188545
# ============================================================

mtg_dir <- "C:/Users/Admin/Desktop/GSE188545_RAW"

mtg_names <- paste0("MTG", 1:12)
mtg_paths <- file.path(mtg_dir, as.character(1:12))

mtg_type <- c(
  rep("AD", 6),
  rep("C", 6)
)

mtg_objects <- list()

for (i in seq_along(mtg_names)) {

  mtg_objects[[mtg_names[i]]] <- process_sample(
    path = mtg_paths[i],
    sample_name = mtg_names[i],
    diagnosis = mtg_type[i],
    max_features = mtg_max_features[i],
    max_mt = mtg_max_mt[i]
  )
}

# ============================================================
# 6. HC SAMPLES - GSE175814
# ============================================================

hc_dir <- "C:/Users/Admin/Desktop/GSE175814_RAW"

hc_names <- paste0("HC", 1:4)
hc_paths <- file.path(hc_dir, as.character(1:4))

hc_type <- c("AD", "C", "AD", "C")

VlnPlot(hc_paths, features = c("nFeature_RNA","nCount_RNA","percent.mt","percent.rbp"), ncol = 3)

hc_objects <- list()

for (i in seq_along(hc_names)) {

  hc_objects[[hc_names[i]]] <- process_sample(
    path = hc_paths[i],
    sample_name = hc_names[i],
    diagnosis = hc_type[i],
    max_features = hc_max_features[i],
    max_mt = hc_max_mt[i]
  )
}

# ============================================================
# 7. MERGE ALL SAMPLES
# ============================================================

all_samples <- c(
  sfg_objects,
  ec_objects,
  mtg_objects,
  hc_objects
)

# Check sample count and names
print(length(all_samples))
print(names(all_samples))

merged.data <- merge(
  x = all_samples[[1]],
  y = all_samples[-1],
  add.cell.ids = names(all_samples),
  project = "Merged_SingleCell"
)

# Confirm sample and diagnosis labels
print(table(merged.data$orig.ident))
print(table(merged.data$Type, merged.data$orig.ident))

# Save merged object
saveRDS(
  merged.data,
  file = "merged.data.rds"
)

# ============================================================
# 8. NORMALISATION AND PCA
# ============================================================

merged.data <- NormalizeData(
  merged.data,
  verbose = FALSE
)

merged.data <- FindVariableFeatures(
  merged.data,
  selection.method = "vst",
  nfeatures = 2000,
  verbose = FALSE
)

merged.data <- ScaleData(
  merged.data,
  verbose = FALSE
)

merged.data <- RunPCA(
  merged.data,
  npcs = 30,
  verbose = FALSE
)

pca_plot <- DimPlot(
  merged.data,
  reduction = "pca",
  group.by = "orig.ident"
)

ggsave(
  "PCA_before_Harmony.tiff",
  plot = pca_plot,
  width = 9,
  height = 7,
  dpi = 600
)

# ============================================================
# 9. HARMONY INTEGRATION
# ============================================================

merged.data <- RunHarmony(
  merged.data,
  group.by.vars = "orig.ident",
  plot_convergence = TRUE
)

# Harmony embeddings
harmony_embeddings <- Embeddings(
  merged.data,
  reduction = "harmony"
)

print(harmony_embeddings[1:5, 1:5])

# UMAP using Harmony embeddings
merged.data <- RunUMAP(
  merged.data,
  reduction = "harmony",
  dims = 1:10,
  verbose = FALSE
)

# Nearest neighbours and clustering
merged.data <- FindNeighbors(
  merged.data,
  reduction = "harmony",
  dims = 1:10
)

merged.data <- FindClusters(
  merged.data,
  resolution = 0.5,
  algorithm = 4
)

# ============================================================
# 10. PLOT UMAP
# ============================================================

p_sample <- DimPlot(
  merged.data,
  reduction = "umap",
  group.by = "orig.ident"
)

p_cluster <- DimPlot(
  merged.data,
  reduction = "umap",
  group.by = "seurat_clusters",
  label = TRUE
)

p_diagnosis <- DimPlot(
  merged.data,
  reduction = "umap",
  group.by = "Type"
)

ggsave(
  "UMAP_Harmony_by_sample.tiff",
  plot = p_sample,
  width = 9,
  height = 7,
  dpi = 600
)

ggsave(
  "UMAP_Harmony_clusters.tiff",
  plot = p_cluster,
  width = 7,
  height = 7,
  dpi = 600
)

ggsave(
  "UMAP_Harmony_diagnosis.tiff",
  plot = p_diagnosis,
  width = 7,
  height = 7,
  dpi = 600
)

# ============================================================
# 11. SAVE FINAL OBJECT
# ============================================================

saveRDS(
  merged.data,
  file = "merged.data_harmony.rds"
)

# Final checks
print(dim(merged.data))
print(table(merged.data$Type))
print(table(merged.data$orig.ident))
print(table(Idents(merged.data)))

# ============================================================
# 12. VISUALIZATION
# ============================================================

library(Seurat)
library(ggplot2)

# Genes
genes <- c("COPS5","CEBPB","TNFRSF1A", "SGK1","KAT2B","VPS35", "NFIA")

# Output directory
dir.create("Gene_Expression_Plots", showWarnings = FALSE)

# Check genes
genes_present <- genes[genes %in% rownames(merged.data)]
genes_missing <- setdiff(genes, rownames(merged.data))

if (length(genes_missing) > 0) {
  message("Missing genes: ", paste(genes_missing, collapse = ", "))
}

genes_present






group_vars <- c("cellType", "region", "Type", "orig.ident")

for (grp in group_vars) {
  
  p <- DotPlot(
    AD,
    features = genes_present,
    group.by = grp
  ) +
    RotatedAxis() +
    theme_classic() +
    theme(
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        vjust = 1
      ),
      axis.title = element_blank(),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9)
    )
  
  ggsave(
    filename = paste0(
      "Gene_Expression_Plots/DotPlot_",
      grp,
      ".tiff"
    ),
    plot = p,
    width = 10,
    height = 7,
    dpi = 600,
    compression = "lzw"
  )
}
















for (grp in group_vars) {
  
  for (gene in genes_present) {
    
    p <- VlnPlot(
      merged.data,
      features = gene,
      group.by = grp,
      pt.size = 0.5
    ) +
      theme_classic() +
      theme(
        axis.text.x = element_text(
          angle = 45,
          hjust = 1,
          vjust = 1
        ),
        axis.title.x = element_blank()
      ) +
      ggtitle(gene)
    
    ggsave(
      filename = paste0(
        "Gene_Expression_Plots/Violin_",
        gene,
        "_by_",
        grp,
        ".tiff"
      ),
      plot = p,
      width = 10,
      height = 6,
      dpi = 600,
      compression = "lzw"
    )
  }
}











for (gene in genes_present) {
  
  p <- FeaturePlot(
    merged.data,
    features = gene,
    reduction = "umap"
  ) +
    theme_classic() +
    ggtitle(gene)
  
  ggsave(
    filename = paste0(
      "Gene_Expression_Plots/FeaturePlot_",
      gene,
      ".tiff"
    ),
    plot = p,
    width = 8,
    height = 7,
    dpi = 600,
    compression = "lzw"
  )
}









library(Seurat)
library(ggplot2)

#genes <- c("COPS5", "PAIP2", "ATAD1", "EGR1")

group_vars <- c(
  "cellType",
  "region",
  "Type",
  "orig.ident"
)

dir.create("Gene_Expression_Plots_ddd", showWarnings = FALSE)

for (grp in group_vars) {
  
  p <- VlnPlot(
    merged.data,
    features = genes,
    group.by = grp,
    pt.size = 1,
    ncol = 2
  ) &
    theme_classic() &
    theme(
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        vjust = 1
      ),
      axis.title.x = element_blank(),
      strip.text = element_text(
        face = "bold",
        size = 11
      )
    )
  
  ggsave(
    filename = paste0(
      "Gene_Expression_Plots/Violin_",
      grp,
      "_ALL_GENES.tiff"
    ),
    plot = p,
    width = 12,
    height = 9,
    dpi = 600,
    compression = "lzw"
  )
}









library(Seurat)
library(ggplot2)

#genes <- c("COPS5", "PAIP2", "ATAD1", "EGR1")

# Keep only AD cells
AD <- subset(
  merged.data,
  subset = Type == "AD"
)

# Metadata variables
group_vars <- c(
  "cellType",
  "region",
  "orig.ident"
)

# Output directory
dir.create(
  "Gene_Expression_Plots/AD",
  recursive = TRUE,
  showWarnings = FALSE
)

# Plot all 4 genes together
for (grp in group_vars) {
  
  p <- VlnPlot(
    AD,
    features = genes,
    group.by = grp,
    pt.size = 0.5,
    ncol = 2
  ) &
    theme_classic() &
    theme(
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        vjust = 1
      ),
      axis.title.x = element_blank(),
      strip.text = element_text(
        face = "bold",
        size = 11
      )
    )
  
  ggsave(
    filename = paste0(
      "Gene_Expression_Plots/AD/Violin_AD_",
      grp,
      "_ALL_GENES.tiff"
    ),
    plot = p,
    width = 12,
    height = 9,
    dpi = 600,
    compression = "lzw"
  )
}













p <- VlnPlot(
  merged.data,
  features = genes,
  group.by = "cellType",
  split.by = "Type",
  pt.size = 0.5,
  ncol = 3
) &
  theme_classic() &
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1
    ),
    axis.title.x = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 11
    )
  )

ggsave(
  filename = "split_ALL_GENES.tiff",
  plot = p,
  width = 9,
  height = 5,
  dpi = 600,
  compression = "lzw"
)

saveRDS(
  merged.data,
  file = "merged.data.rds"
)

