# ============================================================
# REGION-AWARE DIFFERENTIAL EXPRESSION: GSE48350 + GSE5281
# ============================================================
# Replaces the pooled ComBat + per-region limma workflow. Key changes
# (reviewer comments 3, 4, 5, 8):
#
# * No ComBat and no pooling of raw expression. The two studies differ in
#   tissue (GSE48350: homogenised tissue; GSE5281: laser-capture-microdissected
#   neurons) and several regions exist in only one study, so study and region
#   are partly confounded. Each study is analysed separately with limma.
# * One model per study: cell-means design for region x diagnosis, adjusted for
#   age and sex. Donors contributing several regions are handled with
#   duplicateCorrelation (GSE48350 only; GSE5281 provides no donor IDs).
# * Age matching: GSE48350 controls younger than the youngest AD donor (60 y)
#   are excluded (the AD group has no comparable ages).
# * Regions present in both studies (EC, hippocampus, SFG) are combined by a
#   random-effects (DerSimonian-Laird) meta-analysis of the limma log fold
#   changes, with Cochran's Q and I^2 for between-study heterogeneity.
# * Region heterogeneity: diagnosis x region interaction F-test within each study.
# * Composition sensitivity: limma re-fitted with marker-based cell-type scores
#   (neurons, astrocytes, oligodendrocytes, microglia, endothelial, pericytes).
# * Memory filter: DEGs in Consolidated_memory_genes.txt; genes that are memory DEGs in
#   >= 3 of the 7 regions are written to MLmicroarray/ThreeOrMoreLocally.txt.
#
# DEG definition (unchanged from the manuscript): BH FDR < 0.05 and |log2FC| > 0.5.
#
# Run from the repository root:  Rscript R_scripts/MicroModel.R
# Outputs: R_scripts/DEG_region_model/ and MLmicroarray/ThreeOrMoreLocally.txt
# ============================================================

# ----------------------------
# 0) Packages
# ----------------------------
# BiocManager::install(c("GEOquery", "affy", "limma", "statmod",
#                        "hgu133plus2.db", "hgu133plus2cdf"))
suppressPackageStartupMessages({
  library(GEOquery)
  library(affy)
  library(limma)
  library(statmod)
  library(hgu133plus2.db)
  library(hgu133plus2cdf)
  library(R.utils)
})
options(stringsAsFactors = FALSE, timeout = 600)
set.seed(42)

# ----------------------------
# 1) Paths (relative to the repository root)
# ----------------------------
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg))) else getwd()
repo_dir <- normalizePath(file.path(script_dir, ".."))
setwd(script_dir)

out_dir <- file.path(script_dir, "DEG_region_model")
dir.create(out_dir, showWarnings = FALSE)
memory_gene_file <- file.path(repo_dir, "MLmicroarray", "Consolidated_memory_genes.txt")
selected_gene_file <- file.path(repo_dir, "MLmicroarray", "ThreeOrMoreLocally.txt")

tar_files <- c(GSE48350 = "GSE48350_RAW.tar", GSE5281 = "GSE5281_RAW.tar")
stopifnot(all(file.exists(tar_files)), file.exists(memory_gene_file))

LFC_CUT <- 0.5
FDR_CUT <- 0.05
MIN_REGIONS <- 3
MIN_AGE <- 60

out <- function(name) file.path(out_dir, name)

# The default macOS (quartz) TIFF device does not support compression; drop the argument there
tiff <- function(..., compression = "lzw") {
  if (identical(getOption("bitmapType"), "quartz")) grDevices::tiff(...) else grDevices::tiff(..., compression = compression)
}

# ----------------------------
# 2) Helpers
# ----------------------------
extract_gsm <- function(x) regmatches(x, regexpr("GSM[0-9]+", x))

# Value of "key: value" from the characteristics_ch1* columns (column order differs between samples)
get_char <- function(pheno, key) {
  cols <- grep("^characteristics_ch1", colnames(pheno), value = TRUE)
  vals <- apply(pheno[, cols, drop = FALSE], 1, function(row) {
    row <- trimws(gsub(" ", " ", as.character(row)))
    hit <- row[grepl(paste0("^", key, "\\s*:"), row, ignore.case = TRUE)]
    if (length(hit)) trimws(sub("^[^:]*:", "", hit[1])) else NA_character_
  })
  unname(vals)
}

# Region names harmonised across studies. GEO labels GSE5281's region "Medial Temporal
# Gyrus"; the source publication (Liang et al., 2007) describes the middle temporal gyrus.
harmonise_region <- function(x) {
  x <- tolower(trimws(x))
  x <- gsub("-", "", x, fixed = TRUE)
  x <- gsub("singulate", "cingulate", x, fixed = TRUE)
  x[x == "medial temporal gyrus"] <- "middle temporal gyrus"
  x
}

map_probes_to_genes <- function(expr_mat) {
  p2s <- toTable(hgu133plus2SYMBOL)
  colnames(p2s) <- c("PROBEID", "symbol")
  p2s <- p2s[p2s$PROBEID %in% rownames(expr_mat), ]
  sums <- rowsum(expr_mat[p2s$PROBEID, , drop = FALSE], p2s$symbol)
  n_probes <- table(p2s$symbol)[rownames(sums)]
  agg <- sums / as.vector(n_probes)  # mean over probes, as in the original pipeline
  agg[sort(rownames(agg)), , drop = FALSE]
}

read_gene_list <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  tok <- toupper(trimws(unlist(strsplit(txt, "[[:space:],;]+"))))
  unique(tok[tok != "" & !grepl("^[0-9]+\\.?$", tok)])
}

# ----------------------------
# 3) RMA per study (cached)
# ----------------------------
rma_cache <- out("rma_gene_level.rds")
if (file.exists(rma_cache)) {
  gene_expr <- readRDS(rma_cache)
} else {
  gene_expr <- list()
  for (gse in names(tar_files)) {
    cel_dir <- file.path(gse, "CEL")
    dir.create(cel_dir, recursive = TRUE, showWarnings = FALSE)
    if (length(list.files(cel_dir)) == 0) untar(tar_files[[gse]], exdir = cel_dir)
    gz <- list.files(cel_dir, pattern = "\\.gz$", full.names = TRUE)
    if (length(gz)) invisible(sapply(gz, gunzip, overwrite = TRUE, remove = FALSE))
    cels <- list.files(cel_dir, pattern = "\\.[cC][eE][lL]$", full.names = TRUE)
    stopifnot(length(cels) > 0)
    e <- exprs(rma(ReadAffy(filenames = cels)))
    colnames(e) <- extract_gsm(colnames(e))
    gene_expr[[gse]] <- map_probes_to_genes(e)
  }
  saveRDS(gene_expr, rma_cache)
}

# ----------------------------
# 4) Metadata
# ----------------------------
p1 <- pData(getGEO("GSE48350", GSEMatrix = TRUE)[[1]])
p2 <- pData(getGEO("GSE5281", GSEMatrix = TRUE)[[1]])

m1 <- data.frame(
  sample = rownames(p1), study = "GSE48350",
  region = harmonise_region(get_char(p1, "brain region")),
  dx = ifelse(grepl("indiv", p1$title, ignore.case = TRUE), "CTL", "AD"),
  age = as.numeric(get_char(p1, "age \\(yrs\\)")),
  sex = tolower(get_char(p1, "gender")),
  braak = get_char(p1, "braak stage"),
  apoe = get_char(p1, "apoe genotype"),
  mmse = suppressWarnings(as.numeric(get_char(p1, "mmse")))
)
indiv <- get_char(p1, "individual")
m1$donor <- ifelse(m1$dx == "CTL", paste0("C", sub(",.*", "", indiv)), paste0("AD", sub(".*_", "", p1$title)))

m2 <- data.frame(
  sample = rownames(p2), study = "GSE5281",
  region = harmonise_region(get_char(p2, "organ region")),
  dx = ifelse(grepl("normal", get_char(p2, "disease state"), ignore.case = TRUE), "CTL", "AD"),
  # ">90 years" -> 90; one sample is recorded as "85 days" (a typo for years)
  age = as.numeric(regmatches(get_char(p2, "age"), regexpr("[0-9.]+", get_char(p2, "age")))),
  sex = tolower(get_char(p2, "sex")),
  braak = NA, apoe = NA, mmse = NA,
  donor = NA  # not provided by GEO
)

meta_all <- rbind(m1, m2)
meta_all <- meta_all[meta_all$sample %in% unlist(lapply(gene_expr, colnames)), ]
stopifnot(!anyNA(meta_all$region), !anyNA(meta_all$age), !anyNA(meta_all$sex))

# Age matching (GSE48350 controls younger than the youngest AD donor)
meta_all$excluded <- ifelse(meta_all$study == "GSE48350" & meta_all$dx == "CTL" & meta_all$age < MIN_AGE,
                            paste0("control younger than ", MIN_AGE, " y"), "")
write.csv(meta_all, out("sample_metadata_all.csv"), row.names = FALSE)
meta <- meta_all[meta_all$excluded == "", ]

# Study x region x diagnosis matrix, before and after exclusion
count_table <- function(m) {
  t <- as.data.frame.matrix(table(paste(m$study, m$region, sep = " | "), m$dx))
  t$n_donors_CTL <- tapply(m$donor[m$dx == "CTL"], paste(m$study, m$region, sep = " | ")[m$dx == "CTL"],
                           function(d) if (all(is.na(d))) NA else length(unique(d)))[rownames(t)]
  t
}
sample_matrix <- merge(count_table(meta_all), count_table(meta), by = "row.names", suffixes = c("_before", "_after"))
write.csv(sample_matrix, out("study_region_diagnosis_matrix.csv"), row.names = FALSE)

covariates <- do.call(rbind, lapply(split(meta, list(meta$study, meta$dx), drop = TRUE), function(d) data.frame(
  study = d$study[1], dx = d$dx[1], n_samples = nrow(d),
  n_donors = ifelse(all(is.na(d$donor)), NA, length(unique(d$donor))),
  age_mean = mean(d$age), age_sd = sd(d$age), age_min = min(d$age), age_max = max(d$age),
  female_fraction = mean(d$sex == "female"),
  braak_available = sum(!is.na(d$braak) & d$braak != "no info"),
  apoe_available = sum(!is.na(d$apoe) & d$apoe != "no info"),
  mmse_available = sum(!is.na(d$mmse))
)))
write.csv(covariates, out("covariates_by_group.csv"), row.names = FALSE)
cat("\nSamples after age matching:\n"); print(table(meta$study, meta$dx))

# ----------------------------
# 5) Diagnostics: PCA (no correction)
# ----------------------------
common_genes <- Reduce(intersect, lapply(gene_expr, rownames))
pca_plot <- function(mat, m, colour_by, title, file) {
  pc <- prcomp(t(mat), scale. = TRUE)
  f <- factor(m[[colour_by]])
  ve <- round(100 * pc$sdev^2 / sum(pc$sdev^2), 1)
  tiff(file, width = 7, height = 6, units = "in", res = 600, compression = "lzw")
  plot(pc$x[, 1], pc$x[, 2], col = as.integer(f), pch = ifelse(m$dx == "AD", 16, 1),
       xlab = paste0("PC1 (", ve[1], "%)"), ylab = paste0("PC2 (", ve[2], "%)"), main = title)
  legend("topright", c(levels(f), "AD (filled)", "CTL (open)"), col = c(seq_along(levels(f)), 1, 1),
         pch = c(rep(15, nlevels(f)), 16, 1), bty = "n", cex = 0.8)
  dev.off()
}
all_samples <- meta$sample
merged <- do.call(cbind, lapply(gene_expr, function(e) e[common_genes, intersect(colnames(e), all_samples)]))
meta_pca <- meta[match(colnames(merged), meta$sample), ]
pca_plot(merged, meta_pca, "study", "Uncorrected expression, both studies", out("PCA_by_study.tif"))
for (s in names(gene_expr)) {
  ms <- meta[meta$study == s, ]
  pca_plot(gene_expr[[s]][common_genes, ms$sample], ms, "region", paste(s, "by region"), out(paste0("PCA_", s, "_by_region.tif")))
}

# ----------------------------
# 6) Marker-based cell-type scores (composition sensitivity)
# ----------------------------
memory_genes <- read_gene_list(memory_gene_file)
markers <- list(
  neuron = c("SNAP25", "SYT1", "RBFOX3", "STMN2", "GAD1", "GAD2", "SLC17A7", "NEFL"),
  astrocyte = c("GFAP", "AQP4", "ALDH1L1", "GJA1", "SOX9", "SLC1A2"),
  oligodendrocyte = c("MBP", "MOG", "PLP1", "MOBP", "MAG", "CLDN11"),
  microglia = c("CX3CR1", "P2RY12", "CSF1R", "C1QB", "TMEM119", "AIF1"),
  endothelial = c("CLDN5", "FLT1", "VWF", "PECAM1", "ESAM"),
  pericyte = c("RGS5", "KCNJ8", "ABCC9", "ANPEP", "PDGFRB")
)
# Markers that are themselves candidate memory genes are dropped so a gene is never adjusted for itself.
markers <- lapply(markers, function(g) setdiff(intersect(g, common_genes), memory_genes))
cat("\nCell-type markers used:\n"); print(markers)

cell_scores <- function(e) {
  z <- t(scale(t(e[unique(unlist(markers)), ])))
  sapply(markers, function(g) colMeans(z[g, , drop = FALSE]))
}

# ----------------------------
# 7) limma per study: region x diagnosis cell means + age + sex
# ----------------------------
fit_study <- function(s, adjust_composition = FALSE) {
  ms <- meta[meta$study == s, ]
  e <- gene_expr[[s]][common_genes, ms$sample]
  ms$grp <- factor(make.names(paste(ms$region, ms$dx, sep = "_")))
  ms$sex <- factor(ms$sex)
  if (adjust_composition) {
    sc <- cell_scores(e)
    ms <- cbind(ms, sc)
    design <- model.matrix(as.formula(paste("~ 0 + grp + age + sex +", paste(names(markers), collapse = " + "))), ms)
  } else {
    design <- model.matrix(~ 0 + grp + age + sex, ms)
  }
  colnames(design) <- make.names(sub("^grp", "", colnames(design)))

  block <- NULL; cor <- NULL
  if (!all(is.na(ms$donor))) {
    block <- ms$donor
    cor <- duplicateCorrelation(e, design, block = block)$consensus.correlation
    cat(s, "within-donor correlation:", round(cor, 3), "\n")
  }
  fit <- lmFit(e, design, block = block, correlation = cor)

  regions <- sort(unique(ms$region))
  regions <- regions[sapply(regions, function(r) all(c("AD", "CTL") %in% ms$dx[ms$region == r]))]
  con_str <- sapply(regions, function(r) {
    paste0(make.names(paste(r, "AD", sep = "_")), " - ", make.names(paste(r, "CTL", sep = "_")))
  })
  cm <- makeContrasts(contrasts = con_str, levels = design)
  colnames(cm) <- regions
  fit2 <- eBayes(contrasts.fit(fit, cm))

  per_region <- lapply(regions, function(r) {
    tt <- topTable(fit2, coef = r, number = Inf, sort.by = "none", confint = TRUE)
    data.frame(gene = rownames(tt), study = s, region = r,
               logFC = tt$logFC, CI.L = tt$CI.L, CI.R = tt$CI.R,
               SE = fit2$stdev.unscaled[rownames(tt), r] * sqrt(fit2$s2.post[rownames(tt)]),
               t = tt$t, P.Value = tt$P.Value, adj.P.Val = tt$adj.P.Val,
               n_AD = sum(ms$region == r & ms$dx == "AD"), n_CTL = sum(ms$region == r & ms$dx == "CTL"))
  })
  per_region <- do.call(rbind, per_region)

  # Diagnosis x region interaction: does the AD effect differ between regions?
  interaction <- NULL
  if (length(regions) > 1) {
    diffs <- sapply(regions[-1], function(r) cm[, r] - cm[, regions[1]])
    colnames(diffs) <- paste0(make.names(regions[-1]), "_vs_", make.names(regions[1]))
    fi <- eBayes(contrasts.fit(fit, diffs))
    ti <- topTable(fi, number = Inf, sort.by = "none")
    interaction <- data.frame(gene = rownames(ti), study = s, F = ti$F, P.Value = ti$P.Value, adj.P.Val = ti$adj.P.Val)
  }
  list(per_region = per_region, interaction = interaction)
}

fits <- lapply(names(gene_expr), fit_study)
names(fits) <- names(gene_expr)
fits_adj <- lapply(names(gene_expr), fit_study, adjust_composition = TRUE)
names(fits_adj) <- names(gene_expr)

study_results <- do.call(rbind, lapply(fits, `[[`, "per_region"))
write.csv(study_results, out("limma_per_study_region.csv"), row.names = FALSE)
interaction <- do.call(rbind, lapply(fits, `[[`, "interaction"))
write.csv(interaction, out("diagnosis_by_region_interaction.csv"), row.names = FALSE)

# ----------------------------
# 8) Combine studies per region (random-effects meta-analysis)
# ----------------------------
dl_meta <- function(y, v) {
  # y, v: genes x studies matrices; DerSimonian-Laird random effects
  w <- 1 / v
  fe <- rowSums(w * y) / rowSums(w)
  q <- rowSums(w * (y - fe)^2)
  k <- ncol(y)
  tau2 <- pmax(0, (q - (k - 1)) / (rowSums(w) - rowSums(w^2) / rowSums(w)))
  ws <- 1 / (v + tau2)
  est <- rowSums(ws * y) / rowSums(ws)
  se <- sqrt(1 / rowSums(ws))
  data.frame(logFC = est, SE = se, CI.L = est - 1.96 * se, CI.R = est + 1.96 * se,
             P.Value = 2 * pnorm(-abs(est / se)), Q = q, Q_p = pchisq(q, k - 1, lower.tail = FALSE),
             I2 = ifelse(q > 0, pmax(0, (q - (k - 1)) / q), 0), tau2 = tau2)
}

combine_regions <- function(res) {
  out_list <- list()
  for (r in sort(unique(res$region))) {
    d <- res[res$region == r, ]
    studies <- unique(d$study)
    if (length(studies) == 1) {
      x <- d[, c("gene", "logFC", "SE", "CI.L", "CI.R", "P.Value")]
      x$Q <- NA; x$Q_p <- NA; x$I2 <- NA; x$tau2 <- NA
      x$source <- studies
    } else {
      y <- sapply(studies, function(s) d$logFC[d$study == s])
      v <- sapply(studies, function(s) d$SE[d$study == s]^2)
      x <- cbind(gene = d$gene[d$study == studies[1]], dl_meta(y, v))
      x$source <- paste("random-effects meta-analysis:", paste(studies, collapse = " + "))
    }
    x$region <- r
    x$adj.P.Val <- p.adjust(x$P.Value, "BH")
    x$n_AD <- sum(d$n_AD[!duplicated(d$study)])
    x$n_CTL <- sum(d$n_CTL[!duplicated(d$study)])
    x$DEG <- x$adj.P.Val < FDR_CUT & abs(x$logFC) > LFC_CUT
    out_list[[r]] <- x
  }
  do.call(rbind, out_list)
}

region_results <- combine_regions(study_results)
region_results_adj <- combine_regions(do.call(rbind, lapply(fits_adj, `[[`, "per_region")))
write.csv(region_results, out("region_DE_results.csv"), row.names = FALSE)
write.csv(region_results_adj, out("region_DE_results_composition_adjusted.csv"), row.names = FALSE)

regions_all <- sort(unique(region_results$region))
for (r in regions_all) {
  d <- region_results[region_results$region == r & region_results$DEG, ]
  write.csv(d[order(d$adj.P.Val), ], out(paste0(gsub(" ", "_", r), "_DEGs.csv")), row.names = FALSE)
}

# ----------------------------
# 9) DEG counts and memory-gene over-representation per region
# ----------------------------
mem_measured <- intersect(memory_genes, common_genes)
deg_summary <- do.call(rbind, lapply(regions_all, function(r) {
  d <- region_results[region_results$region == r, ]
  deg <- d$gene[d$DEG]
  a <- length(intersect(deg, mem_measured)); b <- length(deg) - a
  c <- length(mem_measured) - a; dd <- nrow(d) - a - b - c
  ft <- fisher.test(matrix(c(a, b, c, dd), 2), alternative = "greater")
  data.frame(region = r, source = d$source[1], n_AD = d$n_AD[1], n_CTL = d$n_CTL[1],
             DEGs = length(deg), up = sum(d$DEG & d$logFC > 0), down = sum(d$DEG & d$logFC < 0),
             memory_DEGs = a, memory_genes_measured = length(mem_measured),
             memory_fraction_in_DEGs = ifelse(length(deg), a / length(deg), NA),
             memory_fraction_in_background = length(mem_measured) / nrow(d),
             fisher_OR = unname(ft$estimate), fisher_p = ft$p.value)
}))
deg_summary$fisher_q <- p.adjust(deg_summary$fisher_p, "BH")
write.csv(deg_summary, out("DEG_summary_by_region.csv"), row.names = FALSE)
cat("\nDEGs per region:\n"); print(deg_summary[, c("region", "n_AD", "n_CTL", "DEGs", "up", "down", "memory_DEGs", "fisher_q")])

# ----------------------------
# 10) Memory DEGs in >= MIN_REGIONS regions
# ----------------------------
mem_deg <- region_results[region_results$DEG & region_results$gene %in% memory_genes, ]
counts <- table(mem_deg$gene)
selected <- sort(names(counts)[counts >= MIN_REGIONS])

per_gene <- do.call(rbind, lapply(names(counts), function(g) {
  d <- mem_deg[mem_deg$gene == g, ]
  data.frame(gene = g, n_regions = nrow(d), regions = paste(d$region, collapse = "; "),
             n_up = sum(d$logFC > 0), n_down = sum(d$logFC < 0),
             direction_consistent = length(unique(sign(d$logFC))) == 1,
             mean_logFC = mean(d$logFC),
             # GSE5281-only regions share donors and tissue type, so also record support from GSE48350 regions
             DE_in_region_with_GSE48350 = any(!d$source %in% "GSE5281"),
             selected = g %in% selected)
}))
per_gene <- per_gene[order(-per_gene$n_regions, per_gene$gene), ]
write.csv(per_gene, out("memory_DEGs_region_counts.csv"), row.names = FALSE)
writeLines(selected, selected_gene_file)
cat("\nMemory DEGs in >=", MIN_REGIONS, "regions:", length(selected), "\n"); print(selected)
cat("Saved", selected_gene_file, "\n")

# Composition sensitivity for the selected genes
sens <- merge(region_results[region_results$gene %in% selected, c("gene", "region", "logFC", "adj.P.Val", "DEG")],
              region_results_adj[region_results_adj$gene %in% selected, c("gene", "region", "logFC", "adj.P.Val", "DEG")],
              by = c("gene", "region"), suffixes = c("", "_composition_adjusted"))
write.csv(sens, out("selected_genes_composition_sensitivity.csv"), row.names = FALSE)
cat("\nSelected gene x region DE calls retained after composition adjustment:",
    sum(sens$DEG & sens$DEG_composition_adjusted), "of", sum(sens$DEG), "\n")

# Interaction tests for the selected genes
write.csv(interaction[interaction$gene %in% selected, ], out("selected_genes_interaction.csv"), row.names = FALSE)

# ----------------------------
# 11) Forest plots and heatmap for the selected genes
# ----------------------------
if (length(selected)) {
  ncol_p <- 4
  nrow_p <- ceiling(length(selected) / ncol_p)
  tiff(out("selected_genes_forest.tif"), width = 16, height = 3.2 * nrow_p, units = "in", res = 600, compression = "lzw")
  par(mfrow = c(nrow_p, ncol_p), mar = c(4, 12, 2.5, 1))
  for (g in selected) {
    d <- region_results[region_results$gene == g, ]
    d <- d[order(d$region), ]
    yy <- seq_len(nrow(d))
    plot(d$logFC, yy, xlim = range(c(d$CI.L, d$CI.R, -LFC_CUT, LFC_CUT)), yaxt = "n", pch = ifelse(d$DEG, 16, 1),
         xlab = "log2FC (AD - CTL), 95% CI", ylab = "", main = g)
    segments(d$CI.L, yy, d$CI.R, yy)
    abline(v = 0, lty = 2); abline(v = c(-LFC_CUT, LFC_CUT), lty = 3, col = "grey50")
    axis(2, at = yy, labels = paste0(d$region, ifelse(is.na(d$I2), "", sprintf(" (I2=%.0f%%)", 100 * d$I2))),
         las = 1, cex.axis = 0.75)
  }
  dev.off()

  hm <- sapply(regions_all, function(r) region_results$logFC[region_results$region == r][match(selected, region_results$gene[region_results$region == r])])
  rownames(hm) <- selected
  sig <- sapply(regions_all, function(r) region_results$DEG[region_results$region == r][match(selected, region_results$gene[region_results$region == r])])
  lim <- max(abs(hm), na.rm = TRUE)
  tiff(out("selected_genes_heatmap.tif"), width = 9, height = 1.5 + 0.35 * length(selected), units = "in", res = 600, compression = "lzw")
  par(mar = c(9, 8, 2, 6))
  image(seq_along(regions_all), seq_along(selected), t(hm), col = colorRampPalette(c("blue", "white", "red"))(101),
        zlim = c(-lim, lim), axes = FALSE, xlab = "", ylab = "", main = "log2FC (AD - CTL); * = DEG")
  axis(1, at = seq_along(regions_all), labels = regions_all, las = 2, cex.axis = 0.8)
  axis(2, at = seq_along(selected), labels = selected, las = 1, cex.axis = 0.8)
  text(rep(seq_along(regions_all), each = length(selected)), rep(seq_along(selected), length(regions_all)),
       ifelse(sig, "*", ""), cex = 1.2)
  dev.off()
}

# Volcano plots per region
for (r in regions_all) {
  d <- region_results[region_results$region == r, ]
  col <- ifelse(d$DEG & d$logFC > 0, "red", ifelse(d$DEG & d$logFC < 0, "blue", "grey"))
  tiff(out(paste0(gsub(" ", "_", r), "_volcano.tif")), width = 7, height = 6, units = "in", res = 600, compression = "lzw")
  plot(d$logFC, -log10(d$adj.P.Val), col = col, pch = 16, cex = 0.5, xlab = "log2FC (AD - CTL)",
       ylab = "-log10(FDR)", main = sprintf("%s (AD n=%d, CTL n=%d)\n%s", r, d$n_AD[1], d$n_CTL[1], d$source[1]), cex.main = 0.8)
  abline(h = -log10(FDR_CUT), v = c(-LFC_CUT, LFC_CUT), lty = 2, col = "grey40")
  dev.off()
}

# quartz writes uncompressed TIFFs; LZW-compress them with libtiff's tiffcp when available
if (nzchar(Sys.which("tiffcp"))) {
  for (f in list.files(out_dir, pattern = "\\.tif$", full.names = TRUE)) {
    tmp <- paste0(f, ".lzw")
    if (system2("tiffcp", c("-c", "lzw", shQuote(f), shQuote(tmp))) == 0) file.rename(tmp, f)
  }
}

writeLines(capture.output(sessionInfo()), out("sessionInfo.txt"))
cat("\nDone. Outputs in", out_dir, "\n")
