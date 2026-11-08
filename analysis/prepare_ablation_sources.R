setwd(commandArgs(trailingOnly = TRUE)[1])
# 1. Read existing fitted-result tables only; no training or inference.
options(digits = 17)
root <- "Source_Data"
read_tsv <- function(path) read.delim(path, check.names = FALSE)
save_source <- function(d, figure, name) {
  dest <- file.path(root, figure, paste0(name, ".tsv"))
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  write.table(d, dest, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
  back <- read_tsv(dest)
  stopifnot(isTRUE(all.equal(d, back, check.attributes = FALSE, tolerance = 1e-13)))
  invisible(dest)
}
summarize <- function(d, keys, value) {
  f <- interaction(d[keys], drop = TRUE, lex.order = TRUE)
  out <- lapply(split(d, f), function(q) {
    z <- q[1, keys, drop = FALSE]
    z$estimate <- mean(q[[value]])
    z$sd <- sd(q[[value]])
    z$n_seeds <- nrow(q)
    z$negative_seeds <- sum(q[[value]] < 0)
    z$lower <- z$estimate - z$sd
    z$upper <- z$estimate + z$sd
    z
  })
  do.call(rbind, out)
}
source_index <- data.frame(panel = character(), source_file = character(), selection = character(), statistic = character())
register <- function(panel, file, selection, statistic) {
  source_index <<- rbind(source_index, data.frame(panel, source_file = file, selection, statistic))
}

# 2. Figure 6a: natural-availability within-cancer component ablations.
paths <- c(
  "Inputs/ablations/encoder_paired_survival_per_seed.tsv",
  "Inputs/ablations/component_paired_survival_per_seed.tsv",
  "Inputs/ablations/simplified_paired_survival_per_seed.tsv")
keep <- list(c("no_pretraining", "no_c2", "no_c3", "random_layout"),
             c("no_risk_consistency", "mean_fusion", "no_modality_dropout"),
             "mean_fusion")
a <- do.call(rbind, lapply(seq_along(paths), function(i) {
  q <- read_tsv(paths[i])
  q <- q[q$scenario == "natural_availability" & q$metric == "within_cancer_c_index" & q$variant %in% keep[[i]], ]
  if (i == 3) q$variant <- "mean_no_risk_consistency"
  q$delta <- q$ablation - q$full
  q$source_file <- paths[i]
  register("Figure6a", paths[i], paste(keep[[i]], collapse = ";"), "ablation - full; five seed differences")
  q
}))
labels <- c(no_pretraining = "No masked pretraining", no_c2 = "No C2", no_c3 = "No C3",
            random_layout = "Random layout", no_risk_consistency = "No risk consistency",
            mean_fusion = "Mean fusion", no_modality_dropout = "No modality dropout",
            mean_no_risk_consistency = "Mean + no risk consistency")
a$label <- unname(labels[a$variant])
stopifnot(nrow(a) == 40, length(unique(a$variant)) == 8)
sa <- summarize(a, c("variant", "label", "source_file"), "delta")
sa$contribution_colour <- ifelse(sa$estimate < 0 & sa$negative_seeds >= 4, "#76569A", "#888888")
save_source(a, "Figure6", "panel_a_seed_differences")
save_source(sa, "Figure6", "panel_a_summary")

# 3. Figure 6b: use named delta_mean, NOT delta_sd (the adjacent column).
bpath <- "Inputs/ablations/paired_measured_summary.tsv"
b <- read_tsv(bpath)
b <- b[b$variant == "no_modality_dropout" & b$metric == "within_cancer_c_index", ]
stopifnot(nrow(b) == 31, all(abs(b$delta_mean - (b$full_mean - b$ablation_mean)) < 1e-12))
sb <- aggregate(delta_mean ~ observed_count, b, mean)
sb$n_combinations <- as.integer(table(factor(b$observed_count, levels = 1:5)))
stopifnot(identical(sb$n_combinations, c(5L, 10L, 10L, 5L, 1L)))
save_source(b, "Figure6", "panel_b_combination_differences")
save_source(sb, "Figure6", "panel_b_group_means")
register("Figure6b", bpath, "variant=no_modality_dropout; metric=within_cancer_c_index", "named delta_mean = full_mean - ablation_mean; average by observed_count")

# 4. Figure 6c-e: retain saved paired intervals and cohort-specific intervals.
cpath <- "Analysis/measured/rna_addition_ci.tsv"
c <- read_tsv(cpath)
c <- c[c$metric == "within_cancer_c_index", ]
stopifnot(nrow(c) == 4)
save_source(c, "Figure6", "panel_c_rna_addition_ci")
register("Figure6c", cpath, "metric=within_cancer_c_index", "C(RNA + measured modality) - C(RNA); paired patient-bootstrap 95% CI")

# Final panel c averages the overlapping combinations within each seed first.
raw <- read_tsv("Inputs/ablations/paired_measured_per_seed.tsv")
raw <- raw[raw$variant == "no_modality_dropout" & raw$metric == "within_cancer_c_index", ]
seed <- aggregate(full_minus_ablation ~ seed + observed_count, raw, mean)
stopifnot(nrow(raw) == 155, nrow(seed) == 25)
save_source(seed, "Figure6", "panel_b_seed_values")
