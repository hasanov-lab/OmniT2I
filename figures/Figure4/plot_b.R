setwd(".")

library(ggplot2)

## 1. Load current model results and the same seed-level aggregation.
out <- "Plots/Figure4"
src <- "Source_Data/Figure4"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
data <- read.delim(file.path(src, "model_per_seed.tsv"))
stopifnot(nrow(data) == 10 * 31 * 5,
          !anyDuplicated(data[c("model", "seed", "source_subset")]))
## 2. Average subsets within each modality count, then summarize five seeds.
seed_means <- aggregate(cbind(pooled_c_index, within_cancer_c_index) ~
                       model + seed + observed_count, data, mean)
long <- rbind(
  data.frame(seed_means[c("model", "seed", "observed_count")],
             scope = "All patients", c_index = seed_means$pooled_c_index),
  data.frame(seed_means[c("model", "seed", "observed_count")],
             scope = "Within cancer", c_index = seed_means$within_cancer_c_index))
means <- aggregate(c_index ~ model + observed_count + scope, long, mean)
sds <- aggregate(c_index ~ model + observed_count + scope, long, sd)
names(means)[4] <- "mean"; names(sds)[4] <- "sd"
summary <- merge(means, sds, by = c("model", "observed_count", "scope"))
summary$n_combinations <- c(5, 10, 10, 5, 1)[summary$observed_count]
summary$n_seeds <- 5L
labels <- c(
  omnit2i = "OmniT2I (ours)", deepsurv = "DeepSurv",
  enet_cox = "Elastic-net Cox", lasso_cox = "Lasso-Cox",
  xgboost_cox = "XGBoost-Cox", rsf = "RSF", survival_gbm = "Survival GBM",
  extra_survival_trees = "ExtraTrees", coxboost = "CoxBoost",
  survival_svm = "Survival SVM", adapted_multisurv = "MultiSurv (adapted)",
  adapted_senmo = "SeNMo (adapted)", adapted_bib_gated = "BIB-style (adapted)"
)
all_labels <- labels
labels <- labels[!grepl("^adapted_", names(labels))]
summary$model <- factor(summary$model, levels = names(labels))
summary$scope <- factor(summary$scope, levels = c("All patients", "Within cancer"))
summary <- summary[order(summary$model, summary$scope, summary$observed_count), ]
long$comparison <- "Universal OmniT2I vs subset-specific specialists (Figure 4b)"
summary$comparison <- "Universal OmniT2I vs subset-specific specialists (Figure 4b)"

## 3. Draw the original muted categorical trajectories.
# Muted categorical colours; all models use solid connecting lines.
colours <- c("#70578F", "#4F748B", "#75876D", "#A4AA83", "#B28A62",
             "#719B9D", "#9A8B73", "#9D9DA6", "#B5A7A3", "#7F7F7F",
             "#8D5364", "#BB8994", "#4B9187")
names(colours) <- names(all_labels)
colours <- colours[names(labels)]
p <- ggplot(summary, aes(observed_count, mean, colour = model,
                         group = model)) +
  geom_line(linewidth = 0.43) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                width = 0.08, linewidth = 0.25, linetype = "solid",
                show.legend = FALSE) +
  geom_point(size = 1.3) +
  geom_line(data = summary[summary$model == "omnit2i", ],
            linewidth = 0.85, show.legend = FALSE) +
  geom_point(data = summary[summary$model == "omnit2i", ],
             size = 2, show.legend = FALSE) +
  facet_wrap(~scope, nrow = 1, scales = "free_y",
             labeller = as_labeller(c("All patients" = "All patients", "Within cancer" = "Within-cancer"))) +
  scale_colour_manual(values = colours, breaks = names(labels), labels = labels) +
  scale_x_continuous(breaks = 1:5, expand = expansion(mult = c(0.04, 0.04))) +
  scale_y_continuous(breaks = seq(0, 1, 0.05),
                     labels = function(x) sprintf("%.2f", x),
                     expand = expansion(mult = c(0.07, 0.09))) +
  labs(title = "Universal OmniT2I versus subset-specific specialists",
       x = "Number of available modalities", y = "Mean test C-index",
       colour = NULL) +
  guides(colour = guide_legend(ncol = 1, byrow = TRUE,
    override.aes = list(linewidth = 1.25, size = 2.5, alpha = 1))) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(text = element_text(colour = "black"),
        aspect.ratio = 1,
        axis.text = element_text(size = 7, colour = "black"),
        axis.line = element_line(linewidth = 0.35),
        axis.ticks = element_line(linewidth = 0.3),
        strip.background = element_blank(),
        strip.text = element_text(size = 8, face = "bold"),
        panel.spacing = grid::unit(5, "mm"),
        plot.title = element_text(size = 9.5, face = "bold", hjust = 0.5,
                                  margin = margin(b = 6)),
        legend.position = "right", legend.text = element_text(size = 7),
        legend.key.width = grid::unit(7, "mm"),
        legend.key.height = grid::unit(3.5, "mm"),
        plot.margin = margin(5, 5, 4, 4))


common_text <- theme(plot.title = element_text(size = 11), strip.text = element_text(size = 9),
 axis.text.x = element_text(size = 7.5), axis.text.y = element_text(size = 7.5),
 axis.title.x = element_text(size = 9), axis.title.y = element_text(size = 9), legend.text = element_text(size = 8))

p <- p + common_text
ggsave("Plots/Figure4/Figure4_panel_b.png", p, width = 90*3922127/1768307, height = 90, units = "mm", dpi = 300, bg = "white")
ggsave("Plots/Figure4/Figure4_panel_b.pdf", p, width = 90*3922127/1768307, height = 90, units = "mm", dpi = 300, bg = "white")

saveRDS(p, "Plots/Figure4/panel_b_plot.rds")
