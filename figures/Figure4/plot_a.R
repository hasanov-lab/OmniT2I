setwd(".")

library(ggplot2)


## 1. Load paired specialist and universal-model results
input_file <- "Source_Data/Figure4/paired_per_seed.tsv"
output_directory <- "Plots/Figure4/"
source_data_directory <- "Source_Data/Figure4/"
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
all_data <- read.delim(input_file)
stopifnot(nrow(all_data) == 9 * 31 * 5,
          !anyDuplicated(all_data[c("model", "seed", "source_subset")]))


## 2. Calculate each specialist's five-seed mean for every modality combination
subset_means <- aggregate(
  cbind(delta_pooled_c_index, delta_within_cancer_c_index) ~
    model + source_subset + observed_count,
  data = all_data,
  FUN = mean
)

model_labels <- c(
  "deepsurv" = "DeepSurv",
  "enet_cox" = "Elastic-net Cox",
  "lasso_cox" = "Lasso-Cox",
  "xgboost_cox" = "XGBoost-Cox",
  "rsf" = "RSF",
  "survival_gbm" = "Survival GBM",
  "extra_survival_trees" = "ExtraTrees",
  "coxboost" = "CoxBoost",
  "survival_svm" = "Survival SVM",
  "adapted_multisurv" = "MultiSurv (adapted)",
  "adapted_senmo" = "SeNMo (adapted)",
  "adapted_bib_gated" = "BIB-style (adapted)"
)

model_labels <- model_labels[!grepl("^adapted_", names(model_labels))]
model_order <- unname(model_labels)

plot_data <- rbind(
  data.frame(
    model = subset_means$model,
    source_subset = subset_means$source_subset,
    observed_count = subset_means$observed_count,
    metric_label = "All patients",
    delta_c_index = -subset_means$delta_pooled_c_index
  ),
  data.frame(
    model = subset_means$model,
    source_subset = subset_means$source_subset,
    observed_count = subset_means$observed_count,
    metric_label = "Within cancer",
    delta_c_index = -subset_means$delta_within_cancer_c_index
  )
)

plot_data$model_label <- factor(
  unname(model_labels[plot_data$model]),
  levels = rev(model_order)
)

plot_data$metric_label <- factor(
  plot_data$metric_label,
  levels = c("All patients", "Within cancer")
)

stopifnot(
  nrow(plot_data) == 9 * 31 * 2,
  !anyNA(plot_data$model_label),
  all(table(plot_data$model, plot_data$metric_label) == 31)
)


## 3. Export the exact values shown in the panel
source_data <- plot_data[, c(
  "model",
  "model_label",
  "source_subset",
  "observed_count",
  "metric_label",
  "delta_c_index"
)]

source_data$difference_definition <- paste0(
  "Five-seed mean specialist C-index minus universal OmniT2I C-index"
)
source_data$n_patients <- 5605L
source_data$n_events <- 1546L



## 4. Draw the 31-combination distributions for all specialist families
metric_colours <- c(
  "All patients" = "#7893A3",
  "Within cancer" = "#A56C77"
)

# Retain the previous range, extending it when added models fall outside it.
x_limits <- c(min(-0.21, floor(min(plot_data$delta_c_index) * 100) / 100),
              max(0.03, ceiling(max(plot_data$delta_c_index) * 100) / 100))

fig4a_boxplot <- ggplot(
  plot_data,
  aes(x = delta_c_index, y = model_label, colour = metric_label)
) +
  geom_vline(
    xintercept = 0,
    linewidth = 0.4,
    linetype = "dashed",
    colour = "#555555"
  ) +
  geom_boxplot(
    width = 0.42,
    outlier.shape = NA,
    fill = "white",
    linewidth = 0.35,
    colour = "#555555"
  ) +
  geom_point(
    position = position_jitter(width = 0, height = 0.12, seed = 42),
    size = 1.15,
    alpha = 0.70
  ) +
  facet_wrap(~metric_label, nrow = 1,
             labeller = as_labeller(c("All patients" = "All patients", "Within cancer" = "Within-cancer"))) +
  scale_colour_manual(values = metric_colours, guide = "none") +
  scale_x_continuous(
    limits = x_limits,
    breaks = seq(ceiling(x_limits[1] / 0.05) * 0.05, x_limits[2], 0.05),
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Universal OmniT2I versus subset-specific specialists",
    x = "C-index difference (specialist - OmniT2I)",
    y = NULL
  ) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(
    text = element_text(colour = "black"),
    strip.background = element_blank(),
    strip.text = element_text(size = 7, face = "bold", colour = "black"),
    panel.spacing.x = grid::unit(5, "mm"),
    axis.line = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks = element_line(linewidth = 0.35, colour = "black"),
    axis.text.x = element_text(size = 6.5, colour = "black"),
    axis.text.y = element_text(size = 6.5, colour = "black"),
    axis.title.x = element_text(size = 7.5, margin = margin(t = 5)),
    plot.title = element_text(
      size = 9.5,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 1)
    ),
    plot.margin = margin(t = 5, r = 4, b = 4, l = 3)
  )



common_text <- theme(plot.title = element_text(size = 11), strip.text = element_text(size = 9),
 axis.text.x = element_text(size = 7.5), axis.text.y = element_text(size = 7.5),
 axis.title.x = element_text(size = 9), axis.title.y = element_text(size = 9), legend.text = element_text(size = 8))

fig4a_boxplot <- fig4a_boxplot + common_text
ggsave("Plots/Figure4/Figure4_panel_a.png", fig4a_boxplot, width = 90*2846623/1768224, height = 90, units = "mm", dpi = 300, bg = "white")
ggsave("Plots/Figure4/Figure4_panel_a.pdf", fig4a_boxplot, width = 90*2846623/1768224, height = 90, units = "mm", dpi = 300, bg = "white")

saveRDS(fig4a_boxplot, "Plots/Figure4/panel_a_plot.rds")
