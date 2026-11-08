setwd(".")

library(ggplot2)

## 1. Load the updated source tables and preserve original encodings.

target_levels <- c("rna", "cnv", "protein", "methylation", "he")
target_labels <- c(
  rna = "RNA", cnv = "CNV", protein = "Protein",
  methylation = "Meth", he = "H&E embedding"
)
purple <- "#76569A"
rose <- "#A76578"

theme_fig5 <- function() {
  theme_classic(base_size = 7.5, base_family = "Helvetica") +
    theme(
      text = element_text(colour = "#000000"),
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.text = element_text(colour = "black"),
      plot.title = element_text(
        size = 8, face = "bold", hjust = 0.5,
        margin = margin(b = 1)
      ),
      legend.title = element_blank(),
      legend.position = "top",
      legend.key.width = grid::unit(4, "mm"),
      legend.margin = margin(0),
      legend.box.spacing = grid::unit(1, "mm"),
      panel.grid = element_blank(),
      plot.margin = margin(4, 5, 4, 5)
    )
}

target_summary <- read.delim(
  "Source_Data/Figure5/target_summary.tsv",
  check.names = FALSE
)
target_summary$target <- factor(target_summary$target, levels = rev(target_levels))
target_summary$target_label <- factor(
  unname(target_labels[as.character(target_summary$target)]),
  levels = unname(target_labels[rev(target_levels)])
)

## 2. Prepare the exact values shown.
plot_data <- rbind(
  data.frame(
    target = target_summary$target,
    target_label = target_summary$target_label,
    scope = "All patients",
    pearson_r = target_summary$feature_pearson_mean_fold_weighted_mean,
    sd = target_summary$feature_pearson_mean_fold_weighted_sd
  ),
  data.frame(
    target = target_summary$target,
    target_label = target_summary$target_label,
    scope = "Within cancer",
    pearson_r = target_summary$within_cancer_feature_pearson_mean_fold_weighted_mean,
    sd = target_summary$within_cancer_feature_pearson_mean_fold_weighted_sd
  )
)
plot_data$scope <- factor(plot_data$scope, levels = c("All patients", "Within cancer"))

## 3. Draw the original panel type and palette.
panel <- ggplot(plot_data, aes(pearson_r, target_label)) +
  geom_line(aes(group = target_label), colour = "#D9DCE1", linewidth = 1.2) +
  geom_errorbar(
    aes(xmin = pearson_r - sd, xmax = pearson_r + sd, colour = scope),
    width = 0.10, linewidth = 0.38, orientation = "y"
  ) +
  geom_point(aes(colour = scope), size = 2.1, stroke = 0.55, fill = "white") +
  scale_colour_manual(values = c("All patients" = purple, "Within cancer" = rose),
                      labels = c("All patients" = "All patients", "Within cancer" = "Within-cancer")) +
  scale_x_continuous(limits = c(0, 0.8), breaks = seq(0, 0.8, 0.2)) +
  labs(
    title = "Correlation between predicted and\nmeasured modalities",
    x = "Mean Pearson r across features",
    y = NULL
  ) +
  theme_fig5()

## 4. Export the standalone panel, without panel letters.
ggsave(
  "Plots/Figure5/Figure5_panel_a.png",
  panel, width = 78, height = 72, units = "mm", dpi = 300, bg = "white"
)
ggsave(
  "Plots/Figure5/Figure5_panel_a.pdf",
  panel, width = 78, height = 72, units = "mm",
  device = grDevices::pdf, family = "Helvetica", useDingbats = FALSE
)

saveRDS(panel, "Plots/Figure5/panel_a_plot.rds")
