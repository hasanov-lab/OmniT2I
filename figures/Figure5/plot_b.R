setwd(".")

library(ggplot2)

## 1. Load the updated source tables and preserve original encodings.

target_levels <- c("rna", "cnv", "protein", "methylation", "he")
target_labels <- c(
  rna = "RNA", cnv = "CNV", protein = "Protein",
  methylation = "Meth", he = "H&E embedding"
)
purple <- "#76569A"
grey <- "#7C8792"
dark <- "#2F3542"

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
    target_label = target_summary$target_label,
    input = "Observed modalities only",
    c_index = target_summary$source_only_pooled_c_index_mean,
    sd = target_summary$source_only_pooled_c_index_sd
  ),
  data.frame(
    target_label = target_summary$target_label,
    input = "With predicted modality",
    c_index = target_summary$virtual_pooled_c_index_mean,
    sd = target_summary$virtual_pooled_c_index_sd
  ),
  data.frame(
    target_label = target_summary$target_label,
    input = "With measured modality",
    c_index = target_summary$measured_matched_pooled_c_index_mean,
    sd = target_summary$measured_matched_pooled_c_index_sd
  )
)
plot_data$input <- factor(
  plot_data$input,
  levels = c("Observed modalities only", "With predicted modality", "With measured modality")
)
plot_data$target_y <- as.numeric(plot_data$target_label)
plot_data$target_y_plot <- plot_data$target_y + unname(c(
  "Observed modalities only" = -0.10,
  "With predicted modality" = 0,
  "With measured modality" = 0.10
)[as.character(plot_data$input)])

## 3. Draw the original panel type and palette.
panel <- ggplot(plot_data, aes(c_index, target_y_plot, colour = input)) +
  geom_path(aes(group = target_label), colour = "#D9DCE1", linewidth = 0.9) +
  geom_errorbar(
    aes(xmin = c_index - sd, xmax = c_index + sd),
    width = 0.05, linewidth = 0.38, orientation = "y"
  ) +
  geom_point(size = 2.0) +
  scale_colour_manual(
    values = c(
      "Observed modalities only" = grey,
      "With predicted modality" = purple,
      "With measured modality" = dark
    ),
    labels = c("Observed inputs", "+ predicted", "+ measured")
  ) +
  scale_x_continuous(
    limits = c(0.748, 0.786), breaks = seq(0.75, 0.78, 0.01),
    labels = function(x) sprintf("%.2f", x)
  ) +
  scale_y_continuous(
    breaks = seq_along(levels(plot_data$target_label)),
    labels = levels(plot_data$target_label),
    expand = expansion(add = 0.35)
  ) +
  labs(
    title = "Survival prediction in all patients",
    x = "Mean C-index (15 subsets)", y = "Added modality"
  ) +
  theme_fig5() +
  theme(
    legend.key.width = grid::unit(1.5, "mm"),
    legend.spacing.x = grid::unit(0.3, "mm")
  )

## 4. Export the standalone panel, without panel letters.
ggsave(
  "Plots/Figure5/Figure5_panel_b.png",
  panel, width = 78, height = 72, units = "mm", dpi = 300, bg = "white"
)
ggsave(
  "Plots/Figure5/Figure5_panel_b.pdf",
  panel, width = 78, height = 72, units = "mm",
  device = grDevices::pdf, family = "Helvetica", useDingbats = FALSE
)

saveRDS(panel, "Plots/Figure5/panel_b_plot.rds")
