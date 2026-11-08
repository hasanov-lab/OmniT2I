setwd(".")

library(ggplot2)

## 1. Load the updated source tables and preserve original encodings.

purple <- "#76569A"
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
        margin = margin(b = 2)
      ),
      panel.grid = element_blank(),
      plot.margin = margin(4, 5, 4, 5)
    )
}

route_summary <- read.delim(
  "Source_Data/Figure5/matched_routes.tsv",
  check.names = FALSE
)
## 2. Prepare the exact values shown.
plot_data <- data.frame(
  measured_gain = route_summary$measured_gain_pooled_c_index_mean,
  virtual_gain = route_summary$virtual_gain_pooled_c_index_mean,
  source_subset = route_summary$source_subset,
  virtual_targets = route_summary$virtual_targets,
  observed_count = route_summary$observed_count,
  virtual_count = route_summary$virtual_count
)
stopifnot(nrow(plot_data) == 180)

overview <- read.delim("Source_Data/Figure5/overview.tsv")
o <- overview[overview$metric == "pooled_c_index", ]
stopifnot(nrow(o) == 1, o$routes == 180)
## 3. Draw the original panel type and palette.
panel <- ggplot(plot_data, aes(measured_gain, virtual_gain)) +
  geom_hline(yintercept = 0, linewidth = 0.35, colour = "#9AA0A6") +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "#9AA0A6") +
  geom_abline(
    intercept = 0, slope = 1, linetype = "dashed",
    linewidth = 0.45, colour = "#9AA0A6"
  ) +
  geom_point(
    shape = 21, fill = "#B9A5D0", colour = purple,
    size = 1.55, stroke = 0.35, alpha = 0.65
  ) +
  scale_x_continuous(
    limits = c(-0.01, 0.115), breaks = seq(0, 0.10, 0.02),
    labels = function(x) sprintf("%+.2f", x)
  ) +
  scale_y_continuous(
    limits = c(-0.005, 0.015), breaks = seq(-0.005, 0.015, 0.005),
    labels = function(x) sprintf("%+.3f", x)
  ) +
  labs(
    title = "Predicted versus measured gains: all patients",
    x = "C-index gain with measured modalities",
    y = "C-index gain with predicted modalities"
  ) +
  theme_fig5() +
  theme(legend.position = "none")

## 4. Export the standalone panel, without panel letters.
ggsave(
  "Plots/Figure5/Figure5_panel_d.png",
  panel, width = 90, height = 78, units = "mm", dpi = 300, bg = "white"
)
ggsave(
  "Plots/Figure5/Figure5_panel_d.pdf",
  panel, width = 90, height = 78, units = "mm",
  device = grDevices::pdf, family = "Helvetica", useDingbats = FALSE
)

saveRDS(panel, "Plots/Figure5/panel_d_plot.rds")
