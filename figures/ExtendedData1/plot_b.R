setwd(".")

library(ggplot2)

source_file <- "Source_Data/ExtendedData1/EDFig1ab_cancer_specific_time_dependent_AUC_95CI.tsv"
output_directory <- "Plots/ExtendedData1"
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

data <- read.delim(source_file, stringsAsFactors = FALSE, check.names = FALSE)
plot_data <- data[
  data$scenario == "complete_case" & data$status == "evaluable",
]
plot_data$horizon <- factor(plot_data$horizon, levels = c("1 year", "3 years", "5 years"))
plot_data$facet_cancer <- paste(plot_data$cancer_type, plot_data$horizon, sep = "___")
facet_order <- unlist(lapply(levels(plot_data$horizon), function(value) {
  frame <- plot_data[plot_data$horizon == value, ]
  frame$facet_cancer[order(frame$estimate)]
}), use.names = FALSE)
plot_data$facet_cancer <- factor(plot_data$facet_cancer, levels = facet_order)

plot <- ggplot(plot_data, aes(x = estimate, y = facet_cancer)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 0.35, colour = "#9CA3AF") +
  geom_errorbar(
    aes(xmin = lower_95ci, xmax = upper_95ci),
    orientation = "y", width = 0.45, linewidth = 0.35, colour = "#2F3745"
  ) +
  geom_point(size = 2.2, colour = "#7555A5") +
  facet_wrap(~horizon, nrow = 1, scales = "free_y") +
  scale_x_continuous(limits = c(0.25, 1.01), breaks = c(0.3, 0.5, 0.7, 0.9)) +
  scale_y_discrete(labels = function(value) sub("___.*$", "", value)) +
  labs(
    title = "Cancer-specific time-dependent AUC in complete cases",
    x = "Time-dependent AUC",
    y = NULL
  ) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(
    plot.title = element_text(face = "bold", size = 10, hjust = 0.5, margin = margin(b = 6)),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 8.5),
    axis.text.y = element_text(size = 6.5),
    axis.text.x = element_text(size = 7),
    axis.title.x = element_text(size = 8.5, margin = margin(t = 5)),
    panel.spacing.x = unit(0.35, "cm"),
    plot.margin = margin(5, 5, 4, 4)
  )

ggsave(
  file.path(output_directory, "ExtendedData1_panel_b.pdf"),
  plot, width = 17.8, height = 10.8, units = "cm",
  device = grDevices::pdf, family = "Helvetica", useDingbats = FALSE,
  bg = "white"
)
ggsave(
  file.path(output_directory, "ExtendedData1_panel_b.png"),
  plot, width = 17.8, height = 10.8, units = "cm", dpi = 300,
  bg = "white"
)

saveRDS(plot, "Plots/ExtendedData1/panel_b_plot.rds")
