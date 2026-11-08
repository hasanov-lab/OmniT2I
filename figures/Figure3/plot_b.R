setwd(".")

library(ggplot2)
dir.create("Plots/Figure3", recursive=TRUE, showWarnings=FALSE)

## 1. Load verified current-model ROC coordinates and AUC intervals
fig3b_roc_data <- read.delim("Source_Data/Figure3/Fig3b_time_dependent_ROC_source_data.tsv")
fig3b_auc_summary <- read.delim("Source_Data/Figure3/Fig3b_time_dependent_AUC_95CI.tsv")
fig3b_roc_data$horizon <- factor(fig3b_roc_data$horizon, levels=c("1 year","3 years","5 years"))
fig3b_auc_summary$horizon <- factor(fig3b_auc_summary$horizon, levels=c("1 year","3 years","5 years"))
fig3b_roc_data$cohort <- factor(fig3b_roc_data$cohort, levels=c("Natural availability","Complete cases"))
fig3b_auc_summary$cohort <- factor(fig3b_auc_summary$cohort, levels=c("Natural availability","Complete cases"))

## 9. Make Fig. 3b
horizon_colours <- c(
  "1 year" = "#4477AA",
  "3 years" = "#EE7733",
  "5 years" = "#7A5195"
)

fig3b_auc_labels <- fig3b_auc_summary
fig3b_auc_labels$label <- sprintf(
  "%s: AUC = %.3f (%.3f-%.3f)",
  fig3b_auc_labels$horizon,
  fig3b_auc_labels$estimate,
  fig3b_auc_labels$lower_95ci,
  fig3b_auc_labels$upper_95ci
)
fig3b_auc_labels$label_x <- 0.97
fig3b_auc_labels$label_y <- c(
  "1 year" = 0.23,
  "3 years" = 0.15,
  "5 years" = 0.07
)[as.character(fig3b_auc_labels$horizon)]

fig3b_plot <- ggplot(
  fig3b_roc_data,
  aes(
    x = false_positive_rate,
    y = true_positive_rate,
    colour = horizon,
    group = horizon
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linewidth = 0.45,
    linetype = "dashed",
    colour = "#A7ADB4"
  ) +
  geom_line(linewidth = 0.85) +
  geom_text(
    data = fig3b_auc_labels,
    aes(
      x = label_x,
      y = label_y,
      label = label,
      colour = horizon
    ),
    inherit.aes = FALSE,
    hjust = 1,
    size = 2.45,
    show.legend = FALSE
  ) +
  facet_wrap(
    ~ cohort,
    nrow = 1
  ) +
  scale_colour_manual(
    values = horizon_colours,
    breaks = names(horizon_colours),
    name = NULL
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.25),
    labels = sprintf("%.2f", seq(0, 1, by = 0.25)),
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.25),
    labels = sprintf("%.2f", seq(0, 1, by = 0.25)),
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_equal() +
  labs(
    title = "Time-dependent prognostic performance",
    x = "1 - specificity",
    y = "Sensitivity"
  ) +
  theme_classic(base_size = 8.5, base_family = "Helvetica") +
  theme(
    text = element_text(colour = "#000000"),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.key = element_blank(),
    legend.key.width = grid::unit(5, "mm"),
    legend.key.height = grid::unit(2.5, "mm"),
    legend.text = element_text(size = 7, colour = "#000000"),
    legend.margin = margin(t = -1, r = 0, b = 6, l = 0),
    legend.box.margin = margin(0, 0, 0, 0),
    legend.box.spacing = grid::unit(0, "pt"),
    axis.line = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks = element_line(linewidth = 0.35, colour = "black"),
    axis.text = element_text(size = 7.5, colour = "#000000"),
    axis.title = element_text(size = 8, colour = "#000000"),
    axis.title.x = element_text(margin = margin(t = 5)),
    axis.title.y = element_text(margin = margin(r = 5)),
    strip.background = element_blank(),
    strip.text = element_text(
      size = 8,
      face = "bold",
      margin = margin(t = 0, b = 1)
    ),
    panel.spacing = grid::unit(7, "mm"),
    plot.title = element_text(
      size = 8.5,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 2)
    ),
    plot.margin = margin(t = 1, r = 1, b = 0, l = 1)
  )




## 11. Save Fig. 3b
ggsave(
  filename = "Plots/Figure3/Figure3_panel_b.png",
  plot = fig3b_plot,
  width = 15.8,
  height = 7.2,
  units = "cm",
  dpi = 300
)

ggsave(
  filename = "Plots/Figure3/Figure3_panel_b.pdf",
  plot = fig3b_plot,
  width = 15.8,
  height = 7.2,
  units = "cm"
)

saveRDS(fig3b_plot, "Plots/Figure3/panel_b_plot.rds")
