setwd(".")

## Required package
library(ggplot2)


## 1. Load the patient-bootstrap estimates and 95% confidence intervals
fig3a_summary <- read.delim(
  "Source_Data/Figure3/Fig3a_C_index_95CI.tsv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

fig3a_summary$metric <- factor(
  fig3a_summary$metric,
  levels = c("All patients", "Within cancer")
)

fig3a_summary$cohort <- factor(
  fig3a_summary$cohort,
  levels = c("Natural availability", "Complete cases")
)

fig3a_summary$summary_label <- sprintf(
  "%.3f",
  fig3a_summary$estimate
)

## 2. Make Fig. 3a
bar_position <- position_dodge(width = 0.65)

fig3a_plot <- ggplot(
  fig3a_summary,
  aes(x = metric, y = estimate, fill = cohort)
) +
  geom_col(
    width = 0.56,
    position = bar_position,
    colour = "white",
    linewidth = 0.25
  ) +
  geom_errorbar(
    aes(ymin = lower_95ci, ymax = upper_95ci),
    width = 0.10,
    position = bar_position,
    linewidth = 0.3,
    colour = "black"
  ) +
  geom_text(
    aes(
      y = upper_95ci + 0.010,
      label = summary_label
    ),
    position = bar_position,
    vjust = 0,
    size = 2.35,
    lineheight = 1.00,
    colour = "#000000"
  ) +
  scale_fill_manual(
    values = c(
      "Natural availability" = "#B6A2BD",
      "Complete cases" = "#665F6D"
    )
  ) +
  scale_y_continuous(
    breaks = seq(0, 0.8, by = 0.2),
    labels = sprintf("%.1f", seq(0, 0.8, by = 0.2)),
    limits = c(0, 0.92),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Survival prediction performance",
    x = "Evaluation scope",
    y = "C-index",
    fill = NULL
  ) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(
    text = element_text(colour = "#000000"),
    legend.position = "top",
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.key.width = grid::unit(3.5, "mm"),
    legend.key.height = grid::unit(2.5, "mm"),
    legend.text = element_text(size = 6.5, colour = "#000000"),
    legend.margin = margin(t = 0, b = 6),
    legend.box.spacing = grid::unit(0, "pt"),
    axis.line = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks = element_line(linewidth = 0.35, colour = "black"),
    axis.text = element_text(size = 7, colour = "#000000"),
    axis.title.x = element_text(size = 7.5, margin = margin(t = 5)),
    axis.title.y = element_text(size = 7.5, margin = margin(r = 6)),
    plot.title = element_text(size = 8, face = "bold", hjust = 0.5, margin = margin(b = 2)),
    plot.margin = margin(t = 3, r = 4, b = 3, l = 3)
  )


## 3. Save Fig. 3a
ggsave(
  filename = "Plots/Figure3/Figure3_panel_a.png",
  plot = fig3a_plot,
  width = 6.2,
  height = 6.1,
  units = "cm",
  dpi = 300
)

ggsave(
  filename = "Plots/Figure3/Figure3_panel_a.pdf",
  plot = fig3a_plot,
  width = 6.2,
  height = 6.1,
  units = "cm"
)

saveRDS(fig3a_plot, "Plots/Figure3/panel_a_plot.rds")
