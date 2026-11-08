setwd(".")

library(ggplot2)
dir.create("Plots/Figure3", recursive=TRUE, showWarnings=FALSE)
library(patchwork)

## 1. Load the current, verified KM curves and cancer-stratified association
km_curve <- read.delim("Source_Data/Figure3/Fig3c_km_source_data.tsv")
risk_table <- read.delim("Source_Data/Figure3/Fig3c_risk_table_source_data.tsv")
statistics <- read.delim("Source_Data/Figure3/Fig3c_statistics.tsv")
risk_times <- c(0,2,4,6,8,10)
risk_table$risk_group <- factor(risk_table$risk_group, levels=c("Higher risk","Lower risk"))
p_label <- if (statistics$stratified_logrank_p < 0.001) "p < 0.001" else sprintf("p = %.3f", statistics$stratified_logrank_p)
hr_label <- sprintf("HR = %.2f (%.2f-%.2f)", statistics$hazard_ratio,
 statistics$hazard_ratio_lower_95, statistics$hazard_ratio_upper_95)

## 2. Restore the original curves, pointwise bands, colours and risk table
## 7. Make Fig. 3c
risk_colours <- c(
  "Lower risk" = "#377EB8",
  "Higher risk" = "#E41A1C"
)

km_plot <- ggplot(
  km_curve,
  aes(
    x = time_years,
    y = survival,
    colour = risk_group
  )
) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper, fill = risk_group),
    colour = NA,
    alpha = 0.18,
    show.legend = FALSE
  ) +
  geom_step(linewidth = 0.60) +
  annotate(
    "text",
    x = 6.40,
    y = 0.95,
    label = paste(p_label, hr_label, sep = "\n"),
    hjust = 0,
    vjust = 1,
    lineheight = 1.00,
    size = 2.60,
    colour = "#000000"
  ) +
  scale_colour_manual(values = risk_colours, name = NULL) +
  scale_fill_manual(values = risk_colours, name = NULL) +
  scale_x_continuous(
    breaks = risk_times,
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_cartesian(xlim = c(0, 10), ylim = c(0, 1)) +
  labs(
    title = "Risk stratification under natural availability",
    x = "Years",
    y = "Survival probability"
  ) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(
    text = element_text(colour = "#000000"),
    legend.position = "none",
    axis.line = element_line(linewidth = 0.35, colour = "black"),
    axis.ticks = element_line(linewidth = 0.35, colour = "black"),
    axis.text = element_text(size = 7, colour = "#000000"),
    axis.title.x = element_text(size = 7.5, margin = margin(t = 5)),
    axis.title.y = element_text(size = 7.5, margin = margin(r = 6)),
    plot.title = element_text(
      size = 8,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 5)
    ),
    plot.margin = margin(t = 3, r = 4, b = 0, l = 3)
  )

risk_table_plot <- ggplot(
  risk_table,
  aes(
    x = time_years,
    y = risk_group,
    label = n_at_risk,
    colour = risk_group,
    hjust = ifelse(time_years == 0, 0, ifelse(time_years == 10, 1, 0.5))
  )
) +
  geom_text(size = 2.35) +
  scale_colour_manual(values = risk_colours, guide = "none") +
  scale_x_continuous(
    breaks = risk_times,
    limits = c(0, 10),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_discrete(
    drop = FALSE,
    labels = c("Higher risk" = "High", "Lower risk" = "Low")
  ) +
  labs(title = "Number at risk", x = "Years", y = NULL) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(
    text = element_text(colour = "#000000"),
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 7, colour = "#000000"),
    axis.text.x = element_text(size = 7, colour = "#000000"),
    axis.title.x = element_text(size = 7.5, margin = margin(t = 5)),
    plot.title = element_text(size = 7.5, face = "bold", hjust = 0, margin = margin(b = 1)),
    plot.margin = margin(t = 2, r = 4, b = 2, l = 3)
  )

fig3c_plot <- km_plot / risk_table_plot +
  plot_layout(heights = c(3.55, 0.95))





## 9. Save Fig. 3c
ggsave(
  filename = "Plots/Figure3/Figure3_panel_c.png",
  plot = fig3c_plot,
  width = 9.0,
  height = 7.5,
  units = "cm",
  dpi = 300
)

ggsave(
  filename = "Plots/Figure3/Figure3_panel_c.pdf",
  plot = fig3c_plot,
  width = 9.0,
  height = 7.5,
  units = "cm"
)

saveRDS(fig3c_plot, "Plots/Figure3/panel_c_plot.rds")
