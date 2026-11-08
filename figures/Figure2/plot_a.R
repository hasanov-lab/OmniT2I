setwd(".")


# Load package ------------------------------------------------------------

library(ggplot2)


# Import data -------------------------------------------------------------

modality_data <- read.delim("Source_Data/Figure2/coverage.tsv", check.names = FALSE)
modality_data$colour <- c("#CED0EF", "#B4BBE6", "#9FABDE", "#7CADC1", "#9BB9C3")
modality_data$angle <- c(90, 18, -54, -126, 162)
counts <- read.delim("Source_Data/Figure2/modality_counts.tsv")
total_patients <- sum(counts$n)
complete_cases <- counts$n[counts$observed_modalities == 5]
modality_data$label <- sprintf(
  "%s\n%s (%.1f%%)",
  modality_data$modality,
  format(modality_data$n, big.mark = ","),
  modality_data$percent
)

# Build flower geometry --------------------------------------------------

make_ellipse <- function(centre_x, centre_y, angle_degrees, modality) {
  theta <- seq(0, 2 * pi, length.out = 240)
  angle_radians <- angle_degrees * pi / 180
  x_base <- 1.48 * cos(theta)
  y_base <- 0.94 * sin(theta)

  data.frame(
    x = centre_x +
      x_base * cos(angle_radians) - y_base * sin(angle_radians),
    y = centre_y +
      x_base * sin(angle_radians) + y_base * cos(angle_radians),
    modality = modality
  )
}

petal_data <- do.call(
  rbind,
  lapply(seq_len(nrow(modality_data)), function(index) {
    angle_radians <- modality_data$angle[index] * pi / 180
    make_ellipse(
      centre_x = 1.05 * cos(angle_radians),
      centre_y = 1.05 * sin(angle_radians),
      angle_degrees = modality_data$angle[index],
      modality = modality_data$modality[index]
    )
  })
)

modality_data$label_radius <- ifelse(
  modality_data$modality %in% c("CNV", "H&E"),
  3.58,
  3.18
)
modality_data$arrow_radius <- 2.75

modality_data$label_x <- modality_data$label_radius *
  cos(modality_data$angle * pi / 180)
modality_data$label_y <- modality_data$label_radius *
  sin(modality_data$angle * pi / 180)
modality_data$arrow_start_x <- modality_data$arrow_radius *
  cos(modality_data$angle * pi / 180)
modality_data$arrow_start_y <- modality_data$arrow_radius *
  sin(modality_data$angle * pi / 180)
modality_data$arrow_end_x <- 1.38 * cos(modality_data$angle * pi / 180)
modality_data$arrow_end_y <- 1.38 * sin(modality_data$angle * pi / 180)

circle_angle <- seq(0, 2 * pi, length.out = 240)
centre_circle <- data.frame(
  x = 0.72 * cos(circle_angle),
  y = 0.72 * sin(circle_angle)
)

modality_colours <- setNames(modality_data$colour, modality_data$modality)


# Draw flower -------------------------------------------------------------

flower_plot <- ggplot() +
  geom_polygon(
    data = petal_data,
    aes(x = x, y = y, group = modality, fill = modality),
    colour = NA,
    alpha = 0.74
  ) +
  geom_polygon(
    data = centre_circle,
    aes(x = x, y = y),
    fill = "#E6EAF3",
    colour = NA
  ) +
  geom_segment(
    data = modality_data,
    aes(
      x = arrow_end_x,
      y = arrow_end_y,
      xend = arrow_start_x,
      yend = arrow_start_y
    ),
    linewidth = 0.45,
    colour = "black",
    arrow = arrow(length = unit(1.35, "mm"), type = "closed")
  ) +
  annotate(
    "text",
    x = 0,
    y = 0.12,
    label = "Complete cases",
    fontface = "bold",
    size = 2.7,
    colour = "black"
  ) +
  annotate(
    "text",
    x = 0,
    y = -0.17,
    label = sprintf(
      "%s (%.1f%%)",
      format(complete_cases, big.mark = ","),
      100 * complete_cases / total_patients
    ),
    fontface = "bold",
    size = 2.55,
    colour = "black"
  ) +
  geom_text(
    data = modality_data,
    aes(x = label_x, y = label_y, label = label),
    fontface = "bold",
    size = 2.6,
    hjust = 0.5,
    lineheight = 0.93,
    colour = "black"
  ) +
  scale_fill_manual(values = modality_colours) +
  coord_fixed(xlim = c(-3.90, 3.90), ylim = c(-3.34, 3.52), clip = "off") +
  labs(
    title = sprintf(
      "Modality availability (n = %s)",
      format(total_patients, big.mark = ",")
    )
  ) +
  theme_void(base_size = 8, base_family = "Helvetica") +
  theme(
    legend.position = "none",
    plot.title = element_text(
      size = 9,
      face = "bold",
      hjust = 0.5,
      colour = "black",
      margin = margin(b = -5)
    ),
    plot.margin = margin(0.5, 1, 0.5, 1)
  )

flower_plot


# Export ------------------------------------------------------------------

ggsave(
  "Plots/Figure2/panel_a.png",
  flower_plot,
  width = 10.5,
  height = 7.8,
  units = "cm",
  dpi = 300,
  bg = "white"
)

ggsave(
  "Plots/Figure2/panel_a.pdf",
  flower_plot,
  width = 10.5,
  height = 7.8,
  units = "cm",
  bg = "white"
)

saveRDS(flower_plot, "Plots/Figure2/panel_a_plot.rds")
