setwd(".")


# Load package ------------------------------------------------------------

library(ggplot2)


# Import data -------------------------------------------------------------

count_data <- read.delim("Source_Data/Figure2/modality_counts.tsv")

# Build donut geometry ---------------------------------------------------

count_data <- count_data[order(count_data$observed_modalities), ]
count_data$fraction <- count_data$n / sum(count_data$n)
count_data$end_angle <- pi / 2 - 2 * pi * cumsum(count_data$fraction)
count_data$start_angle <- c(pi / 2, head(count_data$end_angle, -1))
count_data$mid_angle <- (count_data$start_angle + count_data$end_angle) / 2

make_donut_slice <- function(start_angle, end_angle, group) {
  outer_angle <- seq(start_angle, end_angle, length.out = 100)
  inner_angle <- rev(outer_angle)

  data.frame(
    x = c(cos(outer_angle), 0.55 * cos(inner_angle)),
    y = c(sin(outer_angle), 0.55 * sin(inner_angle)),
    observed_modalities = group
  )
}

donut_data <- do.call(
  rbind,
  lapply(seq_len(nrow(count_data)), function(index) {
    make_donut_slice(
      count_data$start_angle[index],
      count_data$end_angle[index],
      count_data$observed_modalities[index]
    )
  })
)

count_data$leader_x <- 1.03 * cos(count_data$mid_angle)
count_data$leader_y <- 1.03 * sin(count_data$mid_angle)
count_data$label_x <- 1.28 * cos(count_data$mid_angle)
count_data$label_y <- 1.28 * sin(count_data$mid_angle)
count_data$label_hjust <- ifelse(cos(count_data$mid_angle) >= 0, 0, 1)

# Separate the labels of the three smallest adjacent slices
count_data$label_y[count_data$observed_modalities == 1] <- 1.12
count_data$label_y[count_data$observed_modalities == 2] <- 0.80
count_data$label_y[count_data$observed_modalities == 3] <- 0.50
count_data$label_x[count_data$observed_modalities %in% 1:3] <- 1.16

# Extend each leader line to the near edge of its text label
count_data$line_end_x <- count_data$label_x -
  ifelse(count_data$label_hjust == 0, 0.035, -0.035)
count_data$line_end_y <- count_data$label_y

count_colours <- c(
  "1" = "#DED8EA",
  "2" = "#CDC5DE",
  "3" = "#B9ADCF",
  "4" = "#A291BF",
  "5" = "#8976AD"
)


# Draw donut chart -------------------------------------------------------

donut_plot <- ggplot() +
  geom_polygon(
    data = donut_data,
    aes(
      x = x,
      y = y,
      group = observed_modalities,
      fill = factor(observed_modalities)
    ),
    colour = "white",
    linewidth = 0.6
  ) +
  geom_segment(
    data = count_data,
    aes(x = leader_x, y = leader_y, xend = line_end_x, yend = line_end_y),
    linewidth = 0.35,
    colour = "black"
  ) +
  geom_text(
    data = count_data,
    aes(
      x = label_x,
      y = label_y,
      hjust = label_hjust,
      label = sprintf(
        "%d %s\n%s (%.1f%%)",
        observed_modalities,
        ifelse(observed_modalities == 1, "modality", "modalities"),
        format(n, big.mark = ","),
        percent
      )
    ),
    fontface = "bold",
    size = 2.6,
    lineheight = 0.94,
    colour = "black"
  ) +
  annotate(
    "text",
    x = 0,
    y = 0.08,
    label = "Natural\navailability",
    fontface = "bold",
    size = 2.75,
    lineheight = 0.95,
    colour = "black"
  ) +
  annotate(
    "text",
    x = 0,
    y = -0.24,
    label = sprintf("n = %s", format(sum(count_data$n), big.mark = ",")),
    fontface = "bold",
    size = 2.6,
    colour = "black"
  ) +
  scale_fill_manual(values = count_colours) +
  coord_fixed(xlim = c(-1.68, 1.68), ylim = c(-1.25, 1.25), clip = "off") +
  labs(title = "Observed modalities per patient") +
  theme_void(base_size = 8, base_family = "Helvetica") +
  theme(
    legend.position = "none",
    plot.title = element_text(
      size = 9,
      face = "bold",
      hjust = 0.5,
      colour = "black",
      margin = margin(b = -2)
    ),
    plot.margin = margin(0.5, 1, 0.5, 1)
  )

donut_plot


# Export ------------------------------------------------------------------

ggsave(
  "Plots/Figure2/panel_b.png",
  donut_plot,
  width = 10.5,
  height = 7.8,
  units = "cm",
  dpi = 300,
  bg = "white"
)

ggsave(
  "Plots/Figure2/panel_b.pdf",
  donut_plot,
  width = 10.5,
  height = 7.8,
  units = "cm",
  bg = "white"
)

saveRDS(donut_plot, "Plots/Figure2/panel_b_plot.rds")
