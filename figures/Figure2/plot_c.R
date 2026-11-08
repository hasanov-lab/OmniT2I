setwd(".")


## Required packages
library(ggplot2)
library(patchwork)


## 1. Load data
modality_labels <- c("RNA", "CNV", "Protein", "Meth", "H&E")
fig2c_data <- read.delim("Source_Data/Figure2/Fig2c_source_data.tsv", check.names = FALSE)
combination_matrix <- as.matrix(fig2c_data[, modality_labels]) > 0
# Use the ranked combination number as the shared x axis of the two plots
fig2c_data$combination <- factor(fig2c_data$combination_id,
                                 levels = fig2c_data$combination_id)

# A log10 scale keeps both common and rare combinations visible
fig2c_data$log10_patients <- log10(fig2c_data$patients)
fig2c_data$count_label <- format(fig2c_data$patients,
                                 big.mark = ",",
                                 trim = TRUE)

# Highlight only the all-five-modality complete-case combination
fig2c_data$bar_group <- ifelse(
  fig2c_data$availability == "rna+cnv+rppa+methylation+he",
  "Complete cases",
  "Other combinations"
)


## 3. Source data are unchanged; do not rewrite the original table.


## 4. Prepare the UpSet combination matrix
modality_positions <- c("RNA" = 5,
                        "CNV" = 4,
                        "Protein" = 3,
                        "Meth" = 2,
                        "H&E" = 1)

matrix_data <- data.frame(
  combination = rep(fig2c_data$combination, each = length(modality_labels)),
  modality = rep(modality_labels, times = nrow(fig2c_data)),
  modality_position = rep(modality_positions, times = nrow(fig2c_data)),
  observed = as.logical(
    as.vector(t(as.matrix(fig2c_data[,modality_labels])))
  )
)

# The vertical line joins the observed modalities within each combination
connector_data <- data.frame(
  combination = fig2c_data$combination,
  first_observed = apply(combination_matrix, 1, function(row_values) {
    min(modality_positions[row_values])
  }),
  last_observed = apply(combination_matrix, 1, function(row_values) {
    max(modality_positions[row_values])
  })
)


## 5. Make the upper frequency plot
frequency_plot <- ggplot(fig2c_data,
                         aes(x = combination,
                             y = log10_patients)) +
  geom_col(aes(fill = bar_group),
           width = 0.68) +
  geom_text(aes(y = log10_patients + 0.15,
                label = count_label),
            size = 2.15,
            colour = "black") +
  scale_x_discrete(expand = expansion(add = c(0.45, 0.45))) +
  scale_fill_manual(values = c(
    "Complete cases" = "#66528F",
    "Other combinations" = "#9B8FBA"
  )) +
  scale_y_continuous(breaks = 0:4,
                     labels = c("1", "10", "100", "1,000", "10,000"),
                     limits = c(0, 4.15),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "Patients by observed modality combination",
       x = NULL,
       y = "Patients (log10 scale)") +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(legend.position = "none",
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.line.y = element_line(linewidth = 0.35, colour = "black"),
        axis.ticks.y = element_line(linewidth = 0.35, colour = "black"),
        axis.text.y = element_text(size = 7, colour = "black"),
        axis.title.y = element_text(size = 7.5,
                                    colour = "black",
                                    margin = margin(r = 7)),
        plot.title = element_text(size = 8,
                                  face = "bold",
                                  hjust = 0.5,
                                  colour = "black",
                                  margin = margin(b = 8)),
        plot.margin = margin(t = 2, r = 2, b = 0, l = 2))


## 6. Make the lower combination matrix
combination_matrix_plot <- ggplot(matrix_data,
                                  aes(x = combination,
                                      y = modality_position)) +
  geom_segment(data = connector_data,
               aes(x = combination,
                   xend = combination,
                   y = first_observed,
               yend = last_observed),
               linewidth = 0.65,
               colour = "#536978") +
  geom_point(size = 2.15,
             colour = "#D9DEE3") +
  geom_point(data = matrix_data[matrix_data$observed,],
             size = 2.45,
             colour = "#536978") +
  scale_x_discrete(expand = expansion(add = c(0.45, 0.45))) +
  scale_y_continuous(breaks = modality_positions,
                     labels = c("RNA", "CNV", "Protein", "Meth", "H&E"),
                     limits = c(0.5, 5.5),
                     expand = expansion(mult = c(0, 0))) +
  labs(x = NULL,
       y = NULL) +
  theme_void(base_size = 8, base_family = "Helvetica") +
  theme(axis.text.y = element_text(size = 7,
                                   colour = "black",
                                   hjust = 1,
                                   margin = margin(r = 6)),
        plot.margin = margin(t = 0, r = 2, b = 2, l = 2))


## 7. Assemble Fig. 2c
fig2c_plot <- frequency_plot / combination_matrix_plot +
  plot_layout(heights = c(1.75, 1))


## 8. Preview Fig. 2c in the RStudio Plots window
fig2c_plot


## 9. Save Fig. 2c
### 9.1 Save PNG at 300 dpi
ggsave(filename = "Plots/Figure2/panel_c.png",
       plot = fig2c_plot,
       width = 15,
       height = 7,
       units = "cm",
       dpi = 300)

### 9.2 Save PDF
ggsave(filename = "Plots/Figure2/panel_c.pdf",
       plot = fig2c_plot,
       width = 15,
       height = 7,
       units = "cm",
       device = cairo_pdf)

saveRDS(fig2c_plot, "Plots/Figure2/panel_c_plot.rds")
