setwd(".")

library(ggplot2)

input_file <- "Source_Data/ExtendedData3/route_gains.tsv"
output_directory <- "Plots/ExtendedData3"

dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

routes <- read.delim(input_file, check.names = FALSE)
stopifnot(
  nrow(routes) == 360,
  all(table(routes$evaluation) == 180),
  !anyDuplicated(routes[c("evaluation", "source_subset", "virtual_targets")])
)

modalities <- c("rna", "cnv", "protein", "methylation", "he")
modality_labels <- c(
  rna = "RNA", cnv = "CNV", protein = "Protein",
  methylation = "Meth", he = "H&E"
)
subset_levels <- unlist(lapply(1:4, function(n) {
  apply(combn(modalities, n), 2, paste, collapse = "+")
}), use.names = FALSE)

format_subset <- function(x) {
  vapply(strsplit(x, "+", fixed = TRUE), function(parts) {
    paste(unname(modality_labels[parts]), collapse = " + ")
  }, character(1))
}

subset_labels <- format_subset(subset_levels)
routes$source_label <- format_subset(routes$source_subset)
routes$target_label <- format_subset(routes$virtual_targets)
routes$source_label <- factor(
  routes$source_label,
  levels = rev(subset_labels)
)
routes$target_label <- factor(
  routes$target_label,
  levels = subset_labels
)

source_output <- routes[c(
  "evaluation", "source_subset", "virtual_targets",
  "observed_count", "virtual_count", "virtual_gain"
)]
names(source_output)[names(source_output) == "virtual_gain"] <- "delta_c_index_predicted_minus_observed"
source_output <- source_output[order(
  match(source_output$evaluation, c("All patients", "Within cancer")),
  source_output$observed_count,
  match(source_output$source_subset, subset_levels),
  source_output$virtual_count,
  match(source_output$virtual_targets, subset_levels)
), ]
# Source data remain read-only.

grid_data <- expand.grid(
  source_label = subset_labels,
  target_label = subset_labels,
  stringsAsFactors = FALSE
)
grid_data$source_label <- factor(grid_data$source_label, levels = rev(subset_labels))
grid_data$target_label <- factor(grid_data$target_label, levels = subset_labels)

colour_limit <- max(abs(routes$virtual_gain))
group_boundaries <- c(5.5, 15.5, 25.5)

make_plot <- function(evaluation_name) {
  plot_data <- routes[routes$evaluation == evaluation_name, ]

  ggplot() +
    geom_tile(
      data = grid_data,
      aes(x = target_label, y = source_label),
      fill = "#F0F1F3", colour = "white", linewidth = 0.12
    ) +
    geom_tile(
      data = plot_data,
      aes(x = target_label, y = source_label, fill = virtual_gain),
      colour = "white", linewidth = 0.12
    ) +
    geom_vline(
      xintercept = group_boundaries,
      colour = "white", linewidth = 0.75
    ) +
    geom_hline(
      yintercept = group_boundaries,
      colour = "white", linewidth = 0.75
    ) +
    scale_fill_gradient2(
      low = "#B06A78", mid = "#FAF9FB", high = "#6E5292",
      midpoint = 0,
      limits = c(-colour_limit, colour_limit),
      breaks = c(-0.01, 0, 0.01),
      labels = function(x) sprintf("%+.2f", x),
      name = "C-index change versus\nobserved only"
    ) +
    guides(fill = guide_colourbar(
      barheight = grid::unit(32, "mm"),
      barwidth = grid::unit(3.5, "mm"),
      title.position = "top",
      title.hjust = 0.5
    )) +
    scale_x_discrete(drop = FALSE, expand = c(0, 0)) +
    scale_y_discrete(drop = FALSE, expand = c(0, 0)) +
    coord_fixed(clip = "off") +
    labs(
      title = if (evaluation_name == "All patients") {
        "Gains from predicted modalities: all patients"
      } else {
        "Gains from predicted modalities: within-cancer"
      },
      x = "Added predicted modalities",
      y = "Observed modalities"
    ) +
    theme_minimal(base_size = 7.5, base_family = "Helvetica") +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(
        face = "bold", size = 9, hjust = 0.5,
        margin = margin(b = 4)
      ),
      axis.title = element_text(size = 7.5),
      axis.text.x = element_text(
        size = 5.2, angle = 45, hjust = 1, vjust = 1,
        colour = "black"
      ),
      axis.text.y = element_text(size = 5.2, colour = "black"),
      axis.ticks = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 6.5, face = "bold"),
      legend.text = element_text(size = 6),
      plot.margin = margin(5, 6, 5, 5)
    )
}

export_panel <- function(plot, stem, width_cm = 17.8, height_cm = 14.5) {
  pdf(
    file = file.path(output_directory, paste0(stem, ".pdf")),
    width = width_cm / 2.54,
    height = height_cm / 2.54
  )
  print(plot)
  dev.off()

  png(
    filename = file.path(output_directory, paste0(stem, ".png")),
    width = width_cm,
    height = height_cm,
    units = "cm",
    res = 300
  )
  print(plot)
  dev.off()
}

all_patients_plot <- make_plot("All patients")
within_cancer_plot <- make_plot("Within cancer")

export_panel(
  all_patients_plot + theme(legend.position = "none"),
  "ExtendedData3_panel_a"
)
export_panel(
  within_cancer_plot + theme(legend.position = "none"),
  "ExtendedData3_panel_b"
)

legend_table <- ggplotGrob(all_patients_plot)
legend_index <- which(
  grepl("^guide-box", legend_table$layout$name) &
    !vapply(
      legend_table$grobs,
      function(x) inherits(x, "zeroGrob"),
      logical(1)
    )
)
stopifnot(length(legend_index) == 1)
legend_grob <- legend_table$grobs[[legend_index]]

pdf(
  file = file.path(output_directory, "ExtendedData3_shared_legend.pdf"),
  width = 5.5 / 2.54,
  height = 5.5 / 2.54
)
grid::grid.newpage()
grid::grid.draw(legend_grob)
dev.off()

png(
  filename = file.path(output_directory, "ExtendedData3_shared_legend.png"),
  width = 5.5,
  height = 5.5,
  units = "cm",
  res = 300
)
grid::grid.newpage()
grid::grid.draw(legend_grob)
dev.off()

saveRDS(all_patients_plot, "Plots/ExtendedData3/panel_a_plot.rds")

saveRDS(within_cancer_plot, "Plots/ExtendedData3/panel_b_plot.rds")
