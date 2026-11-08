setwd(".")
library(ggplot2)

revision_plot_root <- "Plots"
revision_source_root <- "Source_Data"
revision_theme <- function() {
  theme_classic(base_size = 6.5, base_family = "Helvetica") +
    theme(plot.title = element_text(size = 7, face = "bold", hjust = 0,
                                    margin = margin(b = 6, l = 10)),
          plot.title.position = "plot", plot.subtitle = element_blank(),
          axis.text = element_text(size = 6.5, colour = "black"),
          axis.title = element_text(size = 6.5),
          legend.text = element_text(size = 6.5), legend.title = element_blank(),
          strip.text = element_text(size = 6.5),
          panel.background = element_rect(fill = "white", colour = NA),
          plot.background = element_rect(fill = "white", colour = NA),
          legend.key.width = unit(4, "mm"), legend.spacing.x = unit(1, "mm"),
          legend.margin = margin(0, 0, 0, 0), plot.margin = margin(4, 4, 4, 4))
}
read_source <- function(figure, name) read.delim(file.path(revision_source_root, figure, paste0(name, ".tsv")))
save_panel <- function(p, figure, panel, width, height) {
  p <- p + theme(legend.position = "none")
  path <- file.path(revision_plot_root, figure)
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(path, paste0("panel_", panel, ".pdf")), p, width = width, height = height, units = "mm")
  ggsave(file.path(path, paste0("panel_", panel, ".png")), p, width = width, height = height, units = "mm", dpi = 300, bg = "white")
  file.copy(file.path(path, paste0("panel_", panel, ".png")), file.path(path, paste0(figure, "_panel_", panel, ".png")), overwrite = TRUE)
  file.copy(file.path(path, paste0("panel_", panel, ".pdf")), file.path(path, paste0(figure, "_panel_", panel, ".pdf")), overwrite = TRUE)
  saveRDS(p, file.path(path, paste0("panel_", panel, "_plot.rds")))
}
