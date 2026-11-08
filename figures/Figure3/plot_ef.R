setwd(".")

library(ggplot2)
library(patchwork)

output_dir <- "Plots/Figure3"
source_dir <- "Plots/Figure3"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

## Copy the saved per-cancer estimates and intervals; no inference or resampling.
source_file <- "Source_Data/Figure3/Fig3ef_cancer_specific_95CI_eligible.tsv"
cancer_performance <- read.delim(source_file, stringsAsFactors = FALSE, check.names = FALSE)
cancer_performance$lower <- cancer_performance$lower_95ci
cancer_performance$upper <- cancer_performance$upper_95ci
cancer_performance$eligible <- with(cancer_performance,
  n_patients >= 25 & n_events >= 15 & estimability == "estimable" & is.finite(c_index_mean))
stopifnot(nrow(cancer_performance) == 64L,
          !anyDuplicated(cancer_performance[, c("scenario", "cancer_type")]),
          sum(cancer_performance$estimability == "estimable") == 62L)
patient_limits <- range(cancer_performance$n_patients, na.rm = TRUE)
event_limits <- range(cancer_performance$n_events, na.rm = TRUE)

macro_performance <- unique(cancer_performance[, c("scenario", "scenario_label")])
macro_performance$n_evaluable_cancers <- vapply(macro_performance$scenario,
  function(s) sum(cancer_performance$scenario == s & cancer_performance$estimability == "estimable" &
    is.finite(cancer_performance$c_index_mean)), integer(1))
macro_performance$n_eligible_cancers <- vapply(macro_performance$scenario,
  function(s) sum(cancer_performance$scenario == s & cancer_performance$eligible), integer(1))
macro_performance$macro_c_index <- vapply(macro_performance$scenario,
  function(s) mean(cancer_performance$c_index_mean[cancer_performance$scenario == s &
    cancer_performance$eligible]), numeric(1))
stopifnot(all(macro_performance$n_evaluable_cancers == c(32L, 30L)))

make_cancer_specific_plot <- function(scenario_name) {
  plot_data <- cancer_performance[cancer_performance$scenario == scenario_name &
    cancer_performance$estimability == "estimable" & is.finite(cancer_performance$c_index_mean), ]
  plot_data <- plot_data[order(!plot_data$eligible, -plot_data$c_index_mean), ]
  plot_data$cancer_label <- paste0(plot_data$cancer_type, " (",
    format(plot_data$n_patients, big.mark = ",", scientific = FALSE, trim = TRUE), "/",
    format(plot_data$n_events, big.mark = ",", scientific = FALSE, trim = TRUE), ")")
  plot_data$cancer_label <- factor(plot_data$cancer_label, levels = rev(plot_data$cancer_label))
  macro_value <- macro_performance$macro_c_index[macro_performance$scenario == scenario_name]
  scenario_title <- macro_performance$scenario_label[macro_performance$scenario == scenario_name]
  n_eligible <- sum(plot_data$eligible)
  ggplot(plot_data, aes(y = cancer_label)) +
    geom_vline(xintercept = 0.5, linewidth = 0.35, linetype = "dotted", colour = "#8C9AA8") +
    geom_vline(xintercept = macro_value, linewidth = 0.55, linetype = "longdash", colour = "#24577A") +
    geom_hline(yintercept = nrow(plot_data) - n_eligible + 0.5,
               linewidth = 0.25, colour = "#C5CDD4") +
    geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y",
                  width = 0.28, linewidth = 0.4, colour = "black") +
    geom_point(data = plot_data[plot_data$eligible, ],
      aes(x = c_index_mean, size = n_patients, colour = n_events), alpha = 1) +
    geom_point(data = plot_data[!plot_data$eligible, ],
      aes(x = c_index_mean, size = n_patients), shape = 21,
      colour = "#8C9AA8", fill = "white", stroke = 0.5, alpha = 0.3) +
    scale_x_continuous(limits = c(0.20, 0.95), breaks = seq(0.2, 0.9, 0.1)) +
    scale_y_discrete(limits = levels(plot_data$cancer_label), drop = FALSE) +
    scale_size_continuous(range = c(1.3, 3.7), limits = patient_limits, name = "Patients") +
    scale_colour_gradientn(colours = c("#8FC4CD", "#63AAB8", "#3F8FA4", "#004F6B"),
      limits = event_limits, breaks = seq(0, 400, 100), name = "Events") +
    guides(size = guide_legend(ncol = 1, byrow = TRUE, order = 1,
      title.position = "top", title.hjust = 0.5,
      theme = theme(legend.title = element_text(vjust = 0.5))),
      colour = guide_coloursteps(direction = "vertical", order = 2,
        title.position = "top", title.hjust = 0.5, even.steps = TRUE, show.limits = FALSE,
        theme = theme(legend.title = element_text(vjust = 0.5),
          legend.key.width = grid::unit(0.48, "cm"), legend.key.height = grid::unit(2.4, "cm")))) +
    labs(title = paste0(scenario_title, "\nMacro-average: ", sprintf("%.3f", macro_value)), x = "C-index", y = NULL) +
    theme_classic(base_size = 8, base_family = "Helvetica") +
    theme(plot.title = element_text(face = "bold", size = 9, hjust = 0.5),
      axis.text = element_text(colour = "#1F252B"), legend.position = "none",
      plot.margin = margin(4, 4, 2, 4))
}
fig3e_plot <- make_cancer_specific_plot("natural_availability")
fig3f_plot <- make_cancer_specific_plot("complete_case")
for (plot in list(fig3e_plot, fig3f_plot)) {
  d <- plot$data
  stopifnot(all(diff(as.integer(!d$eligible)) >= 0))
  for (eligible_group in c(TRUE, FALSE))
    stopifnot(all(diff(d$c_index_mean[d$eligible == eligible_group]) <= 0))
}
stopifnot(nrow(fig3e_plot$data) == 32L, nrow(fig3f_plot$data) == 30L)
fig3ef_plot <- (fig3e_plot | fig3f_plot) +
  plot_annotation(tag_levels = list(c("e", "f"))) &
  theme(plot.tag = element_text(face = "bold", size = 10))
## Keep all standalone export dimensions, fonts, scale ranges and spacing unchanged.
for (extension in c("png", "pdf")) {
  ggsave(file.path(output_dir, paste0("Figure3_panel_e.", extension)),
    fig3e_plot, width = 8.6, height = 15.2, units = "cm", dpi = 300, bg = "white")
  ggsave(file.path(output_dir, paste0("Figure3_panel_f.", extension)),
    fig3f_plot, width = 8.6, height = 15.2, units = "cm", dpi = 300, bg = "white")

}
write.table(cancer_performance, file.path(source_dir, "Fig3ef_cancer_specific_95CI_eligible.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE)
write.table(macro_performance, file.path(source_dir, "macro_average_eligible.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE)
display_order <- rbind(
  data.frame(scenario = "natural_availability", display_order = seq_len(nrow(fig3e_plot$data)),
    cancer_type = fig3e_plot$data$cancer_type, eligible = fig3e_plot$data$eligible),
  data.frame(scenario = "complete_case", display_order = seq_len(nrow(fig3f_plot$data)),
    cancer_type = fig3f_plot$data$cancer_type, eligible = fig3f_plot$data$eligible))
write.table(display_order, file.path(source_dir, "cancer_order_by_panel.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE)
print(macro_performance)

legend_plot <- fig3e_plot + theme(legend.position = "right")
g <- ggplotGrob(legend_plot)
i <- which(grepl("^guide-box", g$layout$name) & !vapply(g$grobs, inherits, logical(1), "zeroGrob"))
for (extension in c("png", "pdf")) ggsave(file.path(output_dir, paste0("Figure3_shared_legend.", extension)),
  g$grobs[[i]], width = 3.2, height = 9.7, units = "cm", dpi = 300, bg = "white")

saveRDS(fig3e_plot, "Plots/Figure3/panel_e_plot.rds")

saveRDS(fig3f_plot, "Plots/Figure3/panel_f_plot.rds")
