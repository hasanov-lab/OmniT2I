setwd(".")
d <- read_source("Figure6", "panel_c_rna_addition_ci")
d$row <- match(d$added_modality, rev(c("cnv", "protein", "methylation", "he")))
d$colour <- ifelse(d$delta_c < 0, "#A76578", "#4B9187")
stopifnot(max(abs(d$c_after_mean - d$c_before_mean - d$delta_c)) < 1e-12)
p <- ggplot(d, aes(delta_c, row)) +
  geom_vline(xintercept = 0, colour = "#ADADAD", linetype = 2, linewidth = .25) +
  geom_col(aes(fill = colour), orientation = "y", width = .48) +
  geom_errorbar(aes(xmin = ci_lower, xmax = ci_upper), orientation = "y",
                width = .15, linewidth = .4, colour = "#444444") +
  scale_fill_identity() +
  scale_x_continuous(breaks = seq(-.02, .03, .01), limits = c(-.02, .033)) +
  scale_y_continuous(breaks = 1:4,
    labels = rev(c("RNA + CNV", "RNA + protein", "RNA + meth.", "RNA + H&E"))) +
  labs(title = "Incremental prognostic value of measured modalities",
       x = "Within-cancer C-index change", y = NULL) + revision_theme()

save_panel(p + theme(plot.title.position="panel", plot.title=element_text(hjust=0.5, margin=margin(b=6))), "Figure6", "d", 90, 66)
