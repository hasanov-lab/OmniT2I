setwd(".")
# 1. Retain the saved component order, estimates and seed-level differences.
d <- read_source("Figure6", "panel_a_seed_differences")
s <- read_source("Figure6", "panel_a_summary")
variants <- c("no_pretraining", "random_layout", "no_c3", "no_c2")
s <- s[match(variants, s$variant), ]
d <- d[d$variant %in% variants, ]
s$row <- 4:1; d$row <- 5 - match(d$variant, variants)
lev <- rev(s$label)
display_labels <- sub("^No ", "Without ", lev)
d$offset <- (match(d$seed, sort(unique(d$seed))) - 3) * 0.045
s$contributing <- s$estimate < 0 & s$negative_seeds >= 4
stopifnot(all(s$n_seeds == 5), all(table(d$label) == 5))

# 2. Bars are means, points are seeds; only the predefined contributors are filled.
p <- ggplot(s, aes(estimate, row)) +
  geom_vline(xintercept = 0, colour = "#ADADAD", linetype = 2, linewidth = 0.25) +
  geom_col(aes(fill = contributing), orientation = "y", width = 0.6,
           colour = "#76569A", linewidth = 0.3) +
  geom_point(data = d, aes(delta, row + offset), inherit.aes = FALSE,
             colour = "#463259", size = 0.8, alpha = 0.85) +
  scale_fill_manual(values = c("TRUE" = "#76569A", "FALSE" = "white"), guide = "none") +
  scale_x_continuous(breaks = c(-0.03, -0.02, -0.01, 0),
                     labels = scales::label_number(accuracy = 0.01), limits = c(-0.0355, 0.017)) +
  scale_y_continuous(breaks = seq_along(lev), labels = display_labels,
                     limits = c(0.5, 4.5)) +
  labs(title = "Input representation and encoder ablations",
       x = "Within-cancer C-index difference\n(ablation - full)", y = NULL) + revision_theme()
p <- p + theme(plot.title.position="panel", plot.title=element_text(hjust=0.5, margin=margin(b=6)))
save_panel(p, "Figure6", "b", 90, 78)
