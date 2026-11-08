setwd(".")

# 1. Summarize already saved seed differences, not the delta_sd summary column.
seed <- read_source("Figure6", "panel_b_seed_values")
s <- aggregate(full_minus_ablation ~ observed_count, seed, mean)
names(s)[2] <- "delta_mean"
s$sd <- aggregate(full_minus_ablation ~ observed_count, seed, sd)$full_minus_ablation
s$lower <- s$delta_mean - s$sd; s$upper <- s$delta_mean + s$sd
# 2. Retain the final mean and between-seed SD ribbon.
p <- ggplot(s, aes(observed_count, delta_mean)) +
  geom_hline(yintercept = 0, colour = "#ADADAD", linetype = 2, linewidth = 0.25) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#4B9187", alpha = 0.18, colour = NA) +
  geom_line(colour = "#4B9187", linewidth = 0.55) +
  geom_point(colour = "#4B9187", size = 1.6) +
  scale_x_continuous(breaks = 1:5, limits = c(0.85, 5.15)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.001)) +
  labs(title = "Effect of modality dropout",
       x = "Measured modalities", y = "Gain from modality dropout") + revision_theme()
save_panel(p + theme(plot.title.position="panel", plot.title=element_text(hjust=0.5, margin=margin(b=6))), "Figure6", "c", 90, 66)
