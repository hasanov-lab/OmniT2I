setwd(".")
output_dir <- "Plots/Figure3"
## Preserve the original PPT absolute positions, a-d and legend files. Only the
## last row and canvas grow to preserve the all-cancer panels' existing aspect.
placements <- data.frame(stem = c("Figure3_panel_a", "Figure3_panel_b",
  "Figure3_panel_f", "Figure3_panel_e", "Figure3_panel_d", "Figure3_panel_c", "Figure3_shared_legend"),
  x = c(1227904, 2882153, 3403574, 1229948, 3402410, 1229948, 5275508),
  y = c(112465, 112464, 3685623, 3685408, 1809731, 1809731, 4338310),
  w = c(1593338, 2538036, 2006544, 2026574, 2065968, 2065967, 385740),
  h = c(1566024, 1457371, 2006544 * 15.2 / 8.6, 2026574 * 15.2 / 8.6, 1724351, 1724351, 1147068))
tags <- data.frame(tag = c("a", "b", "c", "d", "e", "f"),
  x = c(1107118, 2749233, 1101934, 3269490, 1101932, 3300588),
  y = c(-15552, -15552, 1665459, 1674464, 3557589, 3557589))
scale <- 1680 / 6858000
height_px <- ceiling(max(105 + placements$y * scale + placements$h * scale) + 35)
draw <- function() {
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(xscale = c(0, 1780), yscale = c(0, height_px)))
  grid::grid.rect(gp = grid::gpar(fill = "white", col = NA))
  for (i in seq_len(nrow(placements))) {
    g <- placements[i, ]
    grid::grid.raster(png::readPNG(file.path(output_dir, paste0(g$stem, ".png"))),
      x = 50 + round(g$x * scale), y = height_px - 105 - round(g$y * scale),
      width = round(g$w * scale), height = round(g$h * scale), default.units = "native",
      just = c("left", "top"), interpolate = TRUE)
  }
  for (i in seq_len(nrow(tags))) {
    g <- tags[i, ]
    grid::grid.text(g$tag, x = max(25, 50 + round(g$x * scale)),
      y = height_px - max(88, 105 + round(g$y * scale)) - 8, default.units = "native",
      just = c("left", "top"), gp = grid::gpar(fontfamily = "Liberation Sans",
        fontface = "bold", fontsize = 43 * 72 / 220, col = "#111827"))
  }
  grid::popViewport()
}
## 300 dpi at the original physical scale, not a change to panel typography.
grDevices::png(file.path(output_dir, "Figure3.png"),
  width = 1780 / 220, height = height_px / 220, units = "in", res = 300, type = "cairo")
draw(); grDevices::dev.off()
grDevices::cairo_pdf(file.path(output_dir, "Figure3.pdf"), width = 1780 / 220, height = height_px / 220)
draw(); grDevices::dev.off()
write.table(placements, file.path(output_dir, "assembly_placements.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
