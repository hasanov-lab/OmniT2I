setwd(".")
rows <- read.delim(file.path(code_root, "ExtendedData3", "layout.tsv"), check.names = FALSE)
destination <- "ExtendedData3"
width <- 1780; height <- 847
scale <- 1680 / 6858000
  draw_plate <- function() {
    grid.newpage(); pushViewport(viewport(xscale = c(0, width), yscale = c(0, height)))
    grid.rect(gp = gpar(fill = "white", col = NA))
    for (i in which(rows$kind == "image")) {
      raster <- png::readPNG(file.path(path, rows$panel[i]))
      grid.raster(raster, x = 50 + round(rows$x_emu[i] * scale),
        y = height - (22 + round(rows$y_emu[i] * scale)),
        width = round(rows$cx_emu[i] * scale), height = round(rows$cy_emu[i] * scale),
        default.units = "native", just = c("left", "top"), interpolate = TRUE)
    }
    for (i in which(rows$kind == "text")) {
      grid.text(rows$text[i], x = max(25, 50 + round(rows$x_emu[i] * scale)),
        y = height - (max(5, 22 + round(rows$y_emu[i] * scale)) + 8), default.units = "native",
        just = c("left", "top"), gp = gpar(fontfamily = "Liberation Sans", fontface = "bold",
          fontsize = 43 * 72 / 220, col = "#111827"))
    }
    popViewport()
  }
  path <- file.path("Plots", destination)
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  ragg::agg_png(file.path(path, paste0(destination, ".png")), width = width / 220, height = height / 220,
                units = "in", res = 300, background = "white")
  draw_plate(); dev.off()
  cairo_pdf(file.path(path, paste0(destination, ".pdf")), width = width / 220,
            height = height / 220, family = "Liberation Sans")
  draw_plate(); dev.off()
