setwd({
  script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)))
  code_root <- dirname(script)
  commandArgs(trailingOnly = TRUE)[1]
})
library(ggplot2)
library(grid)
source(file.path(code_root, "helpers.R"))
figures <- c("Figure2", "Figure3", "Figure4", "Figure5", "Figure6", "ExtendedData1", "ExtendedData2", "ExtendedData3")
requested <- commandArgs(trailingOnly = TRUE)[-1]
if (length(requested)) figures <- requested
for (figure in figures) {
  dir.create(file.path("Plots", figure), recursive = TRUE, showWarnings = FALSE)
  scripts <- sort(list.files(file.path(code_root, figure), pattern = "^plot_.*[.]R$", full.names = TRUE))
  for (script in scripts) sys.source(script, envir = new.env(parent = globalenv()))
  assembly <- file.path(code_root, figure, "assemble.R")
  if (file.exists(assembly)) sys.source(assembly, envir = new.env(parent = globalenv()))
  message(figure, ": exported PNG and PDF")
}
