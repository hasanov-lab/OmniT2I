# Manuscript figures

These are the scripts for Figures 2–6 and Extended Data Figures 1–3 used in the submission. Figure 3 retains the final eligible/low-count cancer ordering, Figure 4 includes the nine specialist families, Figure 5 uses the final November 6 terminology, and Figure 6 includes the seven prognostic external cohorts. `provenance.tsv` records the original scripts. Figure 1 was supplied as artwork and has no executable plotting script here.

## Run

Install R packages `ggplot2`, `patchwork`, `png`, `ragg`, `gridExtra`, and `readxl`. Place the manuscript Source Data under a working directory as `Source_Data/Figure2/`, …, `Source_Data/ExtendedData3/`, retaining the submitted TSV and workbook names. Tables and patient data are not bundled in this code repository.

```bash
Rscript /path/to/OmniT2I/figures/render.R /path/to/workdir
# Or selected complete figures:
Rscript /path/to/OmniT2I/figures/render.R /path/to/workdir Figure3 Figure6
```

Run the common entry point: it loads the shared theme and exports panels before assembling each plate. Outputs go to `Plots/<figure>/`; input Source Data are read only. Rerunning replaces those generated plot files. Standalone panel PDFs retain vector plot elements. Complete plates preserve the approved absolute positions and use raster panels, with the external heatmap in Figure 6 drawn directly in R. All PNG exports use 300 dpi. Fonts can render differently across systems; the original Helvetica/Liberation Sans settings are retained.

The scripts read saved statistics; they do not fit a model, resample patients, or recalculate confidence intervals. Retired draft extended figures are excluded.
