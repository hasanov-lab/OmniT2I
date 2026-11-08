setwd(".")
library(grid)
out <- 'Plots/Figure6'
d <- as.data.frame(readxl::read_excel("Source_Data/Figure6/Figure6_Source_Data.xlsx", sheet = "09_current_panel_a_external"))
stopifnot(nrow(d) == 7, max(abs(d$external_c_index - d$tcga_c_index - d$delta_external_minus_internal)) < 1e-12)
purple <- colorRampPalette(c('#F5F1F8', '#76569A'))(256)
diverging <- colorRampPalette(c('#A76578', '#FFFFFF', '#4B9187'))(256)
colour_for <- function(x, limits, palette) palette[pmax(1, pmin(256, round((x-limits[1])/diff(limits)*255)+1))]
draw_a <- function(font_scale=1) {
  grid.rect(gp=gpar(fill='white', col=NA))
  text <- function(s, x, y, bold=FALSE, just='centre', size=7) grid.text(s,x,y,just=just,
    gp=gpar(fontfamily='Helvetica', fontsize=size*font_scale, fontface=if(bold) 'bold' else 'plain',col='black'))
  xs <- c(.415,.58,.755)
  # Centre over the data columns (including N/deaths), not the label margin.
  title_x <- mean(c(xs[1] - .163/2, .985))
  text('External performance relative to TCGA',title_x,.90,TRUE, size=8)
  for (j in 1:3) text(c('TCGA\nTest','External','ΔC')[j],xs[j],.828)
  text('N / deaths',.925,.828)
  labels <- sub('SCAN-B','SCAN-B\n(breast cancer)',d$external_cohort,fixed=TRUE)
  vals <- as.matrix(d[,c('tcga_c_index','external_c_index','delta_external_minus_internal')])
  for (i in 1:7) {
    y <- .79 - (i-.5)*.087
    text(labels[i],.325,y,just='right')
    for (j in 1:3) {
      col <- colour_for(vals[i,j],if(j==3)c(-.22,.22)else c(.55,.80),if(j==3)diverging else purple)
      grid.rect(xs[j],y,width=.163,height=.087,gp=gpar(fill=col,col='white',lwd=.7))
      text(sprintf(if(j==3)'%+.3f' else '%.3f',vals[i,j]),xs[j],y,TRUE)
    }
    text(paste0(d$external_patients[i],'/',d$external_deaths[i]),.925,y)
  }
  for (j in 1:2) {
    x <- c(.48,.77)[j]; w <- .23; y <- .151
    pal <- if(j==1)purple else diverging
    grid.raster(as.raster(matrix(pal,nrow=1)),x,y,width=w,height=.012,interpolate=FALSE)
    breaks <- if(j==1)c(.55,.675,.80)else c(-.2,0,.2)
    limits <- if(j==1)c(.55,.80)else c(-.22,.22)
    bx <- x-w/2+(breaks-limits[1])/diff(limits)*w
    for(k in 1:3) {grid.lines(c(bx[k],bx[k]),c(y-.007,y-.012),gp=gpar(lwd=.5));text(format(breaks[k],trim=TRUE),bx[k],y-.026,size=6.5)}
    text(c('C-index','ΔC')[j],x,y-.054)
  }
}
png(file.path(out,'Figure6_panel_a.png'),width=125,height=106,units='mm',res=300);draw_a();dev.off()
cairo_pdf(file.path(out,'Figure6_panel_a.pdf'),width=125/25.4,height=106/25.4);draw_a();dev.off()
