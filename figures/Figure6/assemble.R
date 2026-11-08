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
## Preserve the approved b-d panels and layout; edits and assembly remain in R.
draw_all <- function() {
  grid.newpage();grid.rect(gp=gpar(fill='white',col=NA))
  xy <- data.frame(panel=c('a','b','c','d'), x=c(2,131,19.5,131),y=c(6,6,116,116),w=c(125,90,90,90))
  for(i in 1:4) {
    z <- xy[i,]; im <- png::readPNG(file.path(out,paste0('Figure6_panel_',z$panel,'.png')))
    h <- z$w * dim(im)[1]/dim(im)[2]
    pushViewport(viewport(x=z$x/223,y=1-z$y/185,width=z$w/223,height=h/185,just=c('left','top')))
    if(z$panel=='a') draw_a(font_scale=180/223) else grid.raster(im,interpolate=FALSE)
    popViewport()
    grid.text(z$panel,x=if(z$panel %in% c('a','c'))2/223 else 131/223,y=1-(z$y-1)/185,just=c('left','bottom'),gp=gpar(fontfamily='Helvetica',fontface='bold',fontsize=8,col='black'))
  }
}
png(file.path(out,'Figure6.png'),width=180,height=185*180/223,units='mm',res=300,type='cairo');draw_all();dev.off()
cairo_pdf(file.path(out,'Figure6.pdf'),width=180/25.4,height=185*180/223/25.4);draw_all();dev.off()
