setwd(".")
## Reference plates only. Standalone R panels retain their original export sizes.
## Positions are read from the actual v1.5 OmniT2I_latest_share.pptx.
draw_legacy_ppt_plate <- function(number, placements, tags, height_px) {
  scale <- 1680 / 6858000
  draw <- function() {
    grid::grid.newpage()
    grid::pushViewport(grid::viewport(xscale=c(0,1780), yscale=c(0,height_px)))
    grid::grid.rect(gp=grid::gpar(fill="white", col=NA))
    for (i in seq_len(nrow(placements))) {
      g <- placements[i,]
      x <- 50 + round(g$x*scale); y <- 15 + round(g$y*scale)
      w <- round(g$w*scale); h <- round(g$h*scale)
      grid::grid.raster(png::readPNG(g$file),x=x,y=height_px-y,width=w,height=h,
        default.units="native",just=c("left","top"),interpolate=TRUE)
    }
    for(i in seq_len(nrow(tags))) {
      g<-tags[i,]
      grid::grid.text(g$tag,x=max(25,50+round(g$x*scale)),
        y=height_px-max(0,15+round(g$y*scale))-8,default.units="native",
        just=c("left","top"),gp=grid::gpar(fontfamily="Liberation Sans",
        fontface="bold",fontsize=43*72/220,col="#000000"))
    }
    grid::popViewport()
  }
  stem<-file.path("Plots",paste0("Figure",number),paste0("Figure",number))
  grDevices::png(paste0(stem,".png"),width=round(1780/220*300),height=round(height_px/220*300),res=300,type="cairo")
  draw(); grDevices::dev.off()
  ## PDF is the same assembled reference plate; use standalone PDFs for vector editing.
  grDevices::cairo_pdf(paste0(stem,".pdf"),width=1780/220,height=height_px/220)
  draw(); grDevices::dev.off()
}

## 1. Use the original slide's PNG placements without changing standalone panels.
placements <- data.frame(
  file=c("Plots/Figure5/Figure5_panel_d.png",
    "Plots/Figure5/Figure5_panel_e.png",
    "Plots/Figure5/Figure5_panel_a.png",
    "Plots/Figure5/Figure5_panel_b.png",
    "Plots/Figure5/Figure5_panel_c.png"),
  x=c(1379620,3580663,345566,2419126,4492686),
  y=c(2158192,2158192,112465,112465,112465),
  w=c(2080078,2080078,1960650,1960650,1960650),
  h=c(1809832,1809832,1809831,1809831,1809831))
tags<-data.frame(tag=c("a","b","c","d","e"),
  x=c(217550,2291110,4364670,1251071,3452647),y=c(-15552,-15551,-15552,2036456,2031972))
## 2. Original reference-page dimensions and original absolute panel positions.
draw_legacy_ppt_plate(5,placements,tags,1022)
