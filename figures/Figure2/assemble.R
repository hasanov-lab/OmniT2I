setwd(".")
## Reference plates only. Standalone R panels retain their original export sizes.
## Positions are read from the actual v1.5 OmniT2I_latest_share.pptx.
draw_legacy_ppt_plate <- function(number, placements, tags, height_px) {
  scale <- 1680 / 6858000
  draw <- function() {
    grid::grid.newpage()
    grid::pushViewport(grid::viewport(xscale=c(0,1780), yscale=c(0,height_px)))
    grid::grid.rect(gp=grid::gpar(fill="white", col=NA))
    grid::grid.rect(x=0,y=height_px,width=1780,height=82,default.units="native",
      just=c("left","top"),gp=grid::gpar(fill="#f0f0f0",col=NA))
    grid::grid.text(paste("Figure",number),x=42,y=height_px-17-8,default.units="native",
      just=c("left","top"),gp=grid::gpar(fontfamily="Liberation Sans",
      fontface="bold",fontsize=47*72/220,col="#111111"))
    for (i in seq_len(nrow(placements))) {
      g <- placements[i,]
      x <- 50 + round(g$x*scale); y <- 105 + round(g$y*scale)
      w <- round(g$w*scale); h <- round(g$h*scale)
      grid::grid.raster(png::readPNG(g$file),x=x,y=height_px-y,width=w,height=h,
        default.units="native",just=c("left","top"),interpolate=TRUE)
    }
    for(i in seq_len(nrow(tags))) {
      g<-tags[i,]
      grid::grid.text(g$tag,x=max(25,50+round(g$x*scale)),
        y=height_px-max(88,105+round(g$y*scale))-8,default.units="native",
        just=c("left","top"),gp=grid::gpar(fontfamily="Liberation Sans",
        fontface="bold",fontsize=43*72/220,col="#111827"))
    }
    grid::popViewport()
  }
  stem<-file.path("Plots", paste0("Figure", number), paste0("Figure", number))
  grDevices::png(paste0(stem,".png"),width=1780/220,height=height_px/220,units="in",res=300,type="cairo")
  draw(); grDevices::dev.off()
  ## PDF is the same assembled reference plate; use standalone PDFs for vector editing.
  grDevices::cairo_pdf(paste0(stem,".pdf"),width=1780/220,height=height_px/220)
  draw(); grDevices::dev.off()
}

placements <- data.frame(file=c("Plots/Figure2/panel_c.png","Plots/Figure2/panel_b.png","Plots/Figure2/panel_a.png"), x=c(820712,3429000,820712), y=c(2061803,103928,103929), w=c(5072274,2463986,2104232), h=c(2453106,1608711,1711241))
tags <- data.frame(tag=c("a","b","c"), x=c(692696,3304662,692696), y=c(-41732,-19996,1920498))
draw_legacy_ppt_plate(2, placements, tags, 1246)
