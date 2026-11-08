setwd(".")
output_root <- "Plots"
plate <- function(name, files, x, y, w, h, tags, width, height) {
  draw <- function() {
    grid.newpage(); pushViewport(viewport(xscale=c(0,width),yscale=c(0,height)))
    grid.rect(gp=gpar(fill="white",col=NA))
    for(i in seq_along(files)) grid.raster(png::readPNG(files[i]),x=x[i],y=height-y[i],
      width=w[i],height=h[i],default.units="native",just=c("left","top"))
    for(i in seq_len(nrow(tags))) grid.text(tags$label[i],x=tags$x[i],y=height-tags$y[i],
      default.units="native",just=c("left","top"),gp=gpar(fontfamily="Helvetica",fontface="bold",fontsize=43*72/220))
    popViewport()
  }
  path <- file.path(output_root,name)
  ragg::agg_png(file.path(path,paste0(name,".png")),width=width/220,height=height/220,units="in",res=300,background="white")
  draw(); dev.off()
  cairo_pdf(file.path(path,paste0(name,".pdf")),width=width/220,height=height/220,family="Helvetica")
  draw();dev.off()
}
sc <- 1680/6858000
plate("Figure4",file.path(output_root,"Figure4",c("Figure4_panel_a.png","Figure4_panel_b.png")),
  50+round(c(44625,2936936)*sc),22+round(c(53424,53382)*sc),
  round(c(2846623,3922127)*sc),round(c(1768224,1768307)*sc),
  data.frame(label=c("a","b"),x=c(25,50+round(2808920*sc)),y=c(13,13)),1780,503)
