setwd(".")
library(ggplot2)
# Restore the original absolute-C heatmap without changing source statistics.
model_labels <- c(omnit2i="OmniT2I", deepsurv="DeepSurv",
 enet_cox="Elastic-net Cox",lasso_cox="Lasso-Cox",xgboost_cox="XGBoost-Cox",
 rsf="RSF",survival_gbm="Survival GBM",extra_survival_trees="ExtraTrees",
 coxboost="CoxBoost",survival_svm="Survival SVM",adapted_multisurv="MultiSurv (adapted)",
 adapted_senmo="SeNMo (adapted)",adapted_bib_gated="BIB-style (adapted)")
codes <- c(rna="R",cnv="C",protein="P",methylation="M",he="H")
subset_levels <- unlist(lapply(1:5,function(k) apply(combn(names(codes),k),2,paste,collapse="+")))
d <- read.delim("Source_Data/ExtendedData2/model_summary.tsv")
model_labels <- model_labels[!grepl("^adapted_", names(model_labels))]
stopifnot(nrow(d)==10*31, all(d$n_seeds==5))
plot_data <- rbind(transform(d, metric_label="All patients",c_index_mean=pooled_c_index_mean),
                   transform(d, metric_label="Within cancer",c_index_mean=within_cancer_c_index_mean))
plot_data$model_label <- factor(unname(model_labels[plot_data$model]), levels=rev(unname(model_labels)))
plot_data$metric_label <- factor(plot_data$metric_label,levels=c("All patients","Within cancer"))
plot_data$subset_code <- factor(vapply(strsplit(plot_data$source_subset,"+",fixed=TRUE),
  function(z)paste(unname(codes[z]),collapse="+"), character(1)),
  levels=vapply(strsplit(subset_levels,"+",fixed=TRUE),
    function(z)paste(unname(codes[z]),collapse="+"), character(1)))
plot_data$count_label <- factor(plot_data$observed_count,levels=1:5,labels=paste0("n = ",1:5))
plot_data$value_label <- sprintf("%.3f",plot_data$c_index_mean)
plot_data$is_best <- as.logical(ave(plot_data$c_index_mean,plot_data$metric_label,
  plot_data$source_subset,FUN=function(x)x==max(x)))
stopifnot(nrow(plot_data)==620,!anyNA(plot_data$model_label),!anyNA(plot_data$subset_code))

make_panel <- function(scope=NULL) {
  selected <- if(is.null(scope)) plot_data else plot_data[plot_data$metric_label==scope,]
  ggplot(selected,aes(subset_code,model_label,fill=c_index_mean))+
    geom_tile(colour="white",linewidth=.28)+
    geom_tile(data=selected[selected$is_best,],fill=NA,colour="black",linewidth=.35)+
    geom_text(aes(label=value_label),colour="black",fontface="bold",size=1.15)+
    facet_grid(metric_label~count_label,scales="free_x",space="free_x",
               labeller=labeller(metric_label=c("All patients"="All patients","Within cancer"="Within-cancer")))+
    scale_fill_gradient(low="#F4F2F1",high="#5E8498",limits=c(.50,.80),
      oob=scales::squish,name="C-index")+
    labs(title="C-index across 31 modality combinations",x=NULL,y=NULL)+
    guides(fill=guide_colourbar(title.position="top",title.hjust=.5,
      barwidth=grid::unit(24,"mm"),barheight=grid::unit(2.4,"mm")))+
    theme_classic(base_size=8,base_family="Helvetica")+
    theme(text=element_text(colour="black"),strip.background=element_blank(),
      strip.text.x=element_text(size=6.2,face="bold",colour="black"),
      strip.text.y=element_text(size=7,face="bold",colour="black"),
      axis.line=element_blank(),axis.ticks=element_blank(),
      axis.text.x=element_text(size=5.2,angle=45,hjust=1,vjust=1,colour="black"),
      axis.text.y=element_text(size=6.1,colour="black"),
      panel.spacing.x=grid::unit(.8,"mm"),panel.spacing.y=grid::unit(2.5,"mm"),
      legend.position="bottom",legend.box.spacing=grid::unit(1,"pt"),
      legend.margin=margin(t=0,b=0),legend.title=element_text(size=6.2),
      legend.text=element_text(size=6),
      plot.title=element_text(size=9.5,face="bold",hjust=.5,margin=margin(b=1)),
      plot.margin=margin(t=4,r=7,b=3,l=3))
}

for(i in 1:2) save_panel(make_panel(c("All patients", "Within cancer")[i]), "ExtendedData2", letters[i], 175, 75)

output_directory <- "Plots/ExtendedData2/"
## 5. Draw absolute C-index and outline the best model in each combination
heatmap_base <- ggplot(
  plot_data,
  aes(x = subset_code, y = model_label, fill = c_index_mean)
) +
  geom_tile(colour = "white", linewidth = 0.28) +
  geom_tile(
    data = plot_data[plot_data$is_best, ],
    fill = NA,
    colour = "black",
    linewidth = 0.35
  ) +
  geom_text(
    aes(label = value_label),
    colour = "black",
    fontface = "bold",
    size = 1.15
  ) +
  facet_grid(
    metric_label ~ count_label,
    scales = "free_x",
    space = "free_x",
    labeller = labeller(metric_label = c("All patients" = "All patients", "Within cancer" = "Within-cancer"))
  ) +
  scale_fill_gradient(
    low = "#F4F2F1",
    high = "#5E8498",
    limits = c(0.50, 0.80),
    oob = scales::squish,
    name = "C-index"
  ) +
  labs(
    title = "C-index across 31 modality combinations",
    x = NULL,
    y = NULL
  ) +
  guides(
    fill = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = grid::unit(24, "mm"),
      barheight = grid::unit(2.4, "mm")
    )
  ) +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(
    text = element_text(colour = "black"),
    strip.background = element_blank(),
    strip.text.x = element_text(size = 6.2, face = "bold", colour = "black"),
    strip.text.y = element_text(size = 7, face = "bold", colour = "black"),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(
      size = 5.2,
      angle = 45,
      hjust = 1,
      vjust = 1,
      colour = "black"
    ),
    axis.text.y = element_text(size = 6.1, colour = "black"),
    panel.spacing.x = grid::unit(0.8, "mm"),
    panel.spacing.y = grid::unit(2.5, "mm"),
    legend.position = "bottom",
    legend.box.spacing = grid::unit(1, "pt"),
    legend.margin = margin(t = 0, b = 0),
    legend.title = element_text(size = 6.2),
    legend.text = element_text(size = 6),
    plot.title = element_text(
      size = 9.5,
      face = "bold",
      hjust = 0.5,
      margin = margin(b = 1)
    ),
    plot.margin = margin(t = 4, r = 7, b = 3, l = 3)
  )

heatmap_grob <- ggplotGrob(heatmap_base)
legend_index <- which(heatmap_grob$layout$name == "guide-box-bottom")
stopifnot(length(legend_index) == 1)

bottom_row <- gridExtra::arrangeGrob(
  grid::nullGrob(),
  grid::textGrob(
    paste0(
      "Available modalities\n",
      "R: RNA   C: CNV   P: protein   M: methylation   H: histopathology"
    ),
    gp = grid::gpar(fontfamily = "Helvetica", fontsize = 6.5)
  ),
  heatmap_grob$grobs[[legend_index]],
  grid::nullGrob(),
  ncol = 4,
  widths = c(2, 3.8, 1.3, 2)
)

heatmap_plot <- gridExtra::arrangeGrob(
  heatmap_base + theme(legend.position = "none"),
  bottom_row,
  ncol = 1,
  heights = c(1, 0.10)
)


## 6. Export without replacing the earlier B heatmap or the D boxplot
ggsave(
  filename = paste0(
    output_directory,
    "heatmap.png"
  ),
  plot = heatmap_plot,
  width = 175,
  height = 140,
  units = "mm",
  dpi = 300,
  bg = "white"
)

grDevices::pdf(
  file = paste0(
    output_directory,
    "heatmap.pdf"
  ),
  width = 175 / 25.4,
  height = 140 / 25.4,
  family = "Helvetica",
  onefile = TRUE
)

grid::grid.draw(heatmap_plot)
dev.off()
