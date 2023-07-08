# sandro pezzelle
# amsterdam 2023
# code for obtaining violin plot in Figure 2

library(ggplot2)
library(dplyr)
library(viridis)

mypath <- "./"
setwd(mypath)
mydataFULL <- read.csv("data4violin-full.csv", header=TRUE, sep=";")
head(mydataFULL)

mydataFULL %>%
  group_by(type) %>%
  summarise(total_non_na = sum(!is.na(alignment)))

mylabels = mydataFULL %>%
  group_by(type) %>%
  summarise(total_non_na = sum(!is.na(alignment)))

mymeans <- mydataFULL %>%
  group_by(type) %>%
  dplyr::summarize(Mean = mean(alignment, na.rm=TRUE))

mydataFULL$type <- factor(mydataFULL$type, levels = c("original", "quantity", "gender", "gender+number", "location", "object", "full"))


vFULL <- ggplot(mydataFULL, aes(x=type, y=alignment, fill=type)) + geom_violin(trim=FALSE) + stat_summary(fun.data=mean_sdl, mult=1, geom="pointrange", color="black") + ylab("CLIPScore") + xlab("") + scale_y_continuous(breaks = seq(0.2, 1.0, by = 0.2)) + theme(legend.position="top", legend.title= element_blank(), axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), axis.title.y = element_text(size = 12)) + geom_text(data=mylabels,aes(x = type, y = 1.2, label=total_non_na),color="black", fontface="bold") + guides(fill = guide_legend(nrow = 1)) + geom_text(data=mymeans,aes(x = type, y = 0.2, label=round(Mean,4)),color="black", fontface="bold")

vFULL + scale_color_viridis(discrete = TRUE, option = "D") + scale_fill_viridis(discrete = TRUE)