
library(extrafont)
loadfonts(device = "win")
# Custom Theme #
theme_elcon <- function(){

    font <- "sans"   #assign font family up front
    theme_economist_white() %+replace%    #replace elements we want to change
    theme(
      plot.background = element_blank(),
      legend.background = element_blank(),
      plot.title = element_text(
                   family = font,
                   size = 20,
                   face = 'bold',
                   hjust = 0,
                   vjust = 2),
      plot.subtitle = element_text(
                   family = font,
                   size = 14),
      plot.caption = element_text(
                   family = font,
                   size = 9,
                   hjust = 1),
      axis.title = element_text(
                   family = font,
                   size = 10),
      axis.text = element_text(
                   family = font,
                   size = 9),
      axis.text.x = element_text(
                    margin=margin(5, b = 10))
    )
}

vColor <- c('#006e64','#ffbb00','#c17150','#1e8c82','#734848','#dcfae9')

options(scipen = 999)
