library(ggdag)
library(ggplot2)
library(ggthemes)

library(dplyr)

dir <- "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
setwd(dir)

# Source theme_elcon
invisible(source('theme_elcon.R'))

theme_set(theme_dag())

## Contribution Margin ##
margin_dag <- dagify(margin ~ revenue + costs + depreciation,
                     revenue ~ other_WIP + progress,
                     depreciation ~ job + customer + risk,
                     costs ~ goods + labor,
                     goods ~ inflation + extra + overrun_risk,
                     labor ~ billable_rate + extra,
                     inflation ~ interest_rate,
                     interest_rate ~ gdp + unemployment,
                     extra ~ change_in_scope,
                     billable_rate ~ illness + efficiency_risk,
                     risk ~ efficiency_risk + overrun_risk,
                     labels = c(
                           "margin" = "Contribution \n Margin",
                           "revenue" = "Revenue",
                           "costs" = "Costs",
                           "depreciation" = "Depreciation",
                           "other_WIP" = "Other WIP",
                           "progress" = "Progress \n of Job",
                           "job" = "Job",
                           "customer" = "Customer",
                           "risk" = "Risk",
                           "efficiency_risk" = "Risk of \n Inefficiency",
                           "overrun_risk" = "Risk of \n Overruns",
                           "goods" = "Goods",
                           "labor" = "Labor",
                           "waste" = "Time-waste",
                           "billable_rate" = "Billable rate",
                           "extra" = "Extra work",
                           "interest_rate" = "Interest Rate",
                           "inflation" = "Inflation",
                           "gdp" = "GDP",
                           "unemployment" = "Unemployment",
                           "change_in_scope" = "Change \n in Scope",
                           "illness" = "Illness"
                         ),
                         latent = c("efficiency_risk","overrun_risk"),
                         exposure = "risk",
                         outcome = "margin"
) %>%
  tidy_dagitty() %>%
  mutate(colour = ifelse(grepl("risk",name), "Unobserved", "Observed"))

ggdag(margin_dag, text = F, use_labels = "label", stylized = F)

## Plot DAG ##
ggdag::ggdag(margin_dag, text = F, stylized = F,
             edge_type = "link_arc") + 
  theme(legend.title = element_blank()) + 
  geom_dag_point(aes(colour = colour)) +
  geom_dag_label_repel(aes(label = label), colour = "black", show.legend = FALSE) +
  scale_color_manual(values=c(vColor[1], vColor[3])) +
  geom_dag_edges()
ggsave('./Results/Presentation/0_dag.png', width = 10, height = 10)

