library(data.tree)
library(TreeBH)
library(tidyr)
library(collapsibleTree)
library(htmltools)

# Setup the GLM for non-experimental condition tests
#ken <- read.csv("./merge_Annotated_counts_clean_with_density.csv")
ken <- read.csv("//10.159.50.7/LabCommon/Ken/data/OPTRAP/result/Figure6_D_glm_stat_df_no_batch.csv")
output_path = "//10.159.50.7/LabCommon/Ken/data/OPTRAP/result"
dat <- drop_na(ken)


# Assuming 'Variables' is a list of variable names
qvalue_threshold = .001
#qvalue_threshold = .1
# Assuming 'dat' is your data frame
var = 'pvalue'
save_key = 'Figure6_D_glm_stat_df_no_batch'
dat$t.p.val <- dat[[var]]  # Perform your operation here

#dat$t.p.val <- dat$Saline

## convert to tree
tree <- FromDataFrameNetwork(dat)
## slice off false root
if (length(tree$children) == 1) {
  tree <- tree$children[[1]]
  tree$parent <- NULL # make it actually the root
}

## set p values for leaf nodes only
tree$Do(function(node) node$p.val <- node$t.p.val, filterFun = isLeaf)

## split on aggregation strategy
tree.s <- Clone(tree)
tree.f <- Clone(tree)

## propagate p-values up
tree.s$Do(function(node) {
  node$p.val <- Aggregate(
    node,
    attribute = "p.val", aggFun = TreeBH::get_simes_p
  )
})



tree.f$Do(function(node) {
  node$p.val <- Aggregate(
    node,
    attribute = "p.val", aggFun = TreeBH::get_fisher_p
  )
})

## set desired q value per level (all set the same for now)
L <- tree$height
qs <- rep(qvalue_threshold, L) # last value is inconsequential

tree.s$Do(function(node) {
  node$q_l <- qs[node$level]
})

tree.f$Do(function(node) {
  node$q_l <- qs[node$level]
})


## Process a family of hypotheses recursively
FamilyCheckR <- function(node) {
  if (node$isRoot) node$q_adj <- 1 # no adjustment for root
  
  q_target <- node$q_adj * node$q_l
  
  children <- node$children
  
  child_ps <- sapply(children, function(x) x$p.val)
  child_ps_adj <- p.adjust(child_ps, method = "BH")
  
  purrr::walk2(
    children, child_ps_adj,
    function(node, p_adj) node$rejected <- p_adj < q_target
  )
  
  prop_children_rejected <- mean(
    purrr::map_lgl(children, function(x) x$rejected)
  )
  
  child_q_adj <- node$q_adj * prop_children_rejected
  
  purrr::walk(children, function(x) x$q_adj <- child_q_adj)
  
  ## predicate for checking if we should recurse
  recursep <- function(x) {
    !is.na(x$rejected) && x$rejected && !x$isLeaf
  }
  
  purrr::walk(
    children,
    function(x) if (recursep(x)) FamilyCheckR(x)
  )
}


## reject the root manually
tree.s$rejected <- TRUE
tree.f$rejected <- TRUE

## work down the tree
FamilyCheckR(tree.s)
FamilyCheckR(tree.f)


## look at the results
#print(tree.s, "q_l", "rejected")
#print(tree.f, "q_l", "rejected")




## visualize the network: collapsibleTree
#install.packages('collapsibleTree')

## simes first
tree.s$Do(function(x) x$collapsed <- ifelse(x$rejected, FALSE, TRUE))

tree.s$Do(function(x) x$color <- ifelse(x$rejected, ifelse(sign(x$p.val) == 1, "red", "green"), "gray"))
tree.s$Do(function(x) x$name <- x$acronym)

tree.s$Do(function(x) x$tooltip <- sprintf("Region: %s<br>p: %0.2f", x$Name, x$p.val))


#treeList <- ToListExplicit(tree.f, unname = TRUE)
#radialNetwork(treeList)

#install.packages('htmltools')
c = collapsibleTree(tree.s,
                    fill = "color", zoomable = FALSE, width = 1000, attribute = "name",
                    height = 800, tooltip = TRUE, tooltipHtml = "tooltip", collapsed = TRUE
)
filename <- paste0(output_path,"/TreeFDRS_", var,'_',save_key, ".html")

# Saving the HTML file
htmltools::save_html(c, file = filename)
## fisher next

tree.f$Do(function(x) x$collapsed <- ifelse(x$rejected, FALSE, TRUE))

tree.f$Do(function(x) x$color <- ifelse(x$rejected, ifelse(sign(x$p.val) == 1, "red", "green"), "gray"))
tree.f$Do(function(x) x$name <- x$acronym)

tree.f$Do(function(x) x$tooltip <- sprintf("Region: %s<br>p: %0.2f", x$Name, x$p.val))


#treeList <- ToListExplicit(tree.f, unname = TRUE)
#radialNetwork(treeList)
#install.packages('htmltools')
c = collapsibleTree(tree.f,
                    fill = "color", zoomable = FALSE, width = 1000, attribute = "name",
                    height = 800, tooltip = TRUE, tooltipHtml = "tooltip", collapsed = TRUE
)
#htmltools::save_html(c, file='//10.159.50.7/LabCommon/Ken/data/Opioid_cFos/result/TreeFDR_GLMSaline.html')
#htmlwidgets::saveWidget(c, file='//10.159.50.7/LabCommon/Ken/data/Opioid_cFos/result/TreeFDR_SvA2.html')
# Assuming 'var' is the variable that determines the filename
# 'var' should be a character string
filename <- paste0(output_path,"/TreeFDRF_",var,'_',save_key, ".html")

# Saving the HTML file
htmltools::save_html(c, file = filename)

# save result as table
table.f = ToDataFrameTree(tree.f,'acronym','parent_acronym','rejected','q_adj','p.val')
table.f.path = paste0(output_path,"/TreeFDRF_", var,'_',save_key, ".csv")
table.s = ToDataFrameTree(tree.s,'acronym','parent_acronym','rejected','q_adj','p.val')
table.s.path = paste0(output_path,"/TreeFDRS_", var,'_',save_key, ".csv")
write.csv(table.f,table.f.path, row.names = FALSE)
write.csv(table.s,table.s.path, row.names = FALSE)