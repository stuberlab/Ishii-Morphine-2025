# Fig2_TreeBH.R — BRANCH (hierarchical sFDR) for Figure 2
#
# Input : <DATA_DIR>/Figure2_C_glm_stat_df.csv   (per-region GLM p-values from Fig2_BRANCH.py)
# Output: <DATA_DIR>/TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv   (Simes aggregation = BRANCH)
#         <DATA_DIR>/TreeFDRF_pvalue_Figure2_C_glm_stat_df.csv   (Fisher aggregation, for comparison)
#
# Set DATA_DIR to the "01_main_cfos_morphine" folder of the Figshare deposit.

library(data.tree)
library(TreeBH)
library(tidyr)

DATA_DIR <- Sys.getenv("OPIOID_DATA_ROOT", unset = ".")
DATA_DIR <- file.path(DATA_DIR, "01_main_cfos_morphine")  # adjust if running standalone

save_key <- "Figure2_C_glm_stat_df"
var <- "pvalue"
qvalue_threshold <- 0.01

ken <- read.csv(file.path(DATA_DIR, paste0(save_key, ".csv")))
dat <- drop_na(ken)
dat$t.p.val <- dat[[var]]

## build tree from the ontology (acronym / parent_acronym network)
tree <- FromDataFrameNetwork(dat)
if (length(tree$children) == 1) {        # slice off false root
  tree <- tree$children[[1]]
  tree$parent <- NULL
}

## leaf p-values, then aggregate up (Simes for BRANCH; Fisher for comparison)
tree$Do(function(node) node$p.val <- node$t.p.val, filterFun = isLeaf)
tree.s <- Clone(tree)
tree.f <- Clone(tree)
tree.s$Do(function(node) node$p.val <- Aggregate(node, attribute = "p.val", aggFun = TreeBH::get_simes_p))
tree.f$Do(function(node) node$p.val <- Aggregate(node, attribute = "p.val", aggFun = TreeBH::get_fisher_p))

L <- tree$height
qs <- rep(qvalue_threshold, L)
tree.s$Do(function(node) node$q_l <- qs[node$level])
tree.f$Do(function(node) node$q_l <- qs[node$level])

## recursive hierarchical sFDR (TreeBH)
FamilyCheckR <- function(node) {
  if (node$isRoot) node$q_adj <- 1
  q_target <- node$q_adj * node$q_l
  children <- node$children
  child_ps <- sapply(children, function(x) x$p.val)
  child_ps_adj <- p.adjust(child_ps, method = "BH")
  purrr::walk2(children, child_ps_adj, function(node, p_adj) node$rejected <- p_adj < q_target)
  prop <- mean(purrr::map_lgl(children, function(x) x$rejected))
  purrr::walk(children, function(x) x$q_adj <- node$q_adj * prop)
  recursep <- function(x) !is.na(x$rejected) && x$rejected && !x$isLeaf
  purrr::walk(children, function(x) if (recursep(x)) FamilyCheckR(x))
}
tree.s$rejected <- TRUE; FamilyCheckR(tree.s)
tree.f$rejected <- TRUE; FamilyCheckR(tree.f)

## write result tables
write.csv(ToDataFrameTree(tree.s, "acronym", "parent_acronym", "rejected", "q_adj", "p.val"),
          file.path(DATA_DIR, paste0("TreeFDRS_", var, "_", save_key, ".csv")), row.names = FALSE)
write.csv(ToDataFrameTree(tree.f, "acronym", "parent_acronym", "rejected", "q_adj", "p.val"),
          file.path(DATA_DIR, paste0("TreeFDRF_", var, "_", save_key, ".csv")), row.names = FALSE)
