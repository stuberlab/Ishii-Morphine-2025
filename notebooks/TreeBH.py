import pandas as pd
import numpy as np
import networkx as nx
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues
import plotly.graph_objects as go
import os

# ----------------------------- #
#       Utility Functions       #
# ----------------------------- #

def load_data(filepath, pval_col='pvalue'):
    df = pd.read_csv(filepath).dropna()
    df['t.p.val'] = df[pval_col]
    return df

def build_tree(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["acronym"], **row)
        if pd.notna(row["parent_acronym"]):
            G.add_edge(row["parent_acronym"], row["acronym"])
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(roots) != 1:
        raise ValueError("Expected a single root node.")
    return G, roots[0]

def get_simes_p(pvals):
    pvals = np.sort(pvals)
    return np.minimum.accumulate(len(pvals) * pvals / (np.arange(1, len(pvals)+1))).min()

def get_fisher_p(pvals):
    return combine_pvalues(pvals, method='fisher')[1]

def propagate_pvals(G, root, agg_func):
    for node in nx.dfs_postorder_nodes(G, source=root):
        children = list(G.successors(node))
        if children:
            pvals = [G.nodes[c].get("p.val") for c in children if G.nodes[c].get("p.val") is not None]
            if pvals:
                G.nodes[node]["p.val"] = agg_func(pvals)
        else:
            G.nodes[node]["p.val"] = G.nodes[node].get("t.p.val")

def assign_levels_and_q(G, root, q_thresh):
    for node in G.nodes:
        level = nx.shortest_path_length(G, source=root, target=node)
        G.nodes[node]["level"] = level
        G.nodes[node]["q_l"] = q_thresh

def family_check(G, root):
    G.nodes[root]["q_adj"] = 1.0

    def recurse(n):
        children = list(G.successors(n))
        if not children:
            return
        child_pvals = [G.nodes[c]["p.val"] for c in children]
        _, adj_pvals, _, _ = multipletests(child_pvals, method="fdr_bh")
        for c, adj_p in zip(children, adj_pvals):
            q_target = G.nodes[n]["q_adj"] * G.nodes[c]["q_l"]
            G.nodes[c]["rejected"] = adj_p < q_target
        prop_rejected = np.mean([G.nodes[c]["rejected"] for c in children])
        for c in children:
            G.nodes[c]["q_adj"] = G.nodes[n]["q_adj"] * prop_rejected
        for c in children:
            if G.nodes[c]["rejected"] and list(G.successors(c)):
                recurse(c)

    recurse(root)

def extract_tree_results(G):
    rows = []
    for node in G.nodes:
        n = G.nodes[node]
        parent = list(G.predecessors(node))
        rows.append({
            "acronym": node,
            "parent_acronym": parent[0] if parent else None,
            "rejected": n.get("rejected"),
            "q_adj": n.get("q_adj"),
            "p.val": n.get("p.val")
        })
    return pd.DataFrame(rows)

def plot_tree(G, output_file):
    df = extract_tree_results(G)
    df["color"] = df["rejected"].map({True: "red", False: "gray"})
    fig = go.Figure(go.Sunburst(
        labels=df["acronym"],
        parents=df["parent_acronym"],
        values=np.ones(len(df)),
        marker=dict(colors=df["color"]),
        hovertext=["p = {:.3g}".format(p) if pd.notnull(p) else "NA" for p in df["p.val"]],
        hoverinfo="label+text"
    ))
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig.write_html(output_file)

# ----------------------------- #
#     Main Pipeline Function    #
# ----------------------------- #

def run_tree_fdr_pipeline(
    input_csv,
    output_dir,
    pval_col="pvalue",
    save_key="result",
    q_thresh=0.001,
    plot_html=True
):
    df = load_data(input_csv, pval_col)
    G, root = build_tree(df)

    for method, agg_func, suffix in [
        ("simes", get_simes_p, "S"),
        ("fisher", get_fisher_p, "F")
    ]:
        G_copy = G.copy()
        propagate_pvals(G_copy, root, agg_func)
        assign_levels_and_q(G_copy, root, q_thresh)
        G_copy.nodes[root]["rejected"] = True
        family_check(G_copy, root)

        df_out = extract_tree_results(G_copy)
        csv_path = os.path.join(output_dir, f"TreeFDR{suffix}_{pval_col}_{save_key}.csv")
        df_out.to_csv(csv_path, index=False)

        if plot_html:
            html_path = os.path.join(output_dir, f"TreeFDR{suffix}_{pval_col}_{save_key}.html")
            plot_tree(G_copy, html_path)

# ----------------------------- #
#          Run Example          #
# ----------------------------- #

if __name__ == "__main__":
    input_csv = r"G:/My Drive/Opioid_whole_brain_manuscript/result/Figure2_C_glm_stat_df_no_batch.csv"
    output_dir = r"G:/My Drive/Opioid_whole_brain_manuscript/result"
    run_tree_fdr_pipeline(
        input_csv,
        output_dir,
        pval_col="pvalue",
        save_key="Figure2_C_glm_stat_df_no_batch",
        q_thresh=0.001,
        plot_html=True  # Set to False to skip HTML plot
    )
