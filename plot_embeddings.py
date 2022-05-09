import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import transformers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util import set_seeds


def main():
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(description='Generate t-SNE plot of model input embeddings.')
    parser.add_argument('chkpt_dir', type=str, help='Model checkpoint directory')
    parser.add_argument('--vocab_file', type=str, default='./vocab.txt', help='Vocabulary file for model')
    parser.add_argument('--output_dir', type=str, default=f'output_dir/embeddings_{now}')
    args = parser.parse_args()

    # Reproducibility
    set_seeds()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    model = transformers.GPT2LMHeadModel.from_pretrained(args.chkpt_dir)
    vocab_file = args.vocab_file
    tokenizer = transformers.BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)

    # Get weight-tied embeddings
    embeddings = model.get_input_embeddings()
    aa_emb_wts = embeddings.weight[5:25].detach().cpu().numpy()  # skip special tokens
    aa_labels = tokenizer.decode(np.arange(len(embeddings.weight))).split(' ')[5:25]

    # Plot projections and save to output directory
    axis_label_dict = {'tsne': 't-SNE',
                       'pca': 'PCA'}

    for projection in ['tsne', 'pca']:
        x_label = "{} X".format(axis_label_dict[projection])
        y_label = "{} Y".format(axis_label_dict[projection])

        # Compute projection and get dataframe with residue annotations
        sabdab_res_df = get_sabdab_residue_dataframe(aa_emb_wts, aa_labels, projection, x_label, y_label)

        # Plot figure
        plt.subplots(figsize=(9, 4))
        ax = sns.scatterplot(data=sabdab_res_df,
                             x=x_label,
                             y=y_label,
                             hue="Hydropathy",
                             style="Residue",
                             markers=[r"$\bf {}$".format(aa) for aa in aa_labels],
                             s=200,
                             legend='full')

        # Get hydropathy legend only
        handles, labels = ax.get_legend_handles_labels()
        num_hydropathy = len(np.unique(sabdab_res_df['Hydropathy']))
        ax.legend(handles=handles[1:num_hydropathy + 1], labels=labels[1:num_hydropathy + 1], bbox_to_anchor=(1.05, 1),
                  loc='upper left', fontsize=12)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/embeddings_{projection}.pdf', dpi=600, transparent=True)
        plt.show()


def get_sabdab_residue_dataframe(aa_emb_wts, aa_labels, projection, x_label, y_label, tsne_seed=3):
    """
    Computes projection on aa_emb_wts and returns labeled dataframe for plotting.

    aa_emb_wts - embedding layer weights
    aa_labels - residue names in order of aa_emb_wts
    projection - dim reduction technique to use. Supports one of ['tsne', 'pca']
    x_label - x-axis label
    y_label - y-axis label
    """
    if projection == 'tsne':
        proj = TSNE(n_components=2, perplexity=2, n_jobs=12, random_state=tsne_seed)
    elif projection == 'pca':
        proj = PCA(n_components=2)
    else:
        assert False, 'Projection {} is not supported.'.format(projection)

    aa_emb_proj = proj.fit_transform(aa_emb_wts)

    if np.array(aa_labels).ndim == 1:
        aa_labels = np.array(aa_labels)[..., np.newaxis]

    sabdab_res_data = np.concatenate([aa_labels, aa_emb_proj], axis=1)
    sabdab_res_df = pd.DataFrame(sabdab_res_data, columns=["Residue", x_label, y_label])
    sabdab_res_df = sabdab_res_df.astype({
        "Residue": str,
        x_label: float,
        y_label: float
    })

    sabdab_res_df["Hydropathy"] = "Special"
    for res in ["A", "I", "L", "M", "V"]:
        sabdab_res_df.loc[sabdab_res_df["Residue"] == res,
                          "Hydropathy"] = "Hydrophobic (aliphatic)"
    for res in ["F", "W", "Y"]:
        sabdab_res_df.loc[sabdab_res_df["Residue"] == res,
                          "Hydropathy"] = "Hydrophobic (aromatic)"
    for res in ["N", "Q", "S", "T", "X"]:
        sabdab_res_df.loc[sabdab_res_df["Residue"] == res,
                          "Hydropathy"] = "Polar neutral"
    for res in ["H", "K", "R"]:
        sabdab_res_df.loc[sabdab_res_df["Residue"] == res,
                          "Hydropathy"] = "Positive"
    for res in ["D", "E"]:
        sabdab_res_df.loc[sabdab_res_df["Residue"] == res,
                          "Hydropathy"] = "Negative"

    return sabdab_res_df


if __name__ == '__main__':
    main()
