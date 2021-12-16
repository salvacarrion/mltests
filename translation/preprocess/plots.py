import os
from pathlib import Path

# Sort of incompatible with plotly
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import gc

from translation.preprocess import utils

# If this is not set, matplotlib won't most of the images that were created a loop
# Source: https://github.com/matplotlib/matplotlib/issues/8519
matplotlib.use('agg')


def get_tokens_by_sentence(filename):
    with open(filename, 'r') as f:
        token_sizes = [len(line.strip().split(' ')) for line in f.readlines()]
    return token_sizes

#
# def histogram(data, output_dir, fname, title="", legend_title=None, labels=None, nbins=20, bargap=0.2,
#                show_fig=False, save_fig=True, formats=None):
#     import plotly.express as px
#     import plotly.graph_objects as go
#
#     # Fix mathjax issue
#     import plotly.io as pio
#     pio.kaleido.scope.mathjax = None
#
#     if formats is None:
#         formats = ["png", "pdf"]
#
#     fig = px.histogram(data, nbins=nbins, title=title, labels=labels)
#     fig.update_layout(bargap=bargap, legend_orientation='h', title=dict(x=0.5, font=dict(size=18)),
#                       legend_title=legend_title)
#
#     # Save image
#     if save_fig:
#         for ext in formats:
#             # Create png/pdf/... dirs
#             save_dir = os.path.join(output_dir, ext)
#             Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#             # Save image
#             fig.write_image(os.path.join(save_dir, f"{fname}.{ext}"))


def catplot(data, x, y, hue, title, xlabel, ylabel, leyend_title, output_dir, fname, aspect_ratio=(12, 8), size=1.0,
            show_values=True, dpi=150, show_fig=False, save_fig=True, formats=None):
    def fn_format(x, idx=None):
        return utils.human_format(int(x), decimals=0)

    if formats is None:
        formats = ["png", "pdf"]

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
    sns.set(font_scale=size)

    # Plot catplot
    g = sns.catplot(data=data, x=x, y=y, hue=hue, kind="bar", legend=False, height=aspect_ratio[1], aspect=aspect_ratio[0]/aspect_ratio[1])

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # g.tick_params(axis='x', which='major', labelsize=8*size)
    # g.tick_params(axis='y', which='major', labelsize=8*size)
    g.axes.flat[0].yaxis.set_major_formatter(fn_format)  # only for catplot

    # Add values
    if show_values:
        ax = g.facet_axis(0, 0)
        for c in ax.containers:
            labels = [int(v.get_height()) for v in c]
            ax.bar_label(c, labels=labels, label_type='edge', fontsize=8*size)

    # properties
    g.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.legend(title=leyend_title, loc='upper right')
    plt.tight_layout()

    # Save image
    if save_fig:
        for ext in formats:
            # Create png/pdf/... dirs
            save_dir = os.path.join(output_dir, ext)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Save image
            plt.savefig(os.path.join(save_dir, f"{fname}.{ext}"), dpi=dpi)

    # Show plot
    if show_fig:
        plt.show()

    # Close figure
    plt.close(fig)
    plt.close()


def barplot(data, x, y, output_dir, fname, title="", xlabel="x", ylabel="y", aspect_ratio=(12, 8), size=1.0,
            dpi=150, show_fig=False, save_fig=True, formats=None):
    def fn_format(x, idx=None):
        return utils.human_format(int(x), decimals=0)

    if formats is None:
        formats = ["png", "pdf"]

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
    sns.set(font_scale=size)

    # Plot barplot
    g = sns.barplot(data=data, x=x, y=y)

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.tick_params(axis='x', which='major', labelsize=8)  # *size  => because of the vocabulary distribution
    g.tick_params(axis='y', which='major', labelsize=8)  # *size  => because of the vocabulary distribution
    g.yaxis.set_major_formatter(fn_format)

    # properties
    plt.title(title)
    plt.tight_layout()

    # Save image
    if save_fig:
        for ext in formats:
            # Create png/pdf/... dirs
            save_dir = os.path.join(output_dir, ext)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Save image
            plt.savefig(os.path.join(save_dir, f"{fname}.{ext}"), dpi=dpi)

    # Show plot
    if show_fig:
        plt.show()

    # Close figure
    plt.close(fig)
    plt.close()


def histogram(data, x, output_dir, fname, title="", xlabel="x", ylabel="y", bins="auto", aspect_ratio=(12, 8), size=1.0,
              dpi=150, show_fig=False, save_fig=True, formats=None):
    def fn_format(x, idx=None):
        return utils.human_format(int(x), decimals=0)

    if formats is None:
        formats = ["png", "pdf"]

    # Create subplot
    fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
    sns.set(font_scale=size)

    # Plot barplot
    g = sns.histplot(data=data, x=x, bins=bins)

    # Tweaks
    g.set(xlabel=xlabel, ylabel=ylabel)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.tick_params(axis='x', which='major', labelsize=8*size)
    g.tick_params(axis='y', which='major', labelsize=8*size)
    g.yaxis.set_major_formatter(fn_format)

    # properties
    plt.title(title)
    plt.tight_layout()

    # Save image
    if save_fig:
        for ext in formats:
            # Create png/pdf/... dirs
            save_dir = os.path.join(output_dir, ext)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Save image
            plt.savefig(os.path.join(save_dir, f"{fname}.{ext}"), dpi=dpi)

    # Show plot
    if show_fig:
        plt.show()

    # Close figure
    plt.close(fig)
    plt.close()
