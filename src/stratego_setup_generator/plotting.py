import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .constants import PIECE_TO_STR, NUM_SETUP_ROWS, NUM_SETUP_COLUMNS, NUM_SETUP_SQUARES


def plot_setup(setup, tile_size=0.6, font_size=15, piece_labels=PIECE_TO_STR, ax=None):

    lakes = [(-1, 2), (-1, 3), (-1, 6), (-1, 7)]

    if ax is None:
        fig_size = (NUM_SETUP_COLUMNS * tile_size, NUM_SETUP_ROWS * tile_size)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    ax.set_xlim(0, NUM_SETUP_COLUMNS)
    ax.set_ylim(0, NUM_SETUP_ROWS + 1)

    for row, col in lakes:
        rect = patches.Rectangle((col, NUM_SETUP_ROWS - row - 1), width=1, height=1,
                                 linewidth=1, edgecolor='none', facecolor='lightblue', alpha=0.5)
        ax.add_patch(rect)

    setup = np.atleast_2d(setup)

    for square in range(setup.shape[1]):
        row, column = divmod(square, NUM_SETUP_COLUMNS)
        ax.text(column + 0.5, NUM_SETUP_ROWS - row - 0.5, piece_labels[setup[0, square]],
                ha='center', va='center', fontsize=font_size)

    ax.set_xticks(range(NUM_SETUP_COLUMNS + 1))
    ax.set_yticks(range(NUM_SETUP_ROWS + 1))
    ax.grid(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def create_plot_grid(num_plots, num_columns, subplot_width, subplot_height):
    num_rows = int(np.ceil(num_plots / num_columns))
    fig_size = (num_columns * subplot_width, num_rows * subplot_height)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=fig_size, squeeze=False)
    return fig, axes


def plot_setups(setups, num_columns=3, tile_size=0.6, font_size=15,
                piece_labels=PIECE_TO_STR, w_space=0.1, h_space=0.1):

    fig, axes = create_plot_grid(len(setups), num_columns,
                                 NUM_SETUP_COLUMNS * tile_size, NUM_SETUP_ROWS * tile_size)
    fig.subplots_adjust(wspace=w_space, hspace=h_space)
    axes = axes.ravel()

    for setup, ax in zip(setups, axes):
        plot_setup(setup, tile_size=tile_size, font_size=font_size,
                   piece_labels=piece_labels, ax=ax)

    for i in range(len(setups), len(axes)):
        axes[i].axis('off')

    plt.show()


def plot_setup_generation(setup, distributions, width=4, height=2, adjust_y_lim=False):
    fig, axes = create_plot_grid(2*(NUM_SETUP_SQUARES + 1), 2, width, height)
    setup = setup.squeeze()
    for square in range(NUM_SETUP_SQUARES):
        plot_setup(setup[:square], ax=axes[square][0])
        axes[square][1].set_title("Probability distribution for next piece")
        axes[square][1].bar(PIECE_TO_STR.values(), distributions[square])
        if not adjust_y_lim:
            axes[square][1].set_ylim(0.0, 1.0)
    plot_setup(setup, ax=axes[NUM_SETUP_SQUARES][0])
    axes[NUM_SETUP_SQUARES][1].axis('off')
    fig.tight_layout()
    plt.show()