import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import heapq

import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from copy import deepcopy


board_size = 5

pawn1_pos = (2, 4)
pawn2_pos = (2, 0)
fences = [
    (2, 0, "h"),
    (1, 0, "h"),
    (2, 0, "h"),
    (2, 1, "v"),
    (2, 2, "v"),
    (3, 3, "v"),
    (3, 4, "v"),
]

# pawn1_pos = (1, 2)
# pawn2_pos = (4, 1)

# fences = [
#     (0, 3, "v"),
#     (0, 4, "v"),
#     (1, 2, "v"),
#     (1, 3, "v"),
#     (2, 3, "v"),
#     (2, 4, "v"),
#     (3, 1, "v"),
#     (3, 3, "h"),
#     (3, 2, "v"),
# ]

# Step 1: Build graph with weights
graph = defaultdict(list)


def set_edge_weight(a, b, new_weight):
    for i, (neighbor, weight) in enumerate(graph[a]):
        if neighbor == b:
            graph[a][i] = (b, new_weight)
    for i, (neighbor, weight) in enumerate(graph[b]):
        if neighbor == a:
            graph[b][i] = (a, new_weight)


def add_edge_weight(a, b, new_weight):
    for i, (neighbor, weight) in enumerate(graph[a]):
        if neighbor == b:
            graph[a][i] = (b, weight + new_weight)
    for i, (neighbor, weight) in enumerate(graph[b]):
        if neighbor == a:
            graph[b][i] = (a, weight + new_weight)


for row in range(board_size):
    for col in range(board_size):
        node = (row, col)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                neighbor = (nr, nc)
                weight = 1  # default weight
                graph[node].append((neighbor, weight))


def remove_edge(a, b):
    graph[a] = [(n, w) for n, w in graph[a] if n != b]
    graph[b] = [(n, w) for n, w in graph[b] if n != a]


for row, col, orientation in fences:
    if orientation == "h":
        a = (row, col)
        b = (row, col + 1)
    elif orientation == "v":
        a = (row, col)
        b = (row + 1, col)
    remove_edge(a, b)

# Step 2: Dijkstra's shortest path by weight
def dijkstra_path(start, target="top"):
    visited = {}
    heap = [(0, start, [])]  # (cost, current, path)

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current[0] == (board_size - 1 if target == "top" else 0):  # Goal row
            return path + [current]
        if current in visited:
            continue
        visited[current] = cost
        for neighbor, weight in graph[current]:
            if neighbor not in visited:
                heapq.heappush(heap, (cost + weight, neighbor, path + [current]))
    return []


def add_weight_to_path_edges(path, delta):
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        # Update a to b
        for j, (neighbor, weight) in enumerate(graph[a]):
            if neighbor == b:
                graph[a][j] = (b, weight + delta)
                break
        # Update b to a (since the graph is undirected)
        for j, (neighbor, weight) in enumerate(graph[b]):
            if neighbor == a:
                graph[b][j] = (a, weight + delta)
                break


graph_original = deepcopy(graph)
for _ in range(25):
    path1 = dijkstra_path(pawn1_pos)
    add_weight_to_path_edges(path1, 2)

graph_path1 = deepcopy(graph)
graph = deepcopy(graph_original)

for _ in range(25):
    path2 = dijkstra_path(pawn2_pos, target="bottom")
    add_weight_to_path_edges(path2, 2)

graph_path2 = deepcopy(graph)

# Draw pawns
def draw_pawn(pos, ax, color="black", **kwargs):
    row, col = pos
    circ = patches.Circle(
        (col + 0.5, row + 0.5),
        0.18,
        facecolor=color,
        zorder=10,
        edgecolor="black",
        **kwargs
    )
    ax.add_patch(circ)


def render(
    ax,
    title_text,
    graph_input,
    pawns_to_render,
    colorbar=False,
    fences=fences,
    pawn1_pos=pawn1_pos,
    pawn2_pos=pawn2_pos,
    dotted_fences=[],
):
    ax.set_xlim(-0.05, board_size + 0.05)
    ax.set_ylim(-0.05, board_size + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(title_text, fontweight="bold")

    # Draw squares
    for row in range(board_size):
        for col in range(board_size):
            color = "lightgrey" if (row + col) % 2 else "white"
            rect = patches.Rectangle((col, row), 1, 1, facecolor=color)
            ax.add_patch(rect)

    rect = patches.Rectangle(
        (0, 0), board_size, board_size, linewidth=1, edgecolor="black", facecolor="none"
    )
    ax.add_patch(rect)

    # Draw fences
    def draw_fence(row, col, orientation, color="black", dotted=False, **kwargs):
        if orientation == "v":
            # Fence between (row, col) and (row, col+1)
            x = col
            y = row + 0.95
            rect = patches.Rectangle((x, y), 1, 0.1, color=color, zorder=10, **kwargs)

            if dotted:
                ax.plot((x, x + 2), (y, y), color=color, zorder=10, **kwargs)
        else:  # 'v'
            # Fence between (row, col) and (row+1, col)
            x = col + 0.95
            y = row
            rect = patches.Rectangle((x, y), 0.1, 1, color=color, zorder=10, **kwargs)

            if dotted:
                ax.plot((x, x), (y, y+2), color=color, zorder=10, **kwargs)

        if not dotted:
            ax.add_patch(rect)


    for row, col, orientation in fences:
        draw_fence(row, col, orientation)

    for row, col, orientation in dotted_fences:
        draw_fence(row, col, orientation, color="tab:orange", alpha=1, linewidth=4, dotted=True, linestyle=(0, (2, 1)))

    # Draw labels
    # for col in range(board_size):
    #     ax.text(col + 0.5, -0.6, str(col), ha="center", va="top", fontsize=10)
    # for row in range(board_size):
    #     ax.text(-0.6, row + 0.5, str(row), ha="right", va="center", fontsize=10)

    # Draw paths
    def draw_path(path, color):
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            ax.plot(
                [c1 + 0.5, c2 + 0.5], [r1 + 0.5, r2 + 0.5], color=color, linewidth=2
            )

    if 1 in pawns_to_render:
        draw_pawn(pawn1_pos, ax, color="tab:red")

    if 2 in pawns_to_render:
        draw_pawn(pawn2_pos, ax, color="tab:green")

    # draw_path(path1, 'blue')
    # draw_path(path2, 'green')

    if graph_input is not None:
        # Step 1: Collect all weights
        all_weights = [
            weight for node in graph for (neighbor, weight) in graph_input[node]
        ]
        min_w = min(all_weights)
        max_w = max(all_weights)

        # Step 2: Normalize and get colormap
        norm = mcolors.Normalize(vmin=min_w, vmax=max_w)
        cmap = cm.get_cmap("viridis")

        # Step 3: Draw colored edges
        drawn_edges = set()
        for node in graph_input:
            for neighbor, weight in graph_input[node]:
                edge = tuple(sorted([node, neighbor]))
                if edge in drawn_edges:
                    continue
                drawn_edges.add(edge)

                r1, c1 = node
                r2, c2 = neighbor
                color = cmap(norm(weight))
                ax.plot(
                    [c1 + 0.5, c2 + 0.5], [r1 + 0.5, r2 + 0.5], color=color, linewidth=3
                )

        # Step 4: Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy data for the colorbar
        if colorbar:
            cbar = plt.colorbar(sm, ax=colorbar, fraction=1.5, pad=2)
            # cbar = plt.colorbar(sm, fraction=0.046,pad=2, location='left', ax=ax, shrink=00.5)
            # cbar.ax.set_position([0.85, 0.3, 0.03, 0.4])
            cbar.set_label("Edge Weight", labelpad=-20)
            cbar.set_ticks((min_w, max_w))
            cbar.ax.set_yticklabels(["Min", "Max"])

    # Plot small red dots
    # ax.scatter(xx_flat, yy_flat, s=0.01, color='red')  # s=2 is very small dot size
    if graph_input is None:
        rect = patches.Rectangle(
            (0, 0),
            5,
            1,
            hatch="//",
            fill=False,
            edgecolor="tab:green",
            alpha=0.8,
            linestyle="none",
        )
        rect2 = patches.Rectangle(
            (0, 4),
            5,
            1,
            hatch="//",
            fill=False,
            edgecolor="tab:red",
            alpha=0.8,
            linestyle="none",
        )
        ax.add_patch(rect)
        ax.add_patch(rect2)


final_graph = {}

for start in graph:
    final_graph[start] = []
    for (dest1, weight1), (dest2, weight2) in zip(
        graph_path1[start], graph_path2[start]
    ):
        assert dest1 == dest2
        final_graph[start].append((dest1, weight1 - weight2))


# Step 3: Draw board
fig, axs = plt.subplots(2, 3, figsize=(12, 3.75 * 2))



# PUT DATA ABOUT DISTRIBUTION HERE


def render_hist(ax, path, title, legend=False, winning_moves=(17,)):
    data = np.genfromtxt(path, delimiter=",")

    def softmax(x):
        exps = np.exp(x - np.max(x))  # subtract max for numerical stability
        return exps / np.sum(exps)

    data1 = data[data[:, 1] == 200, 2:]
    data1 = data1[:, 0] / data1[:, 1]

    data1 = softmax(data1)

    data2 = data[data[:, 1] == 5000, 2:]
    data2 = data2[:, 0] / data2[:, 1]

    data2 = softmax(data2)
    # Histogram settings
    bins = np.linspace(0, 1, data1.shape[0])
    width = data1.shape[0] * 0.4  # width of each bar

    x = np.arange(data1.shape[0])  # x positions, like 0, 1, 2, ...

    bar_width = 0.4  # Controls spacing between bars

    ax.bar(
        x - bar_width / 2,
        data1,
        width=bar_width,
        color="tab:blue",
        label="$200$ iters.",
    )
    ax.bar(
        x + bar_width / 2,
        data2,
        width=bar_width,
        color="tab:orange",
        label="$5000$ iters.",
    )

    ax.set_title(title, fontweight="bold")

    ax.set_ylabel("Softmax density", labelpad=0)

    ax.set_xticks(winning_moves)
    ax.set_xticklabels(["*"] * len(winning_moves))
    ax.set_xlabel("Legal moves", labelpad=-5)

    ax.set_ylim((0, 0.2))
    ax.set_yticks((0, 0.1, 0.2))
    ax.legend(loc="upper left")


# Optional: custom x-axis labels
# plt.xticks(x, ['A', 'B', 'C', 'D'])  # or just use range if unnamed


def graph_eval_data(ax, path, title, lines=False):

    eval_data = np.genfromtxt(path, delimiter=",")[:5000, :]

    ax.plot(-(eval_data[:, 1] / eval_data[:, 2]), color="tab:purple")

    ax.set_title(title, fontweight="bold")
    ax.set_xticks((0, 2500, 5000))
    ax.set_yticks((-1, 0, 1))
    if lines:
        ax.axvline(200, linestyle="dashed", color="tab:blue")
        ax.axvline(5000, linestyle="dashed", color="tab:orange")
    ax.set_xlabel("MCTS Iterations")
    ax.set_ylabel("Strength", labelpad=-3)


if __name__ == "__main__":
    render(axs[0, 0], "(A) Simple tactic", None, (1, 2))

    render_hist(
        axs[0, 1],
        "Figure_Data/MCTS_Distr/b_naive_move_distr.csv",
        "(B) Naive move evaluations",
        True,
    )
    render_hist(
        axs[0, 2],
        "Figure_Data/MCTS_Distr/e_forest_move_distr.csv",
        "(C) Forest move evaluations",
        False,
    )

    graph_eval_data(
        axs[1, 0],
        "Figure_Data/MCTS_Distr/c_naive_eval_over_time.csv",
        "(D) Naive position evaluation",
        True,
    )
    graph_eval_data(
        axs[1, 1],
        "Figure_Data/MCTS_Distr/d_basic_eval_over_time.csv",
        "(E) Basic position evaluation",
    )
    graph_eval_data(
        axs[1, 2],
        "Figure_Data/MCTS_Distr/f_forest_eval_over_time.csv",
        "(F) Forest position evaluation",
        True,
    )

    plt.tight_layout()

    plt.show()
    # plt.savefig("../report/MCTS_eval.png", dpi=400)
