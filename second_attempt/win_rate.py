import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(1, 4, figsize=(16, 3.75))


def render(ax, path, title, legend=False):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)

    x = np.unique(data[:, 1])
    y_losses = np.array([np.sum(data[(data[:, 1] == v), 3] == 1) for v in x])
    y_draws = np.array([np.sum(data[(data[:, 1] == v), 3] == 2) for v in x])
    y_wins = np.array([np.sum(data[(data[:, 1] == v), 3] == 0) for v in x])

    # Convert to percentages
    total_games = y_wins + y_draws + y_losses
    y_wins_pct = y_wins / total_games * 100
    y_draws_pct = y_draws / total_games * 100
    y_losses_pct = y_losses / total_games * 100

    ax.plot(x, y_wins_pct, color="tab:orange", label="Wins")
    ax.fill_between(x, y_wins_pct, 0, color='tab:orange', alpha=0.5)

    ax.plot(x, y_wins_pct + y_draws_pct, color="tab:blue", label="Draws")
    ax.fill_between(x, y_wins_pct + y_draws_pct,
                    y_wins_pct, color='tab:blue', alpha=0.5)

    ax.plot(x, y_wins_pct + y_draws_pct + y_losses_pct,
            color="tab:green", label="Losses")
    ax.fill_between(x, y_wins_pct + y_draws_pct + y_losses_pct,
                    y_wins_pct + y_draws_pct, color='tab:green', alpha=0.5)

    ax.set_title(title, fontweight="bold")
    if legend:
        ax.legend()

    ax.set_xlabel("MCTS Iterations")
    ax.set_ylabel("Percentage of Games (%)", labelpad=-3)
    ax.set_yticks((0, 50, 100))
    ax.set_xscale('log')


render(axs[0], "Figure_data/Win_rate/naive.csv", "(A) Naive")
render(axs[1], "Figure_data/Win_rate/basic.csv", "(B) Basic")
render(axs[2], "Figure_data/Win_rate/cnn.csv", "(C) CNN")
render(axs[3], "Figure_data/Win_rate/tree.csv", "(D) Random forest")
plt.tight_layout()
# plt.show()

plt.savefig("../report/win_rates.png", dpi=400)
