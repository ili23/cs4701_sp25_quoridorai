import numpy as np
import matplotlib.pyplot as plt


def render_data(path, ax, title, legend=False):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)


    x = np.unique(data[:, 1])
    y_losses = np.array([np.sum(data[(data[:, 1] == v), 3] == 1) for v in x])
    y_draws = np.array([np.sum(data[(data[:, 1] == v), 3] == 2) for v in x])
    y_wins = np.array([np.sum(data[(data[:, 1] == v), 3] == 0) for v in x])

    # Convert to percentages
    total_games = y_wins + y_draws + y_losses
    y_wins_pct = y_wins / total_games
    y_draws_pct = y_draws / total_games
    y_losses_pct = y_losses / total_games

    ax.plot(x, y_wins_pct, color="tab:green")
    ax.fill_between(x, y_wins_pct, 0, color='tab:green', alpha=0.5, label="Wins")

    ax.plot(x, y_wins_pct + y_draws_pct, color="tab:blue")
    ax.fill_between(x, y_wins_pct + y_draws_pct, y_wins_pct, color='tab:blue', alpha=0.5, label="Draws")

    ax.plot(x, y_wins_pct + y_draws_pct + y_losses_pct, color="tab:orange")
    ax.fill_between(x, y_wins_pct + y_draws_pct + y_losses_pct, y_wins_pct + y_draws_pct, color='tab:orange', alpha=0.5, label="Losses")

    ax.set_title(title, fontweight="bold")
    if legend:
        ax.legend()
    ax.set_xlabel("Player 1 MCTS Iterations")
    ax.set_ylabel("Ratio of games")
    ax.set_xscale('log')  # Set x-axis to logarithmic scale

    ax.set_yticks((0, 0.5, 1))

fig, axs = plt.subplots(2, 2, figsize=(12, 3.75 * 2))


render_data("build/naive_vs_naive.csv", axs[0,0],"(A) Naive", True)
render_data("build/basic_vs_naive.csv", axs[0,1], "(B) Basic")

plt.tight_layout()
plt.savefig("../report/win_rates.png", dpi=400)
