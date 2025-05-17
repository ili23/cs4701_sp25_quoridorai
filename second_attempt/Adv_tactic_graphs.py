from MCTS_graphs import *

pawn1_pos = (2, 1)
pawn2_pos = (1, 4)

fences = [
    (0, 2, "v"),
    (2, 2, "v"),
    (0, 3, "v"),
    (2, 3, "v"),
    (1, 3, "h"),
    (2, 3, "h"),
    (1, 1, "h"),
    (2, 1, "h"),
]

dotted_fences = [
    (2, 0, "v"),
    # (2, 1, "v"),
    (3, 2, "h"),
    (3, 3, "h"),
    (3, 1, "h"),
    # (4, 2, "h"),
    # (4, 3, "h"),
    # (4, 1, "h"),
]

winning_moves = [6, 8, 10, 11]

if __name__ == "__main__":
    render(
        axs[0, 0],
        "(A) Advanced tactic",
        None,
        (1, 2),
        False,
        fences,
        pawn1_pos,
        pawn2_pos,
        dotted_fences,
    )

    render_hist(
        axs[0, 2],
        "Figure_Data/Adv_tactic/e_forest_move_distr.csv",
       "(D) Forest move evaluations",
        False,
        winning_moves,
    )
    render_hist(
        axs[0, 1],
        "Figure_Data/Adv_tactic/b_naive_move_distr.csv",
         "(B) Naive move evaluations",
        
        False,
        winning_moves,
    )

    render_hist(
        axs[0, 3],
        "Figure_Data/Adv_tactic/cnn_histogram.csv",
         "(C) CNN move evaluations",
        
        False,
        winning_moves,
    )

    graph_eval_data(
        axs[1, 0],
        "Figure_Data/Adv_tactic/c_naive_eval_over_time.csv",
        "(E) Naive position evaluation",
        True,
    )

    graph_eval_data(
        axs[1, 2],
        "Figure_Data/Adv_tactic/cnn_line.csv",
        "(G) CNN position evaluation",
        True,
    )

    graph_eval_data(
        axs[1, 1],
        "Figure_Data/Adv_tactic/d_basic_eval_over_time.csv",
        "(F) Basic position evaluation",
    )
    graph_eval_data(
        axs[1, 3],
        "Figure_Data/Adv_tactic/f_forest_eval_over_time.csv",
        "(H) Forest position evaluation",
        True,
    )

    plt.tight_layout()

    # plt.show()
    plt.savefig("../report/Adv_tactic.png", dpi=100)
