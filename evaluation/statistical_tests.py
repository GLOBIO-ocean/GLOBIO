import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def wilcoxon_best_vs_all(data: dict[str, np.ndarray], alpha: float = 0.05):
    """
    Returns:
    - best_res: dict {model: True if significantly better than all others}
    - p_matrix: dict of raw p-values (directional)
    """

    models = list(data.keys())

    best_res = {m: False for m in models}
    p_matrix = {m: {} for m in models}

    for m1 in models:

        pvals = []
        others = []

        for m2 in models:
            if m1 == m2:
                continue

            a = np.asarray(data[m1])
            b = np.asarray(data[m2])

            if np.array_equal(a, b):
                p = 1.0
            else:
                _, p = wilcoxon(a, b, alternative="less")  # m1 < m2 (better)

            pvals.append(p)
            others.append(m2)

            p_matrix[m1][m2] = p

        # Holm correction across all comparisons of m1
        reject, _, _, _ = multipletests(pvals, alpha=alpha, method="holm")

        # BEST if better than ALL others (after correction)
        best_res[m1] = bool(np.all(reject))

    return best_res, p_matrix