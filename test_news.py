from SPN.SPN import SPN
from data.DataHandler import DataHandler
import numpy as np
from news_vendor import PROBLEMS, solve_model, eval_solution, get_training_samples
import logging
from scipy import stats

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(stream='sys.stdout', encoding='utf-8', level=logging.INFO)

seed = 0


for prob in PROBLEMS:
    # for n_samples, p in [(10, 0.95), (100, 0.95)]:
    for n_samples, p in [(100, 0.95)]:
        print(n_samples, p)
        for chance_variant in ["sample_robust", "sample_average", "SPN"]:
        # for chance_variant in ["SPN"]:
            if chance_variant == "SPN":
                samples, p_x = get_training_samples(**prob["kwargs"])
                p *= p_x
                X = np.concatenate(samples, axis=1)
                # X = np.concatenate(samples[1:], axis=1)
                dhandler = DataHandler(X, feature_names=["D", "x", "sat"], categ_map={"sat":[0,1]})
                # dhandler = DataHandler(X, feature_names=["x", "sat"], categ_map={"sat":[0,1]})
                n_clusters = 10
                # n_clusters = 2
                spn = SPN(
                    dhandler.encode(X, normalize=False, one_hot=False),
                    dhandler,
                    normalize_data=False,
                    # learn_mspn_kwargs={"min_instances_slice":200, "n_clusters":n_clusters}
                    learn_mspn_kwargs={"min_instances_slice":20000, "n_clusters":n_clusters}
                )
                spn.marginalize([1,2])

                x = np.linspace(40, 160, 1000)
                lls = spn.compute_ll(np.concatenate([np.zeros((1000,1)), x.reshape(-1, 1), np.ones((1000,1))], axis=1))
                # lls = spn.compute_ll(np.concatenate([x.reshape(-1, 1), np.ones((1000,1))], axis=1))
                maxout = np.array([spn.compute_max_approx(np.array([0, x_i, 1])) for x_i in x])
                # maxout = np.array([spn.compute_max_approx(np.array([x_i, 1])) for x_i in x])
                cdf = stats.norm.cdf(x, loc=prob["kwargs"]["demand"]["mean"], scale=prob["kwargs"]["demand"]["std"])

                # maxpw = np.array([spn.compute_maxpw_approx(np.array([0, x_i, 1]), n_clusters) for x_i in x])
                maxpw = np.array([spn.compute_maxpw_approx(np.array([0, x_i, 1]), 10) for x_i in x])

                plt.plot(x, cdf,
                        label='True ℙ$(x \\geq D)$',
                        color='green', linewidth=3, zorder=10)
                plt.plot(x, np.exp(lls) / p_x,
                        label='Trained SPN Approximation divided by $p(x)$',
                        color='blue', zorder=5)
                plt.plot(x, np.exp(maxout) / p_x,
                        label='Max-approximation of SPN (within MIO) divided by $p(x)$',
                        color='red', zorder=6)
                plt.plot(x, np.exp(maxpw) / p_x,
                        label='Max + Piecewise-approximation of SPN (within MIO) divided by $p(x)$',
                        color='magenta', zorder=7)

                plt.title('SPN approximation of ℙ(x ≥ D) for D ~ N(100, 20)', fontsize=14)
                plt.xlabel('x', fontsize=12)
                plt.ylabel('True or apporixmated ℙ(x ≥ D)', fontsize=12)

                # Add grid
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                        fancybox=True, shadow=True, ncol=1)
                plt.tight_layout()
                plt.savefig('SPN_approximation_news.png', dpi=300, bbox_inches='tight')
                plt.show()

                # p *= 0.9 # to account for the max relaxations.
            else:
                spn = None
            print(prob["name"], chance_variant)
            x = solve_model(prob["name"], chance_variant=chance_variant, n_samples=n_samples, p=p, seed=seed, spn=spn, **prob["kwargs"])
            print(x)
            if spn is not None:
                print(np.exp(spn.compute_ll(np.array([[0, x, 1]]))) / p_x)
                # print(np.exp(spn.compute_ll(np.array([[x, 1]]))) / p_x)
            print(eval_solution(x, **prob["kwargs"]))
            print()
        break