import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


def cluster_draw(title, losses, means, convs):
    plt.figure()
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 5)

    higher_distro, lower_distro = means.argmax(), means.argmin()
    mu1, mu2 = means[higher_distro][0], means[lower_distro][0]
    sigma1, sigma2 = math.sqrt(convs[lower_distro][0][0]), math.sqrt(convs[higher_distro][0][0])

    plt.hist(losses, bins=50, density=True)

    x = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 100)
    plt.plot(x, stats.norm.pdf(x, mu1, sigma1), "r")

    x = np.linspace(mu2 - 3 * sigma2, mu2 + 3 * sigma2, 100)
    plt.plot(x, stats.norm.pdf(x, mu2, sigma2), "g")

    x = np.linspace(
        min(mu1 - 3 * sigma1, mu2 - 3 * sigma2),
        max(mu1 + 3 * sigma1, mu2 + 3 * sigma2),
        100,
    )
    plt.plot(x, stats.norm.pdf(x, mu1, sigma1) + stats.norm.pdf(x, mu2, sigma2), "y")

    plt.show()

    os.makedirs("./plt_figures", exist_ok=True)
    filename = title.replace("\n", "").replace(" ", "").replace("=", "")
    plt.savefig(f"./plt_figures/{filename}.png")
