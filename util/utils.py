import numpy as np
import json
import time
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm


def kl(u0, sigma0, u1, sigma1):
    """return KL distance of two multivariate Gaussian distribution f,g: D_KL(f||g)"""

    inv_sigma0 = np.linalg.inv(sigma0)

    # general case
    res = 1 / 2 * (np.trace(np.matmul(inv_sigma0, sigma1)) + np.matmul(np.matmul((u0 - u1).T, inv_sigma0), (u0 - u1))
                   - u0.shape[0] + np.log(np.linalg.det(sigma0) / np.linalg.det(sigma1)))

    return res[0, 0]


def g(data, u, sigma):
    """The log of multivariate Gaussian distribution log pdf [log g(x..., u, sigma)]"""
    M, N = data.shape
    dfrac = np.log((np.sqrt(2 * np.pi)) ** M * np.sqrt(np.linalg.det(sigma)))
    inv_sigma = np.linalg.inv(sigma)

    log_pdf_set = []
    for i in range(N):
        x = data[:, i] - u
        y = np.dot(np.dot(x.T, inv_sigma), x)
        log_pdf = -1 / 2 * y[0, 0] - dfrac
        log_pdf_set.append(log_pdf)

    return log_pdf_set


def geo(k, rho, N):
    """The geometric distribution probability where P(lambda=N+1) represents P(k>N)"""
    if k <= N:
        return (1 - rho) ** (k - 1) * rho
    elif k == N + 1:
        return (1 - rho) ** N


def data_generator(normal, change, rho, datasrc):
    """concatenate pre-change data and post-change data"""
    NN, M = normal.shape
    while True:
        # random Lambda value: change point
        Lambda = int(np.random.geometric(rho, 1))
        N = Lambda + 50

        # pick a random start
        start = np.random.randint(0, NN - N)

        # exclude some extreme cases
        if Lambda >= 0 and Lambda <= 50:
            break

    data = np.concatenate((normal[start:start + Lambda - 1, :].T, change[start + Lambda - 1:start + N, :].T), axis=1)

    return Lambda, N, data


def obtain_posteriors(data, u0, u1, sigma0, sigma1, thres_max, rho, method):
    """compute the sequence of posterior probability ratios"""
    M, N = data.shape

    # initialize parameter guess in case we don't known them
    u1_hat = u0
    sigma1_hat = sigma0

    #
    posteriors = [0] * 1
    for n in range(1, N + 1):

        if method == 'benchmark':
            u1_hat, sigma1_hat = u1, sigma1

        elif method == 'PGD':
            u1_hat, sigma1_hat = PGD(data[:, :n], u0, sigma0, u1_hat, sigma1_hat, rho)

        # compute posterior probability ratio
        posterior_n = posterior(data[:, :n], u0, sigma0, u1_hat, sigma1_hat, rho)
        posteriors.append(posterior_n)

        # save some time
        if posterior_n > thres_max:
            return posteriors

    return posteriors


def posterior(x, u0, sigma0, u1, sigma1, rho):
    """compute posterior probability ratio"""

    n = x.shape[1]

    pre = [0] + g(x, u0, sigma0)
    post = [0] + g(x, u1, sigma1)

    res = 0
    s1 = np.sum(pre)
    s2 = np.sum(post)

    s = rho
    for k in range(1, n + 1):
        s1 -= pre[k - 1]
        s2 -= post[k - 1]

        if s2 - s1 > 100:
            res += s * np.exp(100)
        else:
            res += s * np.exp(s2 - s1)

        s *= (1 - rho)

    return res / geo(n + 1, rho, n)


def objective_fun(prob_set, N, rho):
    """The objective function L"""
    rhos = [geo(k, rho, N) for k in range(1, N + 1)]
    obj = 0
    for k in range(1, N + 1):
        obj += prob_set[k - 1] * rhos[k - 1]
    return obj


def gradients(data0, data1, inv_sigma0, inv_sigma1, phi, prob_set, rho):
    """gradients of L(p1)"""
    N = data1.shape[1]
    D = data1.shape[0]

    #
    delta_u1 = np.zeros((D, 1))
    delta_sigma1 = np.zeros((D, D))

    #
    rhos = [geo(k, rho, N) for k in range(1, N + 1)]

    for k in range(1, N + 1):
        prob = prob_set[k - 1]

        # data-u1 k-N
        data1k = data1[:, k - 1:]

        # u1
        post = np.sum(data1k, axis=1)
        delta_u1 += prob * np.dot(inv_sigma1, post) * rhos[k - 1]

        # sigma1
        temp1 = np.dot(data1k, data1k.T)
        temp = (k - N - 1) * inv_sigma1 + np.dot(np.dot(inv_sigma1, temp1), inv_sigma1)
        delta_sigma1 += prob / 2 * temp * rhos[k - 1]

    return delta_u1 / phi, delta_sigma1 / phi


def PGD(data, u0, sigma0, u1, sigma1, rho):
    """the Projected Gradient Descent learning framework"""
    D, N = data.shape
    if N < 5:
        learning_rate = 0.003
    elif N < 10:
        learning_rate = 0.001
    elif N < 15:
        learning_rate = 0.001
    elif N < 20:
        learning_rate = 0.0008
    else:
        learning_rate = 0.0005
    momentum = 0
    pre_phi = 1
    max_epoch = 30

    delta_u1_pre, delta_sigma1_pre = 0, 0

    u1_best = np.zeros_like(u1)
    sigma1_best = np.zeros_like(sigma1)
    phi_best = -1

    for e in range(max_epoch):

        # prepare
        inv_sigma0 = np.linalg.inv(sigma0)
        inv_sigma1 = np.linalg.inv(sigma1)

        ones = np.ones((1, u0.shape[1]))
        u0_expand = np.dot(u0, ones)
        u1_expand = np.dot(u1, ones)

        data0 = data - u0_expand
        data1 = data - u1_expand

        prob_set = prob_given_paras(data, u0, u1, sigma0, sigma1)

        # compute gradient
        phi = objective_fun(prob_set, data.shape[1], rho)

        if phi == 0:
            break

        # gradients
        delta_u1, delta_sigma1 = gradients(data0, data1, inv_sigma0, inv_sigma1, phi, prob_set, rho)

        # update
        u1 += (delta_u1 + momentum * delta_u1_pre) * learning_rate

        sigma1_log = logm(sigma1)
        sigma1_log += (delta_sigma1 + momentum * delta_sigma1_pre) * learning_rate / 5
        sigma1 = expm(sigma1_log)

        # store pre-gradients to perform momentum
        delta_u1_pre, delta_sigma1_pre = delta_u1, delta_sigma1

        # stop iteration
        if abs(pre_phi - np.log(phi)) < 0:
            print("stop")
            break
        else:
            pre_phi = np.log(phi)

        # compute gradient
        prob_set = prob_given_paras(data, u0, u1, sigma0, sigma1)
        phi = objective_fun(prob_set, data.shape[1], rho)

        if phi > phi_best:
            phi_best = phi
            u1_best = u1
            sigma1_best = sigma1

    u1 = u1_best
    sigma1 = sigma1_best
    return u1, sigma1


def prob_given_paras(x, u0, u1, sigma0, sigma1):
    """P(data|theta, lambda)"""
    N = x.shape[1]

    pre = g(x, u0, sigma0)
    post = g(x, u1, sigma1)

    res = []
    for k in range(1, N + 1):
        temp = np.sum(pre[:k - 1]) + np.sum(post[k - 1:])
        res.append(temp)

    m = np.max(res)

    # approach1
    res2 = np.exp(res - m)

    return res2


def run_all_alphas(normal, change, u0, sigma0, u1, sigma1, times, alpha_set, rho, method, datasrc):
    """run one experiment for all alpha values"""

    # display frequency
    disp_step = 1 if times // 10 == 0 else times // 10

    # save results to
    delay = {}
    curve = {}
    hit = {}
    ticks = {}
    for scenario in method:
        ticks[scenario] = 0
        curve[scenario] = []
        hit[scenario] = []
        delay[scenario] = {}
    for scenario in method:
        for alpha in alpha_set:
            delay[scenario][alpha] = []

    # MonteCarol many times
    for i in range(times):
        if i % disp_step == 0:
            print("iteration:%d / %d" % (i+1, times))

        # Create data(including pre-change and post-change)
        Lambda, N, data = data_generator(normal, change, rho, datasrc)

        # the largest threshold
        thres_max = (1 - alpha_set[-1]) / rho / alpha_set[-1]

        # different scenarios
        for scenario in method:
            start = time.time()
            posteriors = obtain_posteriors(data, u0, u1, sigma0, sigma1, thres_max, rho, scenario)

            # plt.figure()
            # plt.plot(posteriors)
            # plt.show()

            # detected change point tau
            for alpha in alpha_set:
                threshold = (1 - alpha) / rho / alpha
                flag = 0
                for n in range(len(posteriors)):
                    if posteriors[n] > threshold:
                        # print(n, Lambda, alpha)
                        flag = 1
                        if n >= Lambda:
                            delay[scenario][alpha].append(n - Lambda)
                        break
                if not flag:
                    delay[scenario][alpha].append(len(posteriors) - Lambda)

            # tick
            ticks[scenario] += time.time() - start

    # saving results
    for scenario in method:
        print('\n\n------  ' + scenario + '  -------')
        print(ticks[scenario], end='s\n')
        for alpha in alpha_set:
            actual_delay = np.mean(delay[scenario][alpha]) if delay[scenario][alpha] != [] else 0
            hit[scenario].append(1 - len(delay[scenario][alpha]) / times)
            curve[scenario].append(actual_delay / abs(np.log(alpha)))

    return curve, hit


def changePointDetection(normal, change, u0, sigma0, u1, sigma1, rho, mtimes, alpha_set, method, datasrc):
    # --- pre_computation
    kl_distance = kl(u0, sigma0, u1, sigma1)
    geo_distance = -np.log(1 - rho)
    bound = 1 / (geo_distance + kl_distance)

    # print
    print("kl:", kl_distance)
    print("delay lowerbound:", bound)

    # --- detecting
    curve, hit = run_all_alphas(normal, change, u0, sigma0, u1, sigma1, mtimes, alpha_set, rho, method, datasrc)

    # save result to a json file
    save_results(alpha_set, bound, curve, hit, method, datasrc)


def save_results(alpha_range, bound, curve, hit, method, datasrc):
    """save the results to json"""
    with open('json/'+datasrc+'.json', 'w') as fw:
        data = {'alpha_range': alpha_range, 'bound': bound}
        for scenario in method:
            data[scenario] = {'curve': curve[scenario], 'hit': hit[scenario]}
        json.dump(data, fw)
    fw.close()

def draw_bound(method, datasrc):
    """draw the averaged detection delay and empirical false alarm rate"""

    # config matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    # makers and colors
    markers = {
        'benchmark': 's-',
        'PGD': 'o-',
    }

    colors = {
        'benchmark': 'blue',
        'PGD': 'purple'
    }

    # loading data
    data = json.load(open('json/'+datasrc+'.json', 'r'))
    alpha_range = data['alpha_range']
    bound = data['bound']
    x_range = []
    for alpha in alpha_range:
        x_range.append(abs(np.log(alpha)))

    # draw - averaged detection delay
    plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(x_range, [bound] * len(x_range), color='orange', linewidth=3, label='theoretical bound')  # theoretical bound
    for scenario in method:
        curve = data[scenario]['curve']
        plt.plot(x_range, curve, markers[scenario], color=colors[scenario], markersize=7, label=scenario, markerfacecolor='none')

    # axis and fontsize
    ax1.set_title('(a). The averaged detection delay', fontsize=14)
    # plt.yticks([0, 10])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$\\boldsymbol{|\log\\alpha|}$', fontsize=14)
    plt.ylabel('$\\boldsymbol{\\frac{E(\\tau-\\lambda|\\tau\ge\\lambda)}{|\\log \\alpha|}}$', fontsize=14)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95),
              fancybox=True, shadow=True, ncol=1, fontsize=12)

    # draw - empirical false alarm rate
    ax2 = plt.subplot(212)
    ax2.plot(x_range, alpha_range, '-', linewidth=3, color='orange', label='$\\alpha$')
    for scenario in method:
        hit = data[scenario]['hit']
        plt.plot(x_range, hit, markers[scenario], color=colors[scenario], markersize=7, label=scenario, markerfacecolor='none')

    # axis and fontsize
    ax2.set_title('(b). The emprical false alarm rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('$\\boldsymbol{|\log\\alpha|}$', fontsize=14)
    plt.ylabel('$\\bf{empirical}$ $\\boldsymbol{\\alpha}$', fontsize=14)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95),
              fancybox=True, shadow=True, ncol=1, fontsize=12)

    # save figure
    plt.subplots_adjust(hspace=0.6, left=0.15)
    plt.savefig('figures/'+datasrc+'.eps', format='eps')
    plt.show()

