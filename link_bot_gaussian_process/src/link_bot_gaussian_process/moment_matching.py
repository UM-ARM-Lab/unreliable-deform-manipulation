import matplotlib.pyplot as plt
import numpy as np


def predict(models, X, Y, prior):
    kernels = [m.kern for m in models]
    D = Y.shape[1]
    N, M = X.shape
    mu_tilde_t, sigma_tilde_t = prior

    # moment matching distribution
    mu_tilde_t_plus_1 = np.ndarray((D, 1))

    sigma_tilde_t_plus_1 = np.ndarray((D, D))
    mu_tilde_t_vec = np.ones((N, 1)) * mu_tilde_t

    for a in range(D):
        m_a = models[a]
        k_a = kernels[a]
        K_a = k_a.compute_K_symm(X)
        K_a_x_mu_tilde = k_a.compute_K(X, mu_tilde_t_vec)
        sigma_w_squared_a = m_a.likelihood.variance.value
        sigma_f_squared_a = m_a.kern.variance.value
        # the times np.eye is from the PILCO code, I don't see it in the math...
        # GPFlow uses Cholesky but this is simpler for testing purposes
        # this replaces doing the actual inverse
        squared_length_scales_a = np.square(np.atleast_1d(m_a.kern.lengthscales.value))
        lambda_a = np.diag(squared_length_scales_a)
        inv_lambda_a = np.diag(1 / squared_length_scales_a)
        beta_a = np.linalg.lstsq(K_a + sigma_w_squared_a * np.eye(N), Y[:, a], rcond=1)[0]
        iK = np.linalg.lstsq(K_a + sigma_w_squared_a * np.eye(N), np.eye(N), rcond=1)[0]

        # Compute the mean
        T_a = sigma_tilde_t + lambda_a
        inv_T_a = np.linalg.lstsq(T_a, np.eye(M), rcond=1)[0]
        q_a = np.ndarray(N)
        for i in range(N):
            nu_i = X[i] - mu_tilde_t
            # this inverse is fine because T is a non-zero real scalar
            e_i = np.exp(-0.5 * nu_i.T @ inv_T_a @ nu_i)
            # in the PILCO code they have another inv_lambda_a in this term?
            q_a_i = sigma_f_squared_a / np.sqrt(np.linalg.det(sigma_tilde_t @ inv_lambda_a + np.eye(D))) * e_i
            q_a[i] = q_a_i
        mu_tilde_t_plus_1[a] = beta_a.T @ q_a

        # Compute covariance
        for b in range(D):
            m_b = models[b]
            k_b = kernels[b]
            K_b = k_b.compute_K_symm(X)
            K_b_x_mu_tilde = k_b.compute_K(X, mu_tilde_t_vec)
            sigma_w_squared_b = m_a.likelihood.variance.value
            squared_length_scales_b = np.square(np.atleast_1d(m_b.kern.lengthscales.value))
            inv_lambda_b = np.diag(1 / squared_length_scales_b)
            beta_b = np.linalg.lstsq(K_b + sigma_w_squared_b * np.eye(N), Y[:, b], rcond=1)[0]

            # missing negative sign?
            R = sigma_tilde_t @ (inv_lambda_a + inv_lambda_b) + np.eye(D)
            Q = np.ndarray((N, N))
            L = np.ndarray((N, N))
            for i in range(N):
                nu_i = X[i] - mu_tilde_t
                for j in range(N):
                    nu_j = X[j] - mu_tilde_t
                    z_i_j = inv_lambda_a @ nu_i + inv_lambda_b @ nu_j
                    e_ab_ij = np.exp(0.5 * z_i_j.T @ np.linalg.inv(R) @ sigma_tilde_t @ z_i_j)
                    L[i, j] = e_ab_ij
                    Q[i, j] = K_a_x_mu_tilde[i, 0] * K_b_x_mu_tilde[j, 0] / np.sqrt(np.linalg.det(R)) * e_ab_ij

            sigma_tilde_t_plus_1[a, b] = beta_a.T @ Q @ beta_b - mu_tilde_t_plus_1[a] * mu_tilde_t_plus_1[b]
            if a == b:
                sigma_tilde_t_plus_1[a, b] += sigma_f_squared_a - np.trace(iK @ Q)
    return mu_tilde_t_plus_1, sigma_tilde_t_plus_1


def pdf(sample_x_points, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (sample_x_points - mean) ** 2 * 1 / (var))


def plot_approximation(model, X, Y, prior, posterior):
    # sampled output
    mu_tilde_t, sigma_tilde_t = prior
    mu_tilde_t_plus_1, sigma_tilde_t_plus_1 = posterior
    x_tilde_t_samples = np.random.normal(mu_tilde_t[0], sigma_tilde_t[0, 0] ** 0.5, 5000).reshape(-1, 1)
    f_x_tilde_t_samples_mean, f_x_tilde_t_samples_var = model.predict_y(x_tilde_t_samples)
    y_samples = []
    for mean_i, sigma_i in zip(f_x_tilde_t_samples_mean, f_x_tilde_t_samples_var):
        y_samples.extend(np.random.normal(mean_i, sigma_i ** 0.5, 5000))

    # empirical (sample) mean + var
    sample_f_mean = np.mean(y_samples)
    sample_f_var = np.var(y_samples)
    print(sample_f_mean, sample_f_var)

    # generate test points for prediction
    n_test_points = 500
    xx = np.linspace(-1, 1, n_test_points).reshape(n_test_points, 1)  # test points must be of shape (N, D)
    p_x_tilde_t = pdf(xx, mu_tilde_t, sigma_tilde_t)

    # predict mean and variance of latent GP at test points
    mean, var = model.predict_f(xx)

    p_x_tilde_t_plus_1_empirical = pdf(xx, sample_f_mean, sample_f_var)
    p_x_tilde_t_plus_1 = pdf(xx, mu_tilde_t_plus_1, sigma_tilde_t_plus_1)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex='col', sharey='row', gridspec_kw={'height_ratios': [5, 2]})

    axes[0, 0].invert_xaxis()
    axes[0, 0].plot(p_x_tilde_t_plus_1_empirical, xx, color='#376C17', linewidth=8, alpha=0.4,
                    label='empirical posterior pdf')
    axes[0, 0].plot(p_x_tilde_t_plus_1, xx, color='b')
    hist_ax_00 = axes[0, 0].twiny()
    hist_ax_00.hist(y_samples, bins=300, orientation='horizontal', label='sampled posterior', color='green', alpha=0.4)
    hist_ax_00.invert_xaxis()

    axes[0, 1].plot(X, Y, 'kx', mew=2, label='training data')
    axes[0, 1].plot(xx, mean, 'k', lw=2, label='GP mean function')
    z_value = 1.96
    axes[0, 1].fill_between(xx[:, 0],
                            mean[:, 0] - z_value * np.sqrt(var[:, 0]),
                            mean[:, 0] + z_value * np.sqrt(var[:, 0]),
                            color='grey', alpha=0.2)
    axes[1, 1].plot(xx, p_x_tilde_t, color='b', label='prior pdf')
    hist_ax_11 = axes[1, 1].twinx()
    hist_ax_11.hist(x_tilde_t_samples, bins=150, label='sampled prior', alpha=0.4)

    axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].set_ylim(-1, 1)
    axes[0, 0].set_ylim(-1, 1)
    hist_ax_00.set_ylim(-1, 1)
    axes[1, 1].set_xlabel(r"$\tilde{x}_t$")
    axes[1, 1].set_ylabel(r"$p(\tilde{x}_{t+1})$")
    axes[1, 1].set_ylim(0, 2)
    hist_ax_11.set_ylim(0, 150)
    axes[0, 0].set_ylabel(r"$\tilde{x}_{t+1}$")

    axes[1, 0].axis('off')

    axes[0, 0].grid()
    axes[0, 1].grid()
    axes[1, 1].grid()

    fig.tight_layout()

    handles_00, labels_00 = axes[0, 0].get_legend_handles_labels()
    handles_01, labels_01 = axes[0, 1].get_legend_handles_labels()
    handles_11, labels_11 = axes[1, 1].get_legend_handles_labels()
    handles = handles_00 + handles_11 + handles_01
    labels = labels_00 + labels_11 + labels_01
    fig.legend(handles, labels, loc='lower left')
    plt.show()
