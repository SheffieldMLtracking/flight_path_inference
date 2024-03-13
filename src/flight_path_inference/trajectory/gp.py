import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def K(X1, X2):
    """
    Our kernel. An EQ+Bias kernel. The prior is UNCORRELATED between dimensions.
    """
    cov = (50 ** 2) * np.exp(-np.sum(np.subtract(X1[:, None], X2[None, :]) ** 2 / (2 * 1.2 ** 2), 2))  # +(100**2)
    # cov = (50**2)*np.exp(-np.sum(np.subtract(X1[:,None],X2[None,:])**2/(2*0.7**2),2))#+(100**2)
    axsel = tf.cast((X1[:, 1][:, None] == X2[:, 1][None, :]), dtype=tf.float32)
    cov = cov * axsel
    return cov


def buildinputmatrix(min_time, max_time, size):
    """
    Constructs a matrix of [3*size,2], where the first column is time and second is axis (x,y,z).
    """
    A = []
    for ax in range(3):
        Aax = np.c_[np.linspace(min_time, max_time, size), np.full(size, ax)]
        A.extend(Aax)
    A = np.array(A)
    A = tf.Variable(A, dtype=tf.float32)
    return A


def cross(a, b):
    """
    Compute cross product with batching. Currently only allows particular number of dimensions in the input...
    a - [!,3] tensor
    b - [*,!,3] tensor
    e.g.
    a.shape = [20,3]
    b.shape = [15,20,3]
    result.shape = [15,20,3]
    TODO: Generalise for any compatible input batch shapes
    """
    size = a.shape[0]
    A = tf.Variable(
        [[tf.zeros(size), -a[:, 2], a[:, 1]], [a[:, 2], tf.zeros(size), -a[:, 0]], [-a[:, 1], a[:, 0], tf.zeros(size)]])
    A = tf.transpose(A, [2, 0, 1])
    A = A[None, :, :, :]
    b = b[:, :, :, None]
    return (A @ b)[:, :, :, 0]


def compute_matrices(X, Z, jitter):
    Kzz = K(Z, Z) + np.eye(Z.shape[0], dtype=np.float32) * jitter
    Kxx = K(X, X) + np.eye(X.shape[0], dtype=np.float32) * jitter
    Kxz = K(X, Z)
    Kzx = tf.transpose(Kxz)
    KzzinvKzx = tf.linalg.solve(Kzz, Kzx)
    KxzKzzinv = tf.transpose(KzzinvKzx)
    KxzKzzinvKzx = Kxz @ KzzinvKzx
    return Kzz, Kxx, Kxz, Kzx, KzzinvKzx, KxzKzzinv, KxzKzzinvKzx


def getcov(scale):
    return tf.linalg.band_part(scale, -1, 0) @ tf.transpose(tf.linalg.band_part(scale, -1, 0))


def run(obstimes, observations, iterations=500, learning_rate=0.15, likenoisescale=0.05, Nind=None, Nsamps=1000):
    """
    Build and optimise a Gaussian process model for the trajectory.
    obstimes = An (N) array of time of the observations, ideally starts about zero.
    observations = An (Nx7) array of observations [originX, originY, originZ, vectorX, vectorY, vectorZ, confidence]
    learning_rate = optimiser's learning rate
    likenoisescale = 0.05m (default) std of likelihood noise.
    Nind = number of inducing points (default is 1+int(3*np.max(obstimes))).
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Build the inducing point locations:
    if Nind is None:
        Nind = 1 + int(3 * np.max(obstimes))
    Z = buildinputmatrix(0, np.max(obstimes) + 1, Nind)
    X = tf.Variable(np.c_[np.tile(obstimes, 3)[:, None], np.repeat(np.arange(3), len(obstimes), axis=0)],
                    dtype=tf.float32)
    y = tf.Variable(observations[:, :6], dtype=tf.float32)

    # number of inducing points.
    m = Z.shape[0]
    # create variables that describe q(u), the variational distribution.
    mu = tf.Variable(tf.random.normal([m, 1]))
    # scale = tf.Variable(tf.eye(m))#0.001*tf.random.normal([m, m])+0.1*tf.eye(m))
    scale = tf.Variable(np.tril(0.1 * np.random.randn(m, m) + 1.0 * np.eye(m)), dtype=tf.float32)

    # parameters for p(u), the prior.
    mu_u = tf.zeros([1, m])
    cov_u = tf.Variable(K(Z, Z))
    jitter = 1e-6
    noisescale = 1e-3
    pu = tfd.MultivariateNormalFullCovariance(mu_u, cov_u + np.eye(cov_u.shape[0]) * noisescale)

    # We don't optimise the hyperparameters, so precompute.
    Kzz, Kxx, Kxz, Kzx, KzzinvKzx, KxzKzzinv, KxzKzzinvKzx = compute_matrices(X, Z, jitter)

    size = int(X.shape[0] / 3)  # number of input points

    for it in range(iterations):
        with tf.GradientTape() as tape:

            # the variational approximating distribution.
            qu = tfd.MultivariateNormalTriL(mu[:, 0], scale)

            # compute the approximation over our training point locations
            # TODO only need some diagonals and off diagonal parts of qf_cov, so prob could be quicker!
            qf_mu = (KxzKzzinv @ mu)[:, 0]
            qf_cov = Kxx - KxzKzzinvKzx + KxzKzzinv @ getcov(scale) @ KzzinvKzx

            # this gets us the covariance and mean for the relevant parts of the predictions. Specifically
            # a 3x3 covariance and a 3-element mean.
            C = tf.transpose(tf.concat([qf_cov[i::size, i::size][:, :, None] for i in range(size)], axis=2), [2, 0, 1])
            M = tf.transpose(tf.reshape(qf_mu, [3, size]), [1, 0])

            jitter = 1e-6
            for inversionerrorloop in range(5):
                try:
                    samps = tfd.MultivariateNormalTriL(M, tf.linalg.cholesky(C + tf.eye(3) * jitter)).sample(Nsamps)
                    break
                except tf.errors.InvalidArgumentError:
                    jitter *= 10

            # we compute the distance from each of the observed vectors to the samples and compute their
            # log likelihoods assuming a normal distributed likelihood model over the distance from the
            # vector.
            d = tf.norm(cross(y[:, 3:], samps - y[:, :3]), axis=2) / tf.norm(y[:, 3:], axis=1)
            logprobs = tfd.Normal(0, likenoisescale).log_prob(d)
            ell = tf.reduce_mean(tf.reduce_sum(logprobs, 1))

            # we compute the ELBO = - (expected log likelihood of the data - KL[prior, variational_distribution]).
            elbo_loss = -(ell - tfd.kl_divergence(qu, pu))
        # compute gradients and optimise...
        gradients = tape.gradient(elbo_loss, [mu, scale])
        optimizer.apply_gradients(zip(gradients, [mu, scale]))
        if it % 20 == 0: print(it, elbo_loss.numpy())
    return Z, mu, scale
