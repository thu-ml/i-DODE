# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Iterable, Any, Tuple, Type
import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from scipy import integrate
from utils import batch_mul
import functools
import ml_collections
import string
from typing import Any, Tuple


@flax.struct.dataclass
class VDMConfig:
    """VDM configurations."""

    vocab_size: int
    sample_softmax: bool
    antithetic_time_sampling: bool
    with_fourier_features: bool
    with_attention: bool
    second_order: bool
    velocity: bool
    importance: bool
    dequantization: str
    num_importance: int

    # configurations of the noise schedule
    gamma_type: str
    gamma_min: float
    gamma_max: float
    schedule: str

    # configurations of the score model
    sm_n_timesteps: int
    sm_n_embd: int
    sm_n_layer: int
    sm_pdrop: float
    sm_kernel_init: Callable = jax.nn.initializers.normal(0.02)


######### Latent VDM model #########


@flax.struct.dataclass
class VDMOutput:
    loss_recon: chex.Array  # [B]
    loss_klz: chex.Array  # [B]
    loss_diff: chex.Array  # [B]
    loss_diff2: chex.Array
    var_0: float
    var_1: float


class VDM(nn.Module):
    config: VDMConfig

    def setup(self):
        self.encdec = EncDec(self.config)
        self.score_model = ScoreUNet(self.config)

        self.velocity = True
        self.importance = True

        self.schedule = self.config.schedule
        self.second_order = self.config.second_order
        self.velocity = self.config.velocity
        self.importance = self.config.importance
        self.num_importance = self.config.num_importance
        assert self.schedule == "VP" or self.schedule == "SP", "Only VP and SP are supported noise schedules."
        assert (
            not self.second_order or self.velocity
        ), "Second-order flow matching objective must be used together with velocity parameterization."
        assert (
            not self.importance or self.velocity
        ), "Importance sampling must be used together with velocity parameterization."
        self.dequantization = self.config.dequantization
        assert self.config.gamma_type == "fixed", "We only support fixed noise schedule."
        self.gamma = NoiseSchedule_FixedLinear(self.config)

    def likelihood_importance_cum_weight(self, g_0, g_t):
        def cum_weight(g):
            exponent = jnp.exp(-g / 2)
            term1 = jnp.where(exponent <= 1e-3, exponent, jnp.log(1 + exponent))
            term2 = jnp.where(exponent <= 1e-3, 1 - exponent, 1 / (1 + exponent))
            return term1 + term2

        return -2 * (cum_weight(g_t) - cum_weight(g_0))

    def sample_importance_weighted_time_for_likelihood(self, g_0, g_1, quantile, steps=100):
        lb = jnp.ones_like(quantile) * g_0
        ub = jnp.ones_like(quantile) * g_1

        def bisection_func(carry, idx):
            lb, ub = carry
            mid = (lb + ub) / 2.0
            value = self.likelihood_importance_cum_weight(g_0, mid)
            lb = jnp.where(value <= quantile, mid, lb)
            ub = jnp.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        (lb, ub), _ = jax.lax.scan(bisection_func, (lb, ub), jnp.arange(0, steps))
        return (lb + ub) / 2.0

    def noise_schedule(self, g):
        if self.schedule == "VP":
            var_t = nn.sigmoid(g)
            alpha_t = jnp.sqrt(1 - var_t)
            sigma_t = jnp.sqrt(var_t)
        elif self.schedule == "SP":
            sigma_t = nn.sigmoid(g / 2)
            alpha_t = 1 - sigma_t

        return alpha_t, sigma_t

    def log_alpha_hat(self, g):
        if self.schedule == "VP":
            log_alpha_t_hat = -0.5 * nn.sigmoid(g)
        elif self.schedule == "SP":
            log_alpha_t_hat = -0.5 * nn.sigmoid(g / 2)
        return log_alpha_t_hat

    def __call__(
        self, images, conditioning, deterministic: bool = True, hutchinson_type="Rademacher", second_order_weight=0.1
    ):
        g_0, g_1 = self.gamma(0.0), self.gamma(1.0)
        alpha_0, sigma_0 = self.noise_schedule(g_0)
        alpha_1, sigma_1 = self.noise_schedule(g_1)
        x = images
        n_batch = images.shape[0]

        # encode
        f = self.encdec.encode(x)

        # 1. RECONSTRUCTION LOSS
        # add noise and reconstruct
        eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = alpha_0 * f + sigma_0 * eps_0
        z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0  # = z_0/sqrt(1-var)
        loss_recon = -self.encdec.logprob(x, z_0_rescaled, g_0)

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        mean1_sqr = (alpha_1**2) * jnp.square(f)
        loss_klz = 0.5 * jnp.sum(mean1_sqr + sigma_1**2 - jnp.log(sigma_1**2) - 1.0, axis=(1, 2, 3))

        # 3. DIFFUSION LOSS
        # sample time steps
        rng1 = self.make_rng("sample")
        if self.config.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / n_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(n_batch,))

        if self.importance:
            if self.schedule == "VP":
                Z = jnp.log((1 + jnp.exp(-g_0)) / (1 + jnp.exp(-g_1)))
                g_t = jnp.log(1 / (jnp.exp(-Z * t) * (1 + jnp.exp(-g_0)) - 1))
            elif self.schedule == "SP":
                Z = self.likelihood_importance_cum_weight(g_0, g_1)
                g_t = self.sample_importance_weighted_time_for_likelihood(g_0, g_1, Z * t)
        else:
            g_t = self.gamma(t)

        alpha_t_raw, sigma_t_raw = self.noise_schedule(g_t)
        alpha_t, sigma_t = alpha_t_raw[:, None, None, None], sigma_t_raw[:, None, None, None]
        eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_t = alpha_t * f + sigma_t * eps

        if self.second_order:

            def value_div_fn(fn, x, t, eps, cond, deter):
                def value_grad_fn(data):
                    f = fn(data, t, cond, deter)
                    return jnp.sum(f * eps), f

                grad_fn_eps, value = jax.grad(value_grad_fn, has_aux=True)(x)
                return value, jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

            step_rng = self.make_rng("sample")
            if hutchinson_type == "Gaussian":
                v = jax.random.normal(step_rng, f.shape)
            elif hutchinson_type == "Rademacher":
                v = jax.random.rademacher(step_rng, f.shape, dtype=jnp.float32)
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            F_hat, F_div = value_div_fn(self.score_model, z_t, g_t, v, conditioning, deterministic)
        else:
            F_hat = self.score_model(z_t, g_t, conditioning, deterministic)

        if self.velocity:
            if self.schedule == "VP":
                F_target = (eps - sigma_t * z_t) / alpha_t
                weight = 0.5 * Z if self.importance else 0.5 * (g_1 - g_0) * alpha_t_raw**2
            elif self.schedule == "SP":
                F_target = (eps - z_t) / alpha_t / np.sqrt(2)
                weight = Z if self.importance else (g_1 - g_0) * alpha_t_raw**2
        else:
            F_target = eps
            weight = 0.5 * (g_1 - g_0)

        loss_diff_mse = jnp.sum(jnp.square(F_target - F_hat), axis=[1, 2, 3])
        loss_diff = weight * loss_diff_mse

        if self.second_order:
            dim = int(np.prod(f.shape[1:]))
            if self.schedule == "VP":
                loss_diff2 = sigma_t_raw * F_div - alpha_t_raw * (dim - jax.lax.stop_gradient(loss_diff_mse))
            elif self.schedule == "SP":
                loss_diff2 = (
                    sigma_t_raw * F_div
                    - dim / np.sqrt(2)
                    + np.sqrt(2) * alpha_t_raw * jax.lax.stop_gradient(loss_diff_mse)
                )
            loss_diff2 = jnp.square(loss_diff2) / dim * second_order_weight
            loss_diff2 = weight * loss_diff2
        else:
            loss_diff2 = jnp.zeros_like(loss_diff)

        return VDMOutput(
            loss_recon=loss_recon,
            loss_klz=loss_klz,
            loss_diff=loss_diff,
            loss_diff2=loss_diff2,
            var_0=sigma_0**2,
            var_1=sigma_1**2,
        )

    def generate_x(self, z_0):
        g_0 = self.gamma(0.0)
        alpha_0, sigma_0 = self.noise_schedule(g_0)
        z_0_rescaled = z_0 / alpha_0

        logits = self.encdec.decode(z_0_rescaled, g_0)

        # get output samples
        if self.config.sample_softmax:
            out_rng = self.make_rng("sample")
            samples = jax.random.categorical(out_rng, logits)
        else:
            samples = jnp.argmax(logits, axis=-1)

        return samples

    def p_generate_x(self, z_0):
        p_func = jax.pmap(self.generate_x)
        return p_func(z_0)

    def noise_fn(self, x, gamma, conditioning, deterministic=True):
        if self.velocity:
            alpha_t, sigma_t = self.noise_schedule(gamma)
            F_hat = self.score_model(x, gamma, conditioning, deterministic=True)
            if self.schedule == "VP":
                eps_hat = alpha_t[:, None, None, None] * F_hat + sigma_t[:, None, None, None] * x
            elif self.schedule == "SP":
                eps_hat = jnp.sqrt(2) * alpha_t[:, None, None, None] * F_hat + x
        else:
            eps_hat = self.score_model(x, gamma, conditioning, deterministic=deterministic)
        return eps_hat

    def ode(self, x, gamma, conditioning):
        alpha_t, sigma_t = self.noise_schedule(gamma)
        log_alpha_t_hat = self.log_alpha_hat(gamma)
        drift = log_alpha_t_hat[:, None, None, None] * x + 0.5 * sigma_t[:, None, None, None] * self.noise_fn(
            x, gamma, conditioning
        )
        return drift

    def ode_sampler(self, z, conditioning, rng, hutchinson_type="Rademacher", rtol=1e-5, atol=1e-5, method="RK45"):
        @jax.pmap
        def p_value_fn(x, t, cond):
            """Pmapped divergence of the drift function."""
            return self.ode(x, t, cond)

        def to_flattened_numpy(x):
            """Flatten a JAX array `x` and convert it to numpy."""
            return np.asarray(x.reshape((-1,)), dtype=np.float64)

        def from_flattened_numpy(x, shape):
            """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
            return jnp.asarray(x, dtype=jnp.float32).reshape(shape)

        def ode_sampler_fn(prng, z, conditioning):
            shape = z.shape
            N = np.prod(shape[2:])

            def ode_func(t, x):
                sample = from_flattened_numpy(x, shape)
                vec_t = jnp.ones((sample.shape[0], sample.shape[1])) * t
                drift = p_value_fn(sample, vec_t, conditioning)
                drift = to_flattened_numpy(drift)
                return drift

            g_0 = self.gamma(0.0)
            g_1 = self.gamma(1.0)
            alpha_1, sigma_1 = self.noise_schedule(g_1)

            init = to_flattened_numpy(z * sigma_1)
            solution = integrate.solve_ivp(ode_func, (g_1, g_0), init, rtol=rtol, atol=atol, method=method)
            print("nfe", solution.nfev)
            nfe = solution.nfev
            t = solution.t
            zp = jnp.asarray(solution.y[:, -1])
            x = from_flattened_numpy(zp, shape)
            return x, nfe

        return ode_sampler_fn(rng, z, conditioning)

    def likelihood(self, images, conditioning, rng, hutchinson_type="Rademacher", rtol=1e-5, atol=1e-5, method="RK45"):
        def get_value_div_fn(fn):
            """Return both the function value and its estimated divergence via Hutchinson's trace estimator."""

            def value_div_fn(x, t, eps, cond):
                def value_grad_fn(data):
                    f = fn(data, t, cond)
                    return jnp.sum(f * eps), f

                grad_fn_eps, value = jax.grad(value_grad_fn, has_aux=True)(x)
                return value, jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

            return value_div_fn

        @jax.pmap
        def p_value_div_fn(x, t, eps, cond):
            """Pmapped divergence of the drift function."""
            value_div_fn = get_value_div_fn(lambda x, t, cond: self.ode(x, t, cond))
            return value_div_fn(x, t, eps, cond)

        @jax.pmap
        def p_prior_logp_fn(z):
            _, sigma_1 = self.noise_schedule(self.gamma(1.0))
            shape = z.shape
            N = np.prod(shape[1:])
            logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi * sigma_1**2) - jnp.sum(z**2) / (2.0 * sigma_1**2)
            return jax.vmap(logp_fn)(z)

        def to_flattened_numpy(x):
            """Flatten a JAX array `x` and convert it to numpy."""
            return np.asarray(x.reshape((-1,)), dtype=np.float64)

        def from_flattened_numpy(x, shape):
            """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
            return jnp.asarray(x, dtype=jnp.float32).reshape(shape)

        def likelihood_fn(prng, data, conditioning):
            """Compute an unbiased estimate to the log-likelihood in bits/dim.

            Args:
              prng: An array of random states. The list dimension equals the number of devices.
              pstate: Replicated training state for running on multiple devices.
              data: A JAX array of shape [#devices, batch size, ...].

            Returns:
              bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
              z: A JAX array of the same shape as `data`. The latent representation of `data` under the
                probability flow ODE.
              nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
            """
            rng, step_rng = jax.random.split(flax.jax_utils.unreplicate(prng))
            shape = data.shape
            N = np.prod(shape[2:])
            if hutchinson_type == "Gaussian":
                epsilon = jax.random.normal(step_rng, shape)
            elif hutchinson_type == "Rademacher":
                epsilon = jax.random.randint(step_rng, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = from_flattened_numpy(x[: -shape[0] * shape[1]], shape)
                vec_t = jnp.ones((sample.shape[0], sample.shape[1])) * t
                drift, logp_grad = p_value_div_fn(sample, vec_t, epsilon, conditioning)
                drift = to_flattened_numpy(drift)
                logp_grad = to_flattened_numpy(logp_grad)
                return np.concatenate([drift, logp_grad], axis=0)

            K = self.num_importance
            eval_type = self.dequantization
            repeat = False
            if K == 1:
                K = 5
                repeat = True
            g_0 = self.gamma(0.0)
            logps = []
            if eval_type == "u":
                # test by uniform dequantization
                g_0 = -11.8 # Remark A.2 in the paper
                for _ in range(K):
                    rng, step_rng = jax.random.split(rng)
                    u = jax.random.uniform(step_rng, data.shape, dtype=jnp.float32)
                    x = 2 * ((data.round() + u) / 256.0) - 1

                    init = jnp.concatenate([to_flattened_numpy(x), np.zeros((shape[0] * shape[1],))], axis=0)
                    solution = integrate.solve_ivp(
                        ode_func, (g_0, self.gamma(1.0)), init, rtol=rtol, atol=atol, method=method
                    )
                    print("nfe", solution.nfev)
                    nfe = solution.nfev
                    t = solution.t
                    zp = jnp.asarray(solution.y[:, -1])
                    z = from_flattened_numpy(zp[: -shape[0] * shape[1]], shape)
                    delta_logp = zp[-shape[0] * shape[1] :].reshape((shape[0], shape[1]))
                    prior_logp = p_prior_logp_fn(z)

                    logp = prior_logp + delta_logp
                    logps.append(logp)

            elif eval_type == "v":
                # test by variational
                alpha_0, sigma_0 = self.noise_schedule(g_0)
                p_logprob = jax.pmap(lambda x, z_0_rescaled: self.encdec.logprob(x, z_0_rescaled, g_0))
                logq_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi * sigma_0**2) - jnp.sum(z**2) / 2.0
                logq_fn = jax.vmap(logq_fn)
                p_logq_fn = jax.pmap(logq_fn)
                for _ in range(K):
                    rng, step_rng = jax.random.split(rng)
                    z_eps = jax.random.normal(step_rng, data.shape, dtype=jnp.float32)
                    x = 2 * ((data.round() + 0.5) / 256.0) - 1
                    z_0_rescaled = x + jnp.exp(0.5 * g_0) * z_eps  # = z_0/sqrt(1-var)
                    loss_recon = p_logprob(data, z_0_rescaled)
                    x = alpha_0 * x + sigma_0 * z_eps

                    init = jnp.concatenate([to_flattened_numpy(x), np.zeros((shape[0] * shape[1],))], axis=0)
                    solution = integrate.solve_ivp(
                        ode_func, (self.gamma(0.0), self.gamma(1.0)), init, rtol=rtol, atol=atol, method=method
                    )
                    print("nfe", solution.nfev)
                    nfe = solution.nfev
                    t = solution.t
                    zp = jnp.asarray(solution.y[:, -1])
                    z = from_flattened_numpy(zp[: -shape[0] * shape[1]], shape)
                    delta_logp = zp[-shape[0] * shape[1] :].reshape((shape[0], shape[1]))
                    prior_logp = p_prior_logp_fn(z)
                    if K == 1 or repeat:
                        logq = -N / 2.0 * (1 + jnp.log(2 * np.pi * sigma_0**2))
                    else:
                        logq = p_logq_fn(z_eps).reshape(shape[0], shape[1])
                    logp = prior_logp + delta_logp + loss_recon - logq
                    logps.append(logp)
            elif eval_type == "tn":
                # test by truncated-normal dequantization

                alpha_0, sigma_0 = self.noise_schedule(g_0)
                tau = alpha_0 / sigma_0 / 256
                Z = jax.scipy.special.erf(tau / np.sqrt(2))
                logq_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi * sigma_0**2 * Z**2) - jnp.sum(z**2) / 2.0
                logq_fn = jax.vmap(logq_fn)
                p_logq_fn = jax.pmap(logq_fn)
                for _ in range(K):
                    rng, step_rng = jax.random.split(rng)

                    z_eps = jax.random.truncated_normal(step_rng, -tau, tau, data.shape, dtype=jnp.float32)
                    x = 2 * ((data.round() + 0.5) / 256.0) - 1
                    x = alpha_0 * x + sigma_0 * z_eps

                    init = jnp.concatenate([to_flattened_numpy(x), np.zeros((shape[0] * shape[1],))], axis=0)
                    solution = integrate.solve_ivp(
                        ode_func, (self.gamma(0.0), self.gamma(1.0)), init, rtol=rtol, atol=atol, method=method
                    )
                    print("nfe", solution.nfev)
                    nfe = solution.nfev
                    t = solution.t
                    zp = jnp.asarray(solution.y[:, -1])
                    z = from_flattened_numpy(zp[: -shape[0] * shape[1]], shape)
                    delta_logp = zp[-shape[0] * shape[1] :].reshape((shape[0], shape[1]))
                    prior_logp = p_prior_logp_fn(z)
                    if K == 1 or repeat:
                        logq = (
                            -N / 2.0 * (1 + jnp.log(2 * np.pi * sigma_0**2))
                            - N * jnp.log(Z)
                            + N * tau / (jnp.sqrt(2 * np.pi) * Z) * jnp.exp(-0.5 * tau**2)
                        )
                    else:
                        logq = p_logq_fn(z_eps).reshape(shape[0], shape[1])
                    logp = prior_logp + delta_logp - logq
                    logps.append(logp)
            else:
                raise Exception("Unknown dequantization method.")
            if repeat:
                logp = jnp.mean(jnp.stack(logps), axis=0)
            else:
                logp = jax.scipy.special.logsumexp(jnp.stack(logps), axis=0)

            bpd = -logp / N / np.log(2.0)
            if eval_type == "u":
                # A hack to convert log-likelihoods to bits/dim
                # based on the gradient of the inverse data normalizer.
                bpd += 7.0
            return bpd

        return likelihood_fn(rng, images, conditioning)


######### Encoder and decoder #########


class EncDec(nn.Module):
    """Encoder and decoder."""

    config: VDMConfig

    def __call__(self, x, g_0):
        # For initialization purposes
        h = self.encode(x)
        return self.decode(h, g_0)

    def encode(self, x):
        # This transforms x from discrete values (0, 1, ...)
        # to the domain (-1,1).
        # Rounding here just a safeguard to ensure the input is discrete
        # (although typically, x is a discrete variable such as uint8)
        x = x.round()
        return 2 * ((x + 0.5) / self.config.vocab_size) - 1

    def decode(self, z, g_0):
        config = self.config

        # Logits are exact if there are no dependencies between dimensions of x
        x_vals = jnp.arange(0, config.vocab_size)[:, None]
        x_vals = jnp.repeat(x_vals, 3, 1)
        x_vals = self.encode(x_vals).transpose([1, 0])[None, None, None, :, :]
        inv_stdev = jnp.exp(-0.5 * g_0)
        if self.config.schedule == "VP":
            alpha_0 = jnp.sqrt(1 - nn.sigmoid(g_0))
        elif self.config.schedule == "SP":
            alpha_0 = nn.sigmoid(-g_0 / 2)
        logits = -0.5 * jnp.square((z[..., None] / alpha_0 - x_vals) * inv_stdev)

        logprobs = jax.nn.log_softmax(logits)
        return logprobs

    def logprob(self, x, z, g_0):
        x = x.round().astype("int32")
        x_onehot = jax.nn.one_hot(x, self.config.vocab_size)
        logprobs = self.decode(z, g_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
        return logprob


######### Score model #########


class ScoreUNet(nn.Module):
    config: VDMConfig

    @nn.compact
    def __call__(self, z, g_t, conditioning, deterministic=True):
        config = self.config

        # Compute conditioning vector based on 'g_t' and 'conditioning'
        n_embd = self.config.sm_n_embd

        lb = config.gamma_min
        ub = config.gamma_max

        t = (g_t - lb) / (ub - lb)  # ---> [0,1]

        assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        if jnp.isscalar(t):
            t = jnp.ones((z.shape[0],), z.dtype) * t
        elif len(t.shape) == 0:
            t = jnp.tile(t[None], z.shape[0])

        temb = get_timestep_embedding(t, n_embd)
        cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
        cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense0")(cond))
        cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense1")(cond))

        # Concatenate Fourier features to input
        if config.with_fourier_features:
            z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
            h = jnp.concatenate([z, z_f], axis=-1)
        else:
            h = z

        # Linear projection of input
        h = nn.Conv(features=n_embd, kernel_size=(3, 3), strides=(1, 1), name="conv_in")(h)
        hs = [h]

        # Downsampling
        for i_block in range(self.config.sm_n_layer):
            block = ResnetBlock(config, out_ch=n_embd, name=f"down.block_{i_block}")
            h = block(hs[-1], cond, deterministic)[0]
            if config.with_attention:
                h = AttnBlock(num_heads=1, name=f"down.attn_{i_block}")(h)
            hs.append(h)

        # Middle
        h = hs[-1]
        h = ResnetBlock(config, name="mid.block_1")(h, cond, deterministic)[0]
        h = AttnBlock(num_heads=1, name="mid.attn_1")(h)
        h = ResnetBlock(config, name="mid.block_2")(h, cond, deterministic)[0]

        # Upsampling
        for i_block in range(self.config.sm_n_layer + 1):
            b = ResnetBlock(config, out_ch=n_embd, name=f"up.block_{i_block}")
            h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
            if config.with_attention:
                h = AttnBlock(num_heads=1, name=f"up.attn_{i_block}")(h)

        assert not hs

        # Predict noise
        normalize = nn.normalization.GroupNorm()
        h = nn.swish(normalize(h))
        eps_pred = nn.Conv(
            features=z.shape[-1], kernel_size=(3, 3), strides=(1, 1), kernel_init=nn.initializers.zeros, name="conv_out"
        )(h)

        # Base measure
        eps_pred += z

        return eps_pred


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
      timesteps: jnp.ndarray: generate embedding vectors at these timesteps
      embedding_dim: int: dimension of the embeddings to generate
      dtype: data type of the generated embeddings

    Returns:
      embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1
    timesteps *= 1000.0

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


######### Noise Schedule #########


class NoiseSchedule_FixedLinear(nn.Module):
    config: VDMConfig

    @nn.compact
    def __call__(self, t):
        config = self.config
        return config.gamma_min + (config.gamma_max - config.gamma_min) * t


######### ResNet block #########


class ResnetBlock(nn.Module):
    """Convolutional residual block with two convs."""

    config: VDMConfig
    out_ch: Optional[int] = None

    @nn.compact
    def __call__(self, x, cond, deterministic: bool, enc=None):
        config = self.config

        nonlinearity = nn.swish
        normalize1 = nn.normalization.GroupNorm()
        normalize2 = nn.normalization.GroupNorm()

        if enc is not None:
            x = jnp.concatenate([x, enc], axis=-1)

        B, _, _, C = x.shape  # pylint: disable=invalid-name
        out_ch = C if self.out_ch is None else self.out_ch

        h = x
        h = nonlinearity(normalize1(h))
        h = nn.Conv(features=out_ch, kernel_size=(3, 3), strides=(1, 1), name="conv1")(h)

        # add in conditioning
        if cond is not None:
            assert cond.shape[0] == B and len(cond.shape) == 2
            h += nn.Dense(features=out_ch, use_bias=False, kernel_init=nn.initializers.zeros, name="cond_proj")(cond)[
                :, None, None, :
            ]

        h = nonlinearity(normalize2(h))
        h = nn.Dropout(rate=config.sm_pdrop)(h, deterministic=deterministic)
        h = nn.Conv(
            features=out_ch, kernel_size=(3, 3), strides=(1, 1), kernel_init=nn.initializers.zeros, name="conv2"
        )(h)

        if C != out_ch:
            x = nn.Dense(features=out_ch, name="nin_shortcut")(x)

        assert x.shape == h.shape
        x = x + h
        return x, x


class AttnBlock(nn.Module):
    """Self-attention residual block."""

    num_heads: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
        assert C % self.num_heads == 0

        normalize = nn.normalization.GroupNorm()

        h = normalize(x)
        if self.num_heads == 1:
            q = nn.Dense(features=C, name="q")(h)
            k = nn.Dense(features=C, name="k")(h)
            v = nn.Dense(features=C, name="v")(h)
            h = dot_product_attention(q[:, :, :, None, :], k[:, :, :, None, :], v[:, :, :, None, :], axis=(1, 2))[
                :, :, :, 0, :
            ]
            h = nn.Dense(features=C, kernel_init=nn.initializers.zeros, name="proj_out")(h)
        else:
            head_dim = C // self.num_heads
            q = nn.DenseGeneral(features=(self.num_heads, head_dim), name="q")(h)
            k = nn.DenseGeneral(features=(self.num_heads, head_dim), name="k")(h)
            v = nn.DenseGeneral(features=(self.num_heads, head_dim), name="v")(h)
            assert q.shape == k.shape == v.shape == (B, H, W, self.num_heads, head_dim)
            h = dot_product_attention(q, k, v, axis=(1, 2))
            h = nn.DenseGeneral(features=C, axis=(-2, -1), kernel_init=nn.initializers.zeros, name="proj_out")(h)

        assert h.shape == x.shape
        return x + h


def dot_product_attention(
    query,
    key,
    value,
    dtype=jnp.float32,
    bias=None,
    axis=None,
    # broadcast_dropout=True,
    # dropout_rng=None,
    # dropout_rate=0.,
    # deterministic=False,
    precision=None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights. This
    function supports multi-dimensional inputs.


    Args:
      query: queries for calculating attention with shape of `[batch_size, dim1,
        dim2, ..., dimN, num_heads, mem_channels]`.
      key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
        ..., dimN, num_heads, mem_channels]`.
      value: values to be used in attention with shape of `[batch_size, dim1,
        dim2,..., dimN, num_heads, value_channels]`.
      dtype: the dtype of the computation (default: float32)
      bias: bias for the attention weights. This can be used for incorporating
        autoregressive mask, padding mask, proximity bias.
      axis: axises over which the attention is applied.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
    """
    assert key.shape[:-1] == value.shape[:-1]
    assert query.shape[0:1] == key.shape[0:1] and query.shape[-1] == key.shape[-1]
    assert query.dtype == key.dtype == value.dtype
    input_dtype = query.dtype

    if axis is None:
        axis = tuple(range(1, key.ndim - 2))
    if not isinstance(axis, Iterable):
        axis = (axis,)
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim
    for ax in axis:
        if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
            raise ValueError("Attention axis must be between the batch " "axis and the last-two axes.")
    depth = query.shape[-1]
    n = key.ndim
    # batch_dims is  <bs, <non-attention dims>, num_heads>
    batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
    # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    qk_perm = batch_dims + axis + (n - 1,)
    key = key.transpose(qk_perm)
    query = query.transpose(qk_perm)
    # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
    v_perm = batch_dims + (n - 1,) + axis
    value = value.transpose(v_perm)

    key = key.astype(dtype)
    query = query.astype(dtype) / np.sqrt(depth)
    batch_dims_t = tuple(range(len(batch_dims)))
    attn_weights = jax.lax.dot_general(
        query, key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)), precision=precision
    )

    # apply attention bias: masking, droput, proximity bias, ect.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
    attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
    assert attn_weights.dtype == dtype
    attn_weights = attn_weights.astype(input_dtype)

    # compute the new values given the attention weights
    assert attn_weights.dtype == value.dtype
    wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
    y = jax.lax.dot_general(
        attn_weights, value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)), precision=precision
    )

    # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
    perm_inv = _invert_perm(qk_perm)
    y = y.transpose(perm_inv)
    assert y.dtype == input_dtype
    return y


def _invert_perm(perm):
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
        perm_inv[j] = i
    return tuple(perm_inv)


class Base2FourierFeatures(nn.Module):
    start: int = 0
    stop: int = 8
    step: int = 1

    @nn.compact
    def __call__(self, inputs):
        freqs = range(self.start, self.stop, self.step)

        # Create Base 2 Fourier features
        w = 2.0 ** (jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
        w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

        # Compute features
        h = jnp.repeat(inputs, len(freqs), axis=-1)
        h = w * h
        h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
        return h
