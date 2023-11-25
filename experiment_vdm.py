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

import numpy as np
import jax.numpy as jnp
from jax._src.random import PRNGKey
import jax
from typing import Any, Tuple

from experiment import Experiment
import model_vdm


class Experiment_VDM(Experiment):
    """Train and evaluate a VDM model."""

    def get_model_and_params(self, rng: PRNGKey):
        config = self.config
        config = model_vdm.VDMConfig(**config.model)
        model = model_vdm.VDM(config)

        inputs = {"images": jnp.zeros((2, 32, 32, 3), "uint8")}
        inputs["conditioning"] = jnp.zeros((2,))
        rng1, rng2 = jax.random.split(rng)
        params = model.init({"params": rng1, "sample": rng2}, **inputs)
        return model, params

    def loss_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
        rng, sample_rng = jax.random.split(rng)
        rngs = {"sample": sample_rng}
        if is_train:
            rng, dropout_rng = jax.random.split(rng)
            rngs["dropout"] = dropout_rng

        # sample time steps, with antithetic sampling
        outputs = self.state.apply_fn(
            variables={"params": params},
            **inputs,
            rngs=rngs,
            deterministic=not is_train,
        )

        rescale_to_bpd = 1.0 / (np.prod(inputs["images"].shape[1:]) * np.log(2.0))
        bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
        bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
        bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
        bpd_diff2 = jnp.mean(outputs.loss_diff2) * rescale_to_bpd
        bpd = bpd_recon + bpd_latent + bpd_diff
        if is_train:
            bpd = bpd + bpd_diff2
        scalar_dict = {
            "bpd": bpd,
            "bpd_latent": bpd_latent,
            "bpd_recon": bpd_recon,
            "bpd_diff": bpd_diff,
            "bpd_diff2": bpd_diff2,
            "var0": outputs.var_0,
            "var": outputs.var_1,
            "loss_diff": bpd_diff,
        }
        img_dict = {"inputs": inputs["images"]}
        metrics = {"scalars": scalar_dict, "images": img_dict}

        return bpd, metrics

    def sample_fn(self, *, dummy_inputs, rng, params, z=None):
        conditioning = jnp.zeros((dummy_inputs.shape[0], dummy_inputs.shape[1]), dtype="uint8")

        rng, sample_rng = jax.random.split(rng)
        z_init = jax.random.normal(sample_rng, dummy_inputs.shape)
        z_0, nfe = self.state.apply_fn(
            variables={"params": params},
            z=z_init,
            conditioning=conditioning,
            rng=rng,
            method=self.model.ode_sampler,
        )
        samples = self.state.apply_fn(
            variables={"params": params},
            z_0=z_0,
            method=self.model.p_generate_x,
        )
        return samples, nfe

    def likelihood_fn(self, *, inputs, rng, params):
        bpd = self.state.apply_fn(
            variables={"params": params},
            **inputs,
            rng=rng,
            method=self.model.likelihood,
        )
        return bpd
