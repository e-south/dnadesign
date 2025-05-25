"""
--------------------------------------------------------------------------------
<dnadesign project>
overlap/optimizer.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pymc as pm, numpy as np, arviz as az
from .base import Optimizer
from ..persistence.traces import save_trace

class GibbsOptimizer(Optimizer):
    """Discrete MCMC over ACGT^L with a combined PWM log-p"""

    def _logp(self, dna):                # dna = int array( L )
        return self.cfg["beta"] * self.scorer.score(dna)

    def optimise(self, initial):
        L      = len(initial.seq)
        logp_fn= self._logp
        beta   = self.cfg["beta"]
        draws  = self.cfg["draws"]
        tune   = self.cfg["tune"]

        with pm.Model() as model:
            dna = pm.Categorical("dna", p=np.ones(4)/4,
                                 shape=L, logp=logp_fn,
                                 initval=initial.seq)
            trace = pm.sample(draws=draws,
                              tune=tune,
                              step=pm.CategoricalGibbsMetropolis(
                                       proposal="proportional"),
                              chains=self.cfg["chains"],
                              cores=self.cfg["cores"],
                              random_seed=self.rng.integers(2**32))

        save_trace(trace, self.cfg["trace_path"])        # ⇢ persistence/
        unique_ranked = self._postprocess(trace)
        return unique_ranked
