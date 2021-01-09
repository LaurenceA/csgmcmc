"""
Probability distribution for consensus generative processes
note that we use x=0 for noconsensus,
then we can represent the possibility of classes x=1...K and noconsensus
using a base Categorical distribution.
"""

import math
import torch as t
import torch.distributions as td
import torch.nn as nn

__all__ = ["ConsensusUnlabelled", "ConsensusLabelled", "ConsensusBernoulliUnlabelled", "ConsensusBernoulliLabelled"]

def arange_without_i(i, n):
    """
    returns t.arange(n) without index i, i.e.
    [0,...,i-1,i+1,...,n-1]
    """
    return t.cat([t.arange(i), t.arange(i+1, n)])

def arange_without_all_i(n):
    return t.stack([arange_without_i(i, n) for i in range(n)], 0)

def log_prob_noconsensus(lp, S):
    """
    consensus probability = sum_y p_y^S

    while we could use noconsensus probability = 1 - consensus probability
    that turns out to be numerically unstable.
    Instead, for S=4, use,
    \sum_y (p_y (1-p_y) + p_y^2 (1-p_y) + p_y^3 (1-p_y))
    \sum_y (p_y - p_y^2 + p_y^2 - p_y^3 + p_y^3 - p_y^4)
    \sum_y (p_y - p_y^4)
    1 - \sum_y p_y^4
    Interpretation: 1st             sample is y, 2nd not y, or
                    1st + 2nd       sample is y, 3rd not y, or
                    1st + 2nd + 3rd sample is y, 4th not y

    returns log-probability of noconsensus for each datapoint
    """
    if S==1:
        return -1E10*t.ones(lp.shape[:-1], device=lp.device)
    else:
        # 1-p_y = \sum_{y'\neq y} p_y'
        # compute using logsumexp for better numerical stability
        #[N, K, K-1]
        lpnoty = lp[..., arange_without_all_i(lp.shape[-1])].logsumexp(-1)
        #[N, K, S-1]
        result = t.stack([lpnoty + lp*i for i in range(1, S)], -1)
        return result.view(*result.shape[:-2], -1).logsumexp(-1)

class AbstractConsensusDist():
    def __init__(self, logits, S):
        self.logits = logits
        self.S = S
    def sample(self):
        return self.dist().sample()
    def log_prob(self, x):
        return self.dist().log_prob(x)
    def weighted_log_prob(self, x, noconsensus_weight=1.):
        """
        multiply the log-probability of noconsensus points by a weight
        """
        consensus   = x < 0.5
        noconsensus =     0.5 < x
        weight = noconsensus * self.noconsensus_weight + consensus
        return weight*self.dist.log_prob(x)
      

class ConsensusUnlabelled(AbstractConsensusDist):
    def dist(self):
        """
        returns a Bernoulli distribution, with:
        x=0 => no consensus
        x=1 =>    consensus
        
        log p0 = log e^E0/(e^E0 + e^E1)
        log p1 = log e^E1/(e^E0 + e^E1)
        log p1 - log p0 = log e^E1/(e^E0 + e^E1) - log e^E0/(e^E0 + e^E1)
        log p1 - log p0 = log e^E1 - log e^E0
        log p1 - log p0 = E1 - E0 = logits
        """
        lp = self.logits.log_softmax(-1)

        consensus_lp   = (self.S*lp).logsumexp(-1)
        noconsensus_lp = log_prob_noconsensus(lp, self.S)
        # check that consensus and noconsensus lps are consistent
        assert t.allclose(consensus_lp.exp() + noconsensus_lp.exp(), t.ones(()))
         
        dist = td.Categorical(logits=t.stack([noconsensus_lp, consensus_lp], -1))

        #assert t.allclose(dist.log_prob(t.ones_like(logits)),  consensus_lp)
        #assert t.allclose(dist.log_prob(t.zeros_like(logits)), noconsensus_lp)

        return dist

    def log_prob_entropy_minimization(self):
        """
        entropy minimization bound on principled log_prob.
        ignores no-consensus points.
        """
        consensus = 0.5 < x
        entropy = self.td.Categorical(logits=self.logits).entropy()
        return -(self.S-1) * entropy * consensus

    def log_prob_psuedo_labelling(self, x):
        """
        pseudo-labelling bound on principled log_prob.
        ignores no-consensus points.
        """
        consensus = 0.5 < x
        lp = self.logits.log_softmax(-1)
        return (self.S) * lp.max(-1).values * consensus
        

class ConsensusLabelled(AbstractConsensusDist):
    def dist(self):
        """
        returns a Categorical distribution with K+1 classes, where
        x=0     => no consensus
        x=1...K =>    consensus, class x
        """
        lp = self.logits.log_softmax(-1)

        consensus_lp   = self.S*lp
        noconsensus_lp = log_prob_noconsensus(lp, self.S)[..., None]
        lps = t.cat([noconsensus_lp, consensus_lp], -1)
        #assert t.allclose(lps.exp().sum(-1), t.ones(()))

        return td.Categorical(logits=lps)

    def log_prob_raw(self, x):
        """
        ignore no consensus points and return log-probability of consensus points under
        original categorical
        """
        noconsensus =       x < 0.5
        consensus   = 0.5 < x 
        #Leave noconsensus at zero, reduce all consensus by 1
        x = x - consensus.int()
        return td.Categorical(logits=self.logits).log_prob(x) * consensus

def ConsensusBernoulliLabelled(logits, S):
    return ConsensusLabelled(t.stack([t.zeros_like(logits), logits], -1), S)
def ConsensusBernoulliUnlabelled(logits, S):
    return ConsensusUnlabelled(t.stack([t.zeros_like(logits), logits], -1), S)

if __name__ == "__main__":
    S = 4
    num_classes = 3
    N = 10
    batch = 50
    logits = t.randn(batch, N, num_classes)
    lp = logits.log_softmax(-1)

    P_unlab = ConsensusUnlabelled(logits, S)
    assert t.allclose(P_unlab.log_prob(t.zeros(N)), log_prob_noconsensus(lp, S))

    P_lab = ConsensusLabelled(logits, S)
    assert t.allclose(P_lab.log_prob(t.zeros(N)), log_prob_noconsensus(lp, S))

    C_unlab = P_unlab.sample()
    C_lab = P_lab.sample()




    logits = t.randn(N)
    logitsp = t.stack([t.zeros_like(logits), logits], -1)
    lp = logitsp.log_softmax(-1)

    P_unlab = ConsensusBernoulliUnlabelled(logits, S)
    assert t.allclose(P_unlab.log_prob(t.zeros(N)), log_prob_noconsensus(lp, S))

    P_lab = ConsensusBernoulliLabelled(logits, S)
    assert t.allclose(P_lab.log_prob(t.zeros(N)), log_prob_noconsensus(lp, S))

    C_unlab = P_unlab.sample()
    C_lab = P_lab.sample()

    
    
