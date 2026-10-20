"""Cost-normalized contextual learning-gain acquisition; no teacher peeking."""
from dataclasses import dataclass
import numpy as np
from .environment import History

FEATURE_DIM=13

@dataclass
class Candidate:
    id: int
    state: dict
    history: History
    disagreement: float
    used: bool=False

def features(c,length,max_length):
    x=c.history.array();o=x[-1]
    d=np.linalg.norm(o[:3]);g=np.linalg.norm(o[3:6]);motion=np.linalg.norm(x[-1,:3]-x[0,:3])
    contacts=np.clip(o[28:30],0,1)
    return np.array([1.,np.clip(c.disagreement*10,0,3),d/.15,g/.15,
        np.clip(motion/.04,0,4),o[6]/.2,o[27]/.04,contacts[0],contacts[1],
        o[-1],length/max_length,(length/max_length)**2,
        np.clip(c.disagreement*10,0,3)*length/max_length],float)

class GainSelector:
    """Discounted ridge/UCB regressor on measured paired developmental gains.

    UCB is an exploration heuristic, not a calibrated probability or formal
    statistical confidence bound in this nonstationary learning process.
    """
    def __init__(self,seed=0,exploration=.2,discount=.96,ridge=1.,ucb=.05):
        self.rng=np.random.default_rng(seed);self.exploration=exploration
        self.discount=discount;self.ridge=ridge;self.ucb=ucb
        self.A=np.eye(FEATURE_DIM)*ridge;self.b=np.zeros(FEATURE_DIM);self.n=0
    def predict(self,x):
        invx=np.linalg.solve(self.A,x)
        return float(x@np.linalg.solve(self.A,self.b)),float(np.sqrt(max(0.,x@invx)))
    def choose(self,candidates,lengths,remaining,mode='gain'):
        if mode not in ('gain','random','uncertainty','fixed_length','no_sequence'):
            raise ValueError('unknown acquisition rule')
        options=[]
        for c in candidates:
            if c.used:continue
            for h in lengths:
                if h<=remaining:options.append((c,h,features(c,h,max(lengths))))
        if not options:raise ValueError('no affordable candidate')
        if mode=='fixed_length':options=[o for o in options if o[1]==min(lengths)]
        explore=self.n<2 or self.rng.random()<self.exploration
        if mode=='random' or (mode in ('gain','fixed_length','no_sequence') and explore):
            i=int(self.rng.integers(len(options)))
        elif mode=='uncertainty':
            i=max(range(len(options)),key=lambda k:options[k][0].disagreement)
        else:
            scores=[self.predict(x) for _,_,x in options]
            i=max(range(len(options)),key=lambda k:scores[k][0]+self.ucb*scores[k][1])
        c,h,x=options[i];mu,sd=self.predict(x)
        return c,h,x,{'predicted_gain_per_action':mu,'exploration_bonus':self.ucb*sd,
                      'exploration':bool(explore),'mode':mode}
    def observe(self,x,paired_gain,charged_actions):
        if charged_actions<1 or not np.isfinite(paired_gain):raise ValueError('invalid gain record')
        # Select using expected gain per expert action, not raw disagreement.
        target=float(paired_gain)/charged_actions
        self.A=self.discount*self.A+(1-self.discount)*self.ridge*np.eye(FEATURE_DIM)+np.outer(x,x)
        self.b=self.discount*self.b+x*target;self.n+=1
        return target
    def state(self):return {'A':self.A.tolist(),'b':self.b.tolist(),'observations':self.n,
        'exploration':self.exploration,'discount':self.discount,'ucb':self.ucb}
