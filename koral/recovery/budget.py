"""Charge before consulting the expert; candidate scoring is label-free."""
from dataclasses import dataclass, field

@dataclass
class Budget:
    limit: int
    expert_actions: int=0
    expert_calls: int=0
    environment_steps: int=0
    acquisition_steps: int=0
    development_steps: int=0
    optimizer_steps: int=0
    events: list=field(default_factory=list)
    def __post_init__(self):
        if self.limit<0:raise ValueError('negative label budget')
    @property
    def remaining(self):return self.limit-self.expert_actions
    def begin(self,candidate_id,length):
        if length<1 or length>self.remaining:raise ValueError('segment exceeds remaining budget')
        self.expert_calls+=1
        self.events.append({'candidate_id':int(candidate_id),'requested':int(length),'charged':0})
    def charge(self):
        if self.remaining<1:raise RuntimeError('expert budget exhausted')
        if not self.events:raise RuntimeError('begin a segment before a query')
        self.expert_actions+=1;self.events[-1]['charged']+=1
    def audit(self):
        assert self.expert_actions<=self.limit
        assert sum(e['charged'] for e in self.events)==self.expert_actions
        assert len(self.events)==self.expert_calls
        return True
