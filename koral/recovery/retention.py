"""Development-only retention gate with a predeclared interpolation grid.

A finite development-bank acceptance check is not a safety or population
nondegradation guarantee. Its rollouts and selection decisions are fully logged.
"""
import copy
import torch

def interpolate(previous,proposal,weight):
    if not 0<=weight<=1:raise ValueError('interpolation weight outside [0,1]')
    out=copy.deepcopy(previous)
    with torch.no_grad():
        for p,old,new in zip(out.parameters(),previous.parameters(),proposal.parameters()):
            p.copy_((1-weight)*old+weight*new)
    out.eval();return out

def admit(previous,proposal,evaluator,utility,grid=(1.,.5,.25,0.),tolerance=0.):
    if not grid or grid[-1]!=0 or any(w<0 or w>1 for w in grid):raise ValueError('grid must end in zero')
    baseline=evaluator(previous);base=utility(baseline);logs=[]
    for weight in grid:
        if weight==0:
            logs.append({'weight':0.,'utility':base,'records':baseline,'accepted':True,'baseline_records':baseline,'baseline_utility':base})
            return copy.deepcopy(previous),logs
        candidate=interpolate(previous,proposal,weight);rows=evaluator(candidate);value=utility(rows)
        success_ok=sum(r['success'] for r in rows)>=sum(r['success'] for r in baseline)
        keep=bool(success_ok and value>=base-tolerance)
        logs.append({'weight':weight,'utility':value,'records':rows,'accepted':keep,'baseline_records':baseline,'baseline_utility':base})
        if keep:return candidate,logs
    raise RuntimeError('missing fallback')
