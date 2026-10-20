"""Regenerate saved policy rollouts independently of stored outcome flags."""
import argparse,json,torch
from koral.recovery.verification import verify_replay

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/reference')
    p.add_argument('--methods',nargs='+',default=['bc','koral'])
    p.add_argument('--banks',nargs='+',default=['reference','confirmation'])
    a=p.parse_args();torch.set_num_threads(1)
    print(json.dumps(verify_replay(a.results,tuple(a.methods),tuple(a.banks)),indent=2))
