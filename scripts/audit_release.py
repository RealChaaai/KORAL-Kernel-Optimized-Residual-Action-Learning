"""Check model hashes, expert accounting and terminal outcomes from trajectories."""
import argparse,json
from koral.recovery.verification import audit_results

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/reference')
    a=p.parse_args();print(json.dumps(audit_results(a.results),indent=2))
