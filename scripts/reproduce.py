"""Run the complete KORAL experiment from initial demonstrations."""
import argparse
from koral.recovery.workflow import reproduce

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/local_recovery.json')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    reproduce(a.config,a.out)
