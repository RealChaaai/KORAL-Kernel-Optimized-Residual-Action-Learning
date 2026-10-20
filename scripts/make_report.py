import argparse
from koral.recovery.reporting import make_report
p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True)
a=p.parse_args();make_report(a.results,a.out)
