import argparse
import os
from calculate_docs_bias import calculate_docs_bias
from calculate_run_bias import calculate_run_bias
from calculate_cumulative_bias import calculate_cumulative_bias

parser = argparse.ArgumentParser(description='Calculate bias')
parser.add_argument('--biased_run_file', type=str, help='Biased run file')
parser.add_argument('--unbiased_run_file', type=str, help='Unbiased run file')
args = parser.parse_args()

docs_bias_file = 'documents_bias_tf.pkl'
if not os.path.exists(docs_bias_file):
    calculate_docs_bias()

calculate_run_bias(args.biased_run_file, args.unbiased_run_file)
calculate_cumulative_bias()
