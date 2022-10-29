import argparse
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=
        'Plot results from scan_evaluate.py to visualize infilling perplexity as a function of span position along heavy/light chain sequence'
    )
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    parser.add_argument('scan_h5',
                        type=str,
                        help='h5 output from scan_evaluate.py')
    parser.add_argument('--output_dir',
                        type=str,
                        default=f'output_dir/plot_scan_{now}')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    h5_in = h5py.File(args.scan_h5, 'r')

    # Load in losses from h5
    heavy_scores = np.array(h5_in['Heavy_losses'])
    light_scores = np.array(h5_in['Light_losses'])

    # Compute perplexity
    heavy_scores = np.exp(heavy_scores)
    light_scores = np.exp(light_scores)

    heavy_mask = np.array(h5_in['Heavy_num_samples_per_pos']) <= 50
    light_mask = np.array(h5_in['Light_num_samples_per_pos']) <= 50
    heavy_scores[heavy_mask] = np.nan
    light_scores[light_mask] = np.nan

    h5_in.close()

    # Plots
    plt.figure()
    plt.plot(heavy_scores, color='blueviolet', label='Heavy chain')
    plt.plot(light_scores, color='royalblue', label='Light chain')
    plt.legend(fontsize=16)
    skip_10 = np.arange(0, heavy_scores.shape[0], 10)
    plt.xticks(skip_10, skip_10)
    plt.ylabel('Model infilling perplexity', fontsize=14)
    plt.xlabel('Mask position along sequence', fontsize=14)
    plt.savefig(f'{args.output_dir}/scan.pdf', dpi=400, transparent=True)
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
