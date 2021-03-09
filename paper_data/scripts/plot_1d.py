#! /usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys
import os
import math

matplotlib.rcParams.update({'font.size': 10})


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Input file')
parser.add_argument('--output', required=True, help='Output file')
parser.add_argument('--colorbar', action='store_true', help='Plot color bar')

args = parser.parse_args()

df = pd.read_csv(args.input, sep=';', header=None)

normalized = df[1].values.copy()

min_value = normalized.min()
max_value = normalized.max()

plt.rcParams['axes.facecolor'] = 'grey'
plt.rcParams['axes.facecolor'] = '#a8a8a8'

cmap = LinearSegmentedColormap.from_list("mycmap", ((0.0, 'blue'), (0.5, 'w'), (1.0, 'r'),), N=8192)

for i in range(len(normalized)):
    if normalized[i] < 0.0:
        normalized[i] = -(normalized[i] - min_value) / (2 * min_value)
    elif normalized[i] == 0.0:
        normalized[i] = 0.5
    else:
        normalized[i] = normalized[i] / (2 * max_value) + 0.5

plt.figure(figsize=(3.33, 3.33), dpi=600)
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.scatter(df[0].values, df[1].values, c=normalized, cmap=cmap, marker='.', s=1.5)
plt.xlabel('particle distance')
plt.ylabel('score')

if args.colorbar:
    cbar = plt.colorbar(ticks=(0.0, 0.5, 1.0))
    labels = [f"{v:.1f}" for v in (min_value, 0, max_value)]
    cbar.ax.set_yticklabels(labels)

xticks = (0, math.pow(2.0, 1.0/6.0), 1.7,)
xlabels = ('0', '$\sqrt[6]{2}$', 1.7,)
yticks = (-0.1, 0.0, 4.0)
ylabels = ('-0.1', '', '4.0')
plt.xticks(xticks, xlabels)
plt.yticks(yticks, ylabels)

plt.gca().set_axisbelow(True)
plt.grid(c='gray')
plt.gca().set_title('(a)', loc='left')
plt.savefig(args.output, bbox_inches='tight')
