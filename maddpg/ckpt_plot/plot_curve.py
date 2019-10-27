import os
import sys
import numpy as np

import matplotlib as mpl
import csv

mpl.use('Agg')
from matplotlib import pyplot as plt


def avg_list(l, avg_group_size=2):
    ret_l = []
    n = len(l)
    h_size = avg_group_size / 2
    for i in range(n):
        left = int(max(0, i - h_size))
        right = int(min(n, i + h_size))

        ret_l.append(np.mean(l[left:right]))
    return ret_l


def plot_result(t1, r1, fig_name, x_label, y_label):
    plt.close()
    base = None
    base, = plt.plot(t1, avg_list(r1))


    plt.grid()
    #plt.legend([base, teach1, teach2, teach3], ['CA error < 5%', 'CA error < 10%', 'CA error < 20%', 'MADDPG + global count'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')


def plot_result2(t1, r1, r2, fig_name, x_label, y_label):
    plt.close()
    base = None
    l1, = plt.plot(t1, r1)
    l2, = plt.plot(t1, r2)

    plt.grid()
    plt.legend([l1, l2], ['train', 'val'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')


def plot_result_mul(fig_name, x_label, y_label, legend, t1, r1,  t2, r2, t3, r3):
    plt.close()

    l1, = plt.plot(t1, r1)
    if t2 is not None:
        l2, = plt.plot(t2, r2)
    if t3 is not None:
        l3, = plt.plot(t3, r3)
    plt.grid()
    plt.legend([l1, l2, l3], legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)

    try:
        plt.savefig(fig_name + '.pdf')
        plt.savefig(fig_name + '.png')
    except:
        print('ERROR:', sys.exc_info()[0])
        print('Terminate Program')
        sys.exit()
    print('INFO: Wrote plot to ' + fig_name + '.pdf')


def read_csv(csv_path):
    res = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            res[row[0]] = [float(r) for r in row[1:]]
    return res

