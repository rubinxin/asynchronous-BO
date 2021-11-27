"""Misc utilities"""

import multiprocessing as mp
import time

import numpy as np


def timed_print(*args):
    """
    Utility function that adds the process name as well as the current
    time to any message.
    """
    mssg = ' '.join([str(a) for a in args])
    s = time.strftime('%Y-%m-%d %H:%M:%S')
    pn = mp.current_process().name
    print("{} -- {}: {}".format(s, pn, mssg))


def print_experiment_times(durations, total):
    """
    prints some info about how long each experiment takes and how long
    it will take to complete all of them

    durations = list with times in seconds of completed experiments
    total = int total number of planned experiments
    """
    mean_duration = np.mean(durations)
    m, s = divmod(mean_duration, 60)
    print("Average duration of one experiment = %02d:%02d" % (m, s))

    exps_left = total - len(durations)
    time_left = exps_left * mean_duration
    m_t, s_t = divmod(time_left, 60)
    if time_left > 3600:
        h_t, m_t = divmod(m_t, 60)
        print("Estimated time left = %02d:%02d:%02d" % (h_t, m_t, s_t))
    else:
        print("Estimated time left = %02d:%02d" % (m_t, s_t))
