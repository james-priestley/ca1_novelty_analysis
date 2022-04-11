import numpy as np


def find_intervals(bool_array, circular=False):
    """Get the start and stop (exclusive) index of all intervals of True in an
    array of booleans.

    Parameters
    ----------
    bool_array : array
    circular : bool, optional
        If True, intervals can wrap around the end of the array
    """

    bool_array = np.asarray(bool_array, dtype='int')
    borders = np.concatenate([[0], np.diff(bool_array)])

    starts = np.where(borders == 1)[0]
    ends = np.where(borders == -1)[0]
    max_bin = len(borders)

    intervals = []

    if not starts.size > 0:
        # there are no positive borders.
        # the interval must start at bin 0, and end at the only negative border
        intervals.append([0, ends[0]])

    elif not ends.size > 0:
        # there are no negative borders.
        # the interval must end at bin 0, and start at the only positive border
        intervals.append([starts[0], max_bin])

    else:
        for s in starts:
            if s > ends[-1]:
                # the interval started after any ends. if circular, the end
                # may wrap around or is max_bin. if not circular, it is max_bin
                if circular & (ends[0] < starts[0]):
                    intervals.append([s, ends[0]])
                else:
                    intervals.append([s, max_bin])
            else:
                # otherwise find the closest end to this start
                valid_ends = ends[ends > s]
                intervals.append([s, valid_ends[np.argmin(valid_ends - s)]])

        if ends[0] < starts[0]:
            # there's an orphaned end (it occurs before any start)
            # if circular, the start may have wrapped around, or it is 0
            if circular & (starts[-1] > (ends[-1])):
                pass
            else:
                intervals.append([0, ends[0]])

    return intervals


def find_first_n(a, n, window=None):
    """Find the index of the first n consecutive non-zero elements in an array.
    Optionally pass a window size, which will find the index of the first n
    non-zero elements contained within a window starting at that index
    (the first index in the window must be non-zero).

    Parameters
    ----------
    a : array
        Input vector
    n : int
        Number of true values in sample window, including the first
        element in the window
    window : int
        Window size
    """

    if window is None:
        window = n

    for index, item in enumerate(a):
        if item:
            count = np.sum(a[index:index + window] > 0)
            if count >= n:
                return index
    return None
