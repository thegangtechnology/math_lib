from typing import Tuple, Union

import numpy as np


def line_point_shortest_dist(r: np.ndarray, v: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """ Find the shortest distance between point p
    and line y(t) = r + v * t

    :param r: a point on the line
    :param v: direction of line
    :param p: interested point
    :return: the shortest distance and t value that gives it
    """

    t = np.dot(v, p - r) / np.dot(v, v)
    d = np.linalg.norm((r + v * t) - p)
    return d, t


def line_line_shortest_dist_unbounded(r1: np.ndarray, v1: np.ndarray, r2: np.ndarray, v2: np.ndarray,
                                      eps: float = 1e-5) -> Tuple[float, Tuple[float, float]]:
    """ Find the shortest distance between two lines
    in the form y(t1) = r1 + v1 * t1 and y(t2) = r2 + v2 * t2

    If the two lines are parallel then distance will be correct
    but there are infinitely many (t1, t2) that gives that distance,
    of which the function will pick only one of

    For bounded t1 and t2, see shortest_distance_bounded

    :param r1: a point on line 1
    :param v1: direction of line 1
    :param r2: a point on line 2
    :param v2: direction of line 2
    :param eps: tolerance for checking parallel lines
    :return: the shortest distance and t1, t2 that gives it
    """

    # check that lines are not parallel
    # normalised dot product must not be 1 or -1
    if np.abs(np.dot(v1, v2)) < np.linalg.norm(v1) * np.linalg.norm(v2) - eps:
        R = r2 - r1
        A = np.array([[np.dot(v1, v1), -np.dot(v1, v2)],
                      [np.dot(v2, v1), -np.dot(v2, v2)]])
        b = np.array([np.dot(R, v1), np.dot(R, v2)])
        t1, t2 = np.matmul(np.linalg.inv(A), b)
        d = np.linalg.norm((r1 + v1 * t1) - (r2 + v2 * t2))
    else:
        # case where two lines are parallel
        # then fix one point and find shortest distance to that point
        t1 = 0
        d, t2 = line_point_shortest_dist(r2, v2, r1)

    return d, (t1, t2)


def line_line_shortest_dist_bounded(r1: np.ndarray, v1: np.ndarray, r2: np.ndarray, v2: np.ndarray,
                                    eps: float = 1e-5) -> Tuple[float, Tuple[float, float]]:
    """ Find the shortest distance between two lines
    in the form y(t1) = r1 + v1 * t1 and y(t2) = r2 + v2 * t2
    where 0 <= t1, t2 <= 1

    If the two lines are parallel then there may be
    infinitely many (t1, t2) that gives minimal distance
    but the function will only give one of the valid solutions

    For unbounded t1 and t2, see shortest_distance_unbounded

    :param r1: a point on line 1
    :param v1: direction of line 1
    :param r2: a point on line 2
    :param v2: direction of line 2
    :param eps: tolerance for checking parallel lines
    :return: the shortest distance and t1, t2 that gives it
    """

    # check against unbounded version first
    best_dist, (best_t1, best_t2) = line_line_shortest_dist_unbounded(r1, v1, r2, v2, eps=eps)

    if not (0 <= best_t1 <= 1 and 0 <= best_t2 <= 1):

        # enters here if unbounded optimal not in feasible region
        # solution therefore must be on the boundary
        # so check all edges and corners
        dr = r1 - r2
        v1_dot_v2 = np.dot(v1, v2)
        dr_dot_v1 = np.dot(dr, v1)
        dr_dot_v2 = np.dot(dr, v2)
        norm_v1_sq = np.dot(v1, v1)
        norm_v2_sq = np.dot(v2, v2)

        best_dist = np.inf
        best_t1, best_t2 = None, None
        dist = lambda t1, t2: np.linalg.norm(dr + (v1 * t1) - (v2 * t2))

        for t1_fixed in [0, 1, None]:

            for t2_fixed in [0, 1, None]:

                if t1_fixed is None and t2_fixed is None:
                    # this case not on edge, skipped
                    continue
                elif t1_fixed is None:
                    # case when t2_guess fixed
                    t1_guess = (v1_dot_v2 * t2_fixed - dr_dot_v1) / norm_v1_sq
                    t2_guess = t2_fixed
                elif t2_fixed is None:
                    # case when t1_guess fixed
                    t1_guess = t1_fixed
                    t2_guess = (v1_dot_v2 * t1_fixed + dr_dot_v2) / norm_v2_sq
                else:
                    t1_guess = t1_fixed
                    t2_guess = t2_fixed

                if 0 <= t1_guess <= 1 and 0 <= t2_guess <= 1:
                    # only consider if the terms are in the bounds
                    d = dist(t1_guess, t2_guess)
                    if d < best_dist:
                        best_dist = d
                        best_t1 = t1_guess
                        best_t2 = t2_guess

    return best_dist, (best_t1, best_t2)
