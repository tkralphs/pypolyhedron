#!/usr/bin/env python
"""
Convex hull algorithms based on polyhedron and cddlib

hull - create a hull given 2d/3d points
inside - return mask of points that are inside a given hull (2d/3d)
"""
__all__ = ['hull', 'inside']
__author__ = 'Rolv, Pearu'

import numpy as np
from .polyhedron import Vrep, Hrep


def _as_float_array(x):
    """Ensure consistent dtype for NumPy 1/2 compatibility."""
    return np.asarray(x, dtype=float)


def _mkhull(points):
    points = _as_float_array(points)
    p = Vrep(points)
    return Hrep(p.A, p.b)


def hull(points):
    return _mkhull(points).generators


def inside(p, points):
    if not isinstance(p, Hrep):
        p = _mkhull(p)

    points = _as_float_array(points)

    if points.shape[-1] == 1:
        raise ValueError("Cannot do 1d points")

    # NumPy 2: replace alltrue → all
    inside_ = lambda point: np.all(np.dot(p.A, point) <= p.b)
    mask = np.apply_along_axis(inside_, 1, points)
    return mask


def _test_2d(n=100000):
    print(
        'percent inside (should be around 0.1):',
        size(np.nonzero(inside(
                ((0, 0), (0, 1), (0.1, 1), (0.1, 0)),
                np.random.random((n, 2)))
            ))/float(n)
    )


# Domains
_d_1d = np.asarray([0, 0.1])
_d_2d = np.asarray([(0, 0), (0, 1), (0.1, 1), (0.1, 0)])
_d_3d = 0.1 ** (1/3) * np.asarray([
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
])


def _test():
    points = np.random.random((20, 3))
    p = _mkhull(points)
    print('Hull vertices:\n', p.generators)

    points2 = 1.1 * np.random.random((10, 3))
    for i in range(len(points2)):
        point = points2[i]
        if np.all(np.dot(p.A, point) <= p.b):
            print('point', point, 'is IN')
        else:
            print('point', point, 'is OUT')

    points3 = np.random.random((3000, 3))
    mask = inside(p, points3)
    p3 = points3[np.nonzero(mask)]

    n = 100000
    for d in (_d_2d, _d_3d):
        h = hull(d)
        dim = h.shape[-1]
        points = np.random.random((n, dim))
        print(
            0.1,
            dim,
            np.size(np.nonzero(inside(h, points)))/ float(n)
        )


def _test_plot():
    try:
        import pylab
    except ImportError:
        pylab = False

    if pylab:
        pylab.ion()
        import matplotlib.axes3d as p3

        u = np.r_[0:2 * np.pi:100j]
        v = np.r_[0:np.pi:100j]
        x = 10 * np.outer(np.cos(u), np.sin(v))
        y = 10 * np.outer(np.sin(u), np.sin(v))
        z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

        points = np.transpose(np.vstack((x.flat, y.flat, z.flat)))

        hull_ = np.asarray([
            (0, 0, 0), (0, 0, 1), (-10, 0, 0), (-10, 0, 1),
            (-10, -10, 0), (-10, -10, 1), (0, -10, 0), (0, -10, 1)
        ])

        ax = p3.Axes3D(pylab.figure())
        ax.plot_surface(x, y, z)

        x_, y_, z_ = list(map(np.squeeze, np.hsplit(hull_, 3)))
        ax.scatter3d(x_, y_, z_, color='g')

        s = np.nonzero(z_ == 0)
        x, y, z = x_[s].tolist(), y_[s].tolist(), z_[s].tolist()
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])

        ax.plot3D(x, y, z, 'g-')
        ax.plot3D(x, y, np.ones(len(z)), 'g-')

        x, y, z = list(map(
            np.squeeze,
            np.hsplit(points[np.nonzero(inside(hull_, points))], 3)
        ))
        ax.scatter3d(x, y, z, color='r')

        pylab.draw()
        pylab.savefig('hull3d.png')


if __name__ == '__main__':
    print(_mkhull(_d_1d))