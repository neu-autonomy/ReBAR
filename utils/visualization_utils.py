import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import pypoman
import numpy as np
from scipy.spatial import ConvexHull


def plot_polytope(A, B, figure_number, axis, color='#ff66ff', fill=False, alpha=1, label="BP set Over-Approximation"):

    plt.figure(figure_number)

    try:
        vertices = pypoman.polygon.compute_polygon_hull(A.astype(np.double), B.astype(np.double))
    except Exception:
        vertices = []

    if isinstance(vertices, list):
        points = -np.array(vertices)
    else:
        points = -vertices

    hull = ConvexHull(points)
    points = points[hull.vertices, :]

    patch = Polygon(
        points,
        alpha = alpha,
        color = color,
        linestyle = 'solid',
        fill = fill,
        linewidth = 5,
        label=label
    )

    axis.add_patch(patch)


def plot_util_sets(r, dist_max, figure_number, axis):

    agent_rectangle = Rectangle(
            [-r, -r],
            2*r, 2*r,
            fill=True,
            color='r',
            alpha=0.2,
            linewidth=5,
            label="Collision Set"
        )

    axis.add_patch(agent_rectangle)

    breachset_rectangle = Rectangle(
            [-dist_max, -dist_max],
            2*dist_max, 2*dist_max,
            fill=False,
            color='darkgreen',
            linewidth=5,
            label="Backreachable Set"
        )

    axis.add_patch(breachset_rectangle)