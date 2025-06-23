import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def compute_residual(position, positions):
    x, y = position
    return np.sqrt((x-positions[:, 0])**2 + (y-positions[:, 1])**2) - positions[:, 2]

def find_point(positions, initial_guess=None, distances=None):
    positions = np.array(positions)
    if initial_guess is None:
        initial_guess = np.mean(positions, axis=0)
    if distances is not None:
        distances = np.array(distances).reshape(-1, 1)
        positions = np.hstack((positions, distances))
    return least_squares(compute_residual, initial_guess, args=(positions,))

def transform(pose, local_points):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return local_points @ R.T + np.array([x, y])

def residuals(pose, local_points, world_points):
    transformed = transform(pose, local_points)
    return (transformed - world_points).ravel()

def estimate_robot_pose(local_points, world_points):
    # local_points: (N, 2) robot-frame known marker positions
    # world_points: (N, 2) measured world-frame positions
    initial_guess = [0, 0, 0]  # x, y, theta
    result = least_squares(residuals, initial_guess,
                           args=(local_points, world_points))
    return result.x
def residuals_for_trilateration(pose, robot_local_points, world_points, distances, robot_indices):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    res = []
    for wp, dist, idx in zip(world_points, distances, robot_indices):
        p_local = robot_local_points[idx]
        p_world = R @ p_local + np.array([x, y])
        res.append(np.linalg.norm(p_world - wp) - dist)
    return res
class Robot:
    def __init__(self, local_points):
        self.local_points = local_points
    def estimate_robot_pose(self, world_points):
        return estimate_robot_pose(self.local_points, world_points)
    def transform(self, pose):
        return transform(pose, self.local_points)
    def estimate_robot_pos_from_distances(self, world_points, distances):
        robot_world_points = np.array([])
        for poses, dists in zip(world_points, distances):
            result = np.array(find_point(poses, distances=dists).x).reshape(-1, 1)
            robot_world_points = np.hstack((robot_world_points, result))
        return self.estimate_robot_pose(robot_world_points)
    def estimate_pose_from_trilateration(self, world_points, distances, robot_indices, initial_guess=(0, 0, 0)):
        result = least_squares(
            residuals_for_trilateration,
            initial_guess,
            args=(self.local_points, world_points, distances, robot_indices)
        )
        return result.x  # (x, y, theta)
def plot_robot_and_measurements(robot, pose, measured_points=None, distances=None):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Robot Pose and Measurements")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Draw robot pose (red dot + orientation arrow)
    x, y, theta = pose
    ax.plot(x, y, 'ro', label='Robot Center')
    ax.arrow(x, y, 0.2 * np.cos(theta), 0.2 * np.sin(theta),
             head_width=0.05, head_length=0.1, fc='red', ec='red')

    # Transform local points to world
    transformed_points = robot.transform(pose)

    # Draw robot fixed points (blue dots)
    ax.plot(transformed_points[:, 0], transformed_points[:, 1], 'bo', label='Robot Markers')

    # If measured world points are provided, draw them
    if measured_points is not None:
        measured_points = np.array(measured_points)
        ax.plot(measured_points[:, 0], measured_points[:, 1], 'ko', label='Measured Points')

        # Optional: draw lines from predicted (transformed) to measured
        for p_pred, p_meas in zip(transformed_points, measured_points):
            ax.plot([p_pred[0], p_meas[0]], [p_pred[1], p_meas[1]], 'k--', linewidth=0.5)

    # If distances are provided, draw circles
    if distances is not None:
        for (x_m, y_m), d in zip(measured_points, distances):
            circle = plt.Circle((x_m, y_m), d, color='gray', linestyle='--', fill=False)
            ax.add_patch(circle)

    ax.legend()
    plt.axis('equal')
    plt.show()
def plot_trilateration(world_points, distances, estimated_position):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Trilateration from Circles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Draw each circle and its center
    for (x, y), r in zip(world_points, distances):
        ax.plot(x, y, 'ko')  # beacon position (black dot)
        circle = plt.Circle((x, y), r, color='blue', linestyle='--', fill=False)
        ax.add_patch(circle)
        ax.text(x + 0.1, y + 0.1, f"{x:.2f}, {y:.2f}", fontsize=8)

    # Plot estimated robot position
    est_x, est_y = estimated_position
    ax.plot(est_x, est_y, 'ro', label='Estimated Robot Position')
    ax.text(est_x + 0.1, est_y + 0.1, f"{est_x:.2f}, {est_y:.2f}", color='red')

    ax.legend()
    plt.axis('equal')
    plt.show()

def plot_trilateration_with_robot_pose(robot_pose, robot_local_points,
                                       world_points, distances, robot_indices):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Trilateration with Robot Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Unpack pose
    x, y, theta = robot_pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Transform robot local points to world frame
    robot_world_points = robot_local_points @ R.T + np.array([x, y])

    # Plot robot pose
    ax.plot(x, y, 'ro', label='Robot Center')
    ax.arrow(x, y, 0.2 * c, 0.2 * s, head_width=0.05, head_length=0.1, fc='red', ec='red')

    # Plot robot-fixed points
    ax.plot(robot_world_points[:, 0], robot_world_points[:, 1], 'bo', label='Robot Markers')
    for i, (px, py) in enumerate(robot_world_points):
        ax.text(px + 0.05, py + 0.05, f'P{i}', color='blue', fontsize=8)

    # Plot each trilateration circle
    for (cx, cy), r, idx in zip(world_points, distances, robot_indices):
        ax.plot(cx, cy, 'ko')  # Beacon center
        circle = plt.Circle((cx, cy), r, color='gray', linestyle='--', fill=False)
        ax.add_patch(circle)

        # Line to corresponding robot point
        rp = robot_world_points[idx]
        ax.plot([cx, rp[0]], [cy, rp[1]], 'k--', linewidth=0.5)
        ax.text(cx + 0.05, cy + 0.05, f'{cx:.1f},{cy:.1f}', fontsize=8)

    ax.legend()
    plt.axis('equal')
    plt.show()


def solve_table_geometry(measured, config='x+'):
    """
    A ---- B
    |      |
    |      |
    D ---- C

    measured = {
        'AB': ...,
        'DA': ...,
        'BC': ...,
        'CD': ...,
        'AC': ...,
        'BD': ...
    }
    config: 'x+' → B is on X axis; 'y+' → D is on Y axis
    """

    A = np.array([0.0, 0.0])

    if config == 'x+':
        B_fixed = np.array([measured['AB'], 0.0])  # fixed
        D_fixed = None
        guess = [measured['AB'], measured['DA'], 0.0, measured['DA']]  # C, D

        def residuals(X):
            C = X[:2]
            D = X[2:]
            r = [
                np.linalg.norm(B_fixed - C) - measured['BC'],
                np.linalg.norm(C - D) - measured['CD'],
                np.linalg.norm(D - A) - measured['DA'],
                np.linalg.norm(C - A) - measured['AC'],
                np.linalg.norm(D - B_fixed) - measured['BD']
            ]
            return r

        result = least_squares(residuals, guess)
        C = result.x[:2]
        D = result.x[2:]
        B = B_fixed

    elif config == 'y+':
        D_fixed = np.array([0.0, measured['DA']])  # fixed
        B_fixed = None
        guess = [measured['AB'], measured['DA'], measured['AB'], 0.0]  # C, B

        def residuals(X):
            C = X[:2]
            B = X[2:]
            r = [
                np.linalg.norm(B - C) - measured['BC'],
                np.linalg.norm(C - D_fixed) - measured['CD'],
                np.linalg.norm(D_fixed - A) - measured['DA'],
                np.linalg.norm(C - A) - measured['AC'],
                np.linalg.norm(D_fixed - B) - measured['BD']
            ]
            return r

        result = least_squares(residuals, guess)
        C = result.x[:2]
        B = result.x[2:]
        D = D_fixed

    else:
        raise ValueError("Invalid config: use 'x+' or 'y+'")

    return {
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'success': result.success
    }
def plot_table(corners, title="Corrected Table Geometry"):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Draw table edges
    points = [corners['A'], corners['B'], corners['C'], corners['D'], corners['A']]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        ax.plot([x0, x1], [y0, y1], 'k-')
        ax.plot(x0, y0, 'ro')
        ax.text(x0 + 0.05, y0 + 0.05, f"{chr(ord('A') + i)}", fontsize=12, color='red')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()