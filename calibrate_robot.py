from utils.bin_reader import read_file, CalibrationDataLegacy
import utils.SimulatingRobot as sr
from utils.SimulatingRobot import findParameters
from matplotlib import pyplot as plt
import numpy as np
import json
from utils.position_estimation import solve_table_geometry, Robot, plot_table

if __name__ == "__main__":
    table_dimensions = {
        'AB': 3000,
        'DA': 1999,
        'BC': 1999,
        'CD': 3004,
        'AC': 3606,
        'BD': 3608}

    #with open('table_dimensions.json', 'w') as f:
    #    json.dump(table_dimensions, f, indent=4)

    table_geometry = solve_table_geometry(table_dimensions, "x+")
    robot = Robot(local_points=np.array([
        [-50, 135],
        [-50, -135],
    ]))
    pos_measurement = [
        [[1382, 'A', 0], [2264, 'D', 0], [1377, 'B', 1], [2258, 'C', 1]],
        [[1512, 'D', 1], [1503, 'C', 1], [2181, 'B', 0], [2208, 'A', 0]],
        [[407, 'D', 1], [1784, 'A', 1], [2421, 'C', 0], [2990, 'B', 0]],
        [[566, 'A', 0], [466, 'A', 1], [1706, 'D', 0], [2544, 'B', 1]]
    ]
    positions = [robot.estimate_pose_from_trilateration_map(table_geometry, pos_measurement[i]) for i in range(len(pos_measurement))]
    plot_table(table_geometry)
    first_pos = positions[0]
    print(first_pos, "Angle", np.rad2deg(first_pos[2]))
    print(table_geometry)
    print(positions)
    positions = np.array(positions)

    indexes, result = read_file("./hard_calibration.bin", CalibrationDataLegacy)
    estimator = sr.OdometryEstimator(-197.1236433, 0.082176323, -0.082439216)
    estimator.set_pos(first_pos)
    indexes.insert(0, 0)
    print(indexes)
    poses = []
    for r in result:
        poses.append(estimator.update(r.left, r.right)[0])
    poses = np.array(poses)
    highlighted = poses[indexes]
    plt.figure(figsize=(8, 8))
    plt.plot(poses[:, 0], poses[:, 1], label='Real Pose')
    plt.scatter(highlighted[:, 0], highlighted[:, 1], s=100, edgecolors='red', facecolors='none',
                label='Highlighted Points')

    plt.scatter(positions[:, 0], positions[:, 1], s=100, edgecolors='blue', facecolors='none',
                label='Real Points')
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Real Pose")
    plt.legend()
    plt.show()
    result = np.array([[r.left, r.right] for r in result])
    parameters = findParameters(result, positions, indexes, estimator.params_as_array())
    print(parameters)