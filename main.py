from utils.position_estimation import find_point, Robot, plot_robot_and_measurements, plot_trilateration_with_robot_pose, solve_table_geometry, plot_table
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('QtAgg')

robot = Robot(local_points=np.array([
    [-50, 135],
    [-50, -135],
]))
data = {'AB': 3000,
 'DA': 1999,
 'BC': 1999,
 'CD': 3004,
 'AC': 3606,
 'BD': 3608}
result = solve_table_geometry(data, "x+")
print(result)
plot_table(result)
world_points =[[2.5,1.8],
 [1.8, 2.2],
 [1.2, 1.3],
 [2.6, 0.9],
 [1.7, 1.0]]
distances =[0.5099, 0.7289, 0.7347, 0.6709, 0.7932]

robot_indices =[0, 1, 2, 0, 2]

estimated_pose = robot.estimate_pose_from_trilateration(world_points, distances, robot_indices)

plot_trilateration_with_robot_pose(estimated_pose, robot.local_points, world_points, distances, robot_indices)

