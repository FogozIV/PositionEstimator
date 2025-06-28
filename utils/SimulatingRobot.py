from itertools import accumulate

import numpy as np
from numpy import deg2rad
from scipy.integrate import solve_ivp
from math import pi, cos, ceil
from matplotlib import pyplot as plt
from scipy.optimize import least_squares,differential_evolution


def angle_diff(a, b):
    """Compute smallest signed difference a - b (result in [-pi, pi])"""
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi
class Robot:
    def __init__(self, real_track, left_wheel_mm_tick, right_wheel_mm_tick, odometry_tick_per_rev):
        self.real_track = real_track
        self.left_wheel_mm_tick = left_wheel_mm_tick
        self.right_wheel_mm_tick = right_wheel_mm_tick
        self.odometry_tick_per_rev = odometry_tick_per_rev
        self.left_odometry_pos = np.array([0.0, real_track / 2])
        self.right_odometry_pos = np.array([0.0, -real_track / 2])
        #self.left_odo_axis = np.array([1.0, 0.0])
        #self.right_odo_axis = np.array([1.0, 0.0])
        angle_left = 0
        angle_right = 0
        self.left_odo_axis = np.array([np.cos(deg2rad(angle_left)), np.sin(deg2rad(angle_left))])
        self.right_odo_axis = np.array([np.cos(deg2rad(angle_right)), np.sin(deg2rad(angle_right))])
        self.pose = np.array([0.0, 0.0, 0.0])
        self.odo_tick_left = 0
        self.odo_tick_right = 0
        self.__odo_tick_left = 0.0
        self.__odo_tick_right = 0.0
        self.estimator = None

        self._poses = []
        self._estimated_poses = []
        self._time = []
        self._encoders_ticks = []
        self.get_speed = lambda x: np.array([0,0])
        self.internal_time = 0.0
    @property
    def poses(self):
        return np.array(self._poses)

    @property
    def estimated_poses(self):
        return np.array(self._estimated_poses)

    @property
    def time(self):
        return np.array(self._time)

    @property
    def encoders_ticks(self):
        return np.array(self._encoders_ticks)

    def set_speed(self, speed):
        self.get_speed = lambda x: np.array(speed)

    def __kinematics(self, t, state):
        x, y, theta, odo_l, odo_r = state
        vl, vr = self.get_speed(t)
        v = (vl + vr) / 2
        omega = (vr - vl) / self.real_track

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega

        # Robot linear velocity
        robot_vel = np.array([dx, dy])  # in world frame
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        def odo_velocity(odo_pos, odo_axis):
            rot_vel_robot = omega * np.array([-odo_pos[1], odo_pos[0]])
            rot_vel_world = rot_matrix @ rot_vel_robot
            odo_axis_world = rot_matrix @ odo_axis
            wheel_velocity = robot_vel + rot_vel_world
            return np.dot(wheel_velocity, odo_axis_world)

        # Integrate displacement directly as ticks
        d_odo_left = odo_velocity(self.left_odometry_pos, self.left_odo_axis) / self.left_wheel_mm_tick
        d_odo_right = odo_velocity(self.right_odometry_pos, self.right_odo_axis) / self.right_wheel_mm_tick

        return [dx, dy, dtheta, d_odo_left, d_odo_right]

    def step(self, dt):
        state0 = np.concatenate((self.pose, [self.__odo_tick_left, self.__odo_tick_right]))
        # RK45 integration using scipy's solve_ivp
        sol = solve_ivp(
            fun=lambda t, y: self.__kinematics(t, y),
            t_span=(self.internal_time, self.internal_time+dt),
            y0=state0,
            method='RK45',
            t_eval=[self.internal_time + dt],
        )
        self.internal_time += dt
        x, y, theta, odo_l, odo_r = sol.y[:, -1]
        self.pose = np.array([x, y, theta])
        self.__odo_tick_left = odo_l
        self.__odo_tick_right = odo_r
        self.odo_tick_left = int(odo_l)
        self.odo_tick_right = int(odo_r)
        self._poses.append(self.pose.copy())
        if self.estimator is not None:
            self._estimated_poses.append(self.estimator.update(self.odo_tick_left, self.odo_tick_right)[0])
        self._time.append(dt)
        self._encoders_ticks.append([self.odo_tick_left, self.odo_tick_right])

        return self.pose.copy(), self.odo_tick_left, self.odo_tick_right
    def goto_forward(self, speed, distance, dt):
        self.set_speed([speed, speed])
        for i in range(ceil(distance/speed/dt)):
            self.step(dt)
    def turn_angle(self, speed, angle, dt):
        angle = np.deg2rad(angle)
        dangle = angle_diff(angle, self.pose[2])
        if dangle > 0:
            self.set_speed([-speed, speed])
        else:
            self.set_speed([speed, -speed])
        while abs(angle_diff(angle, self.pose[2])) > np.deg2rad(0.1):
            self.step(dt)
    def turn_turns(self, speed, turns, dt):
        mult = 1
        if turns > 0:
            self.set_speed([-speed, speed])
        else:
            self.set_speed([speed, -speed])
            mult = -1
        starting_angle = self.pose[2]
        while (starting_angle + turns * 2 * pi - self.pose[2])*mult > 0:
            self.step(dt)

class OdometryEstimator:
    def __init__(self, track_mm, left_mm_per_tick, right_mm_per_tick):
        self.track_mm = track_mm
        self.left_mm_per_tick = left_mm_per_tick
        self.right_mm_per_tick = right_mm_per_tick
        self.prev_left_ticks = 0.0
        self.prev_right_ticks = 0.0
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.calib_left_ticks = 0.0
        self.calib_right_ticks = 0.0

    def params_as_array(self):
        return np.array([self.left_mm_per_tick, self.right_mm_per_tick, self.track_mm])
    def update(self, left_ticks, right_ticks):
        """
        :param left_ticks: current left encoder tick count (int)
        :param right_ticks: current right encoder tick count (int)
        :return: (new_pose, delta_distance, delta_angle)
        """
        # Compute tick deltas
        delta_left_ticks = float(left_ticks) - self.prev_left_ticks
        delta_right_ticks = float(right_ticks) - self.prev_right_ticks

        # Update previous tick values
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks

        # Convert to mm
        left = delta_left_ticks * self.left_mm_per_tick
        right = delta_right_ticks * self.right_mm_per_tick

        distance = (left + right) / 2
        angle = (right - left) / self.track_mm

        # Motion update
        if angle == 0:
            dx = distance * np.cos(self.current_pos[2])
            dy = distance * np.sin(self.current_pos[2])
        else:
            r = distance * self.track_mm / (right - left)
            angle_rad = self.current_pos[2]
            dx = r * (-np.sin(angle_rad) + np.sin(angle_rad + angle))
            dy = r * (np.cos(angle_rad) - np.cos(angle_rad + angle))

        dtheta = angle
        delta_pose = np.array([dx, dy, dtheta])
        self.current_pos += delta_pose

        return self.current_pos.copy(), distance, angle
    def set_pos(self, pos):
        self.current_pos = np.array(pos)

    def calib_begin(self):
        self.calib_left_ticks = self.prev_left_ticks
        self.calib_right_ticks = self.prev_right_ticks

    def calib_forward(self, distance):
        left = (self.prev_left_ticks - self.calib_left_ticks) * self.left_mm_per_tick
        right = (self.prev_right_ticks - self.calib_right_ticks) * self.right_mm_per_tick
        corr_left = 1
        corr_right = 1
        if(left * distance) < 0:
            corr_left *= -1
            left *= -1
        if (right * distance) < 0:
            corr_right *= -1
            right *= -1
        multiplier = left/right
        corr_right *= multiplier
        right *= multiplier
        fake_distance = (left + right) / 2
        distance_multiplier = distance/fake_distance
        corr_right *= distance_multiplier
        corr_left *= distance_multiplier
        self.left_mm_per_tick *= corr_left
        self.right_mm_per_tick *= corr_right
        print("corr_left", corr_left, "corr_right", corr_right)
        print("left_mm_per_tick", self.left_mm_per_tick, "right_mm_per_tick", self.right_mm_per_tick)
    def calib_rotation(self, turns):
        left = (self.prev_left_ticks - self.calib_left_ticks) * self.left_mm_per_tick
        right = (self.prev_right_ticks - self.calib_right_ticks) * self.right_mm_per_tick
        estimated_angle = (right - left) / self.track_mm
        real_angle = turns * 2 * pi
        corr = estimated_angle / real_angle
        self.track_mm *= corr
        print("corr", corr, "track_mm", self.track_mm)

    def reset(self):
        self.prev_left_ticks = 0.0
        self.prev_right_ticks = 0.0
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.calib_left_ticks = 0.0
        self.calib_right_ticks = 0.0

def simulate_estimator_output(left_mm_tick, right_mm_tick, track_mm, encoder_ticks, measured_indexes):
    estimator = OdometryEstimator(track_mm, left_mm_tick, right_mm_tick)
    all_est_poses = []
    for i, (l,r) in enumerate(encoder_ticks):
        pose = estimator.update(l, r)[0]
        all_est_poses.append(pose)

    all_est_poses = np.array(all_est_poses)
    return all_est_poses[measured_indexes]

def residual(params, true_poses, encoder_ticks, measured_indexes):
    l_tick, r_tick, track_mm = params
    estimated_poses = simulate_estimator_output(l_tick, r_tick, track_mm, encoder_ticks, measured_indexes)
    return np.linalg.norm((true_poses - estimated_poses)[:, :2])



def get_speed(t):
    return [100 + 0.1 * 100 * cos(2*pi*2.3*t), -(100 + 0.1 * 100 * cos(2*pi*3.1*t+pi*17/180))]


def step(robot, step: float):
    robot.get_speed = get_speed
    robot.step(step)

def findParameters(ticks, poses, indexes, initial):
    return least_squares(
        residual,
        initial,
        args=(poses, ticks, indexes),
        method='trf',
        verbose=2,
        xtol=1e-10,

    )


if __name__ == "__main__":

    estimator = OdometryEstimator(track_mm=2.00012593e+02,
                                  left_mm_per_tick=4.10036716e-02,
                                  right_mm_per_tick=8.30027490e-02)
    robot = Robot(
        real_track=200,
        left_wheel_mm_tick=0.041,  # mm per tick (can be wheel circumference / ticks_per_rev)
        right_wheel_mm_tick=0.083,
        odometry_tick_per_rev=360*4
    )
    distance = 1000
    do_square_ccw = 10
    do_square_cw = 10
    do_rls = 1
    do_optim = 1
    robot.estimator = estimator
    do_calib_forward = 0
    do_calib_turn = 0
    measure_indexes = []
    forward_speed = 1000
    rotate_speed = 1000

    if do_calib_forward:
        estimator.calib_begin()
        robot.goto_forward(forward_speed, distance, dt=1/200)
        estimator.calib_forward(distance)


    if do_calib_turn:
        estimator.calib_begin()
        initial_angle = robot.pose[2]
        robot.turn_turns(rotate_speed, -100, 1/200)
        estimator.calib_rotation(-100)

    def rotate_angle(angles, forward_speed, rotate_speed, measure_indexes, dt=1/200):
        for angle in angles:
            robot.goto_forward(forward_speed, distance, dt=dt)
            measure_indexes.append([len(robot.poses)])
            robot.turn_turns(rotate_speed, angle, dt=dt)

    for i in range(do_square_ccw):
        angles = [90, 180, -90, 0]
        rotate_angle(angles, forward_speed, rotate_speed, measure_indexes)
        print(robot.estimated_poses[-1])
        print(robot.poses[-1])

    for i in range(do_square_cw):
        angles = [-90, 180, 90, 0]
        rotate_angle(angles, forward_speed, rotate_speed, measure_indexes)
        print(robot.estimated_poses[-1])
        print(robot.poses[-1])


    x0 = np.array([estimator.left_mm_per_tick, estimator.right_mm_per_tick, estimator.track_mm])
    true_poses = robot.poses[measure_indexes]
    if do_rls:
        result = least_squares(
            residual,
            x0,
            args=(true_poses, robot.encoders_ticks, measure_indexes),
            method='trf',
            verbose=2,
            xtol=1e-10,
        )
        print(result)
        print(result.x)


    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(robot.poses[:, 0], robot.poses[:, 1], label='Real Pose')
    if robot.estimator is not None:
        plt.plot(robot.estimated_poses[:, 0], robot.estimated_poses[:, 1], label='Estimated Pose')
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Real vs Estimated Pose")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(8, 8))
    plt.plot(robot.poses[:, 0], robot.poses[:, 1], label='Real Pose')
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Real Pose")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    if robot.estimator is not None:
        plt.figure(figsize=(8, 8))
        plt.plot(robot.estimated_poses[:, 0], robot.estimated_poses[:, 1], label='Estimated Pose')
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.title("Estimated Pose")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    accumulated_time = list(accumulate(robot.time))

    if robot.estimator is not None:
        plt.figure(figsize=(8, 8))
        plt.plot(accumulated_time, robot.estimated_poses[:, 2], label='Estimated Pose angle')
        plt.xlabel("time (s)")
        plt.ylabel("Angle (rad)")
        plt.title("Estimated Pose Angular")
        plt.legend()
        plt.grid(True)
        plt.show()
    plt.figure(figsize=(8, 8))
    if robot.estimator is not None:
        plt.plot(accumulated_time, (robot.estimated_poses[:, 2]/pi * 180 + 180)%360 - 180, label='Estimated Pose angle')
    plt.plot(accumulated_time, (robot.poses[:, 2]/pi * 180 + 180)%360 - 180, label='Real Pose angle')
    plt.xlabel("time (sec)")
    plt.ylabel("Angle (deg)")
    plt.title("Orientation")
    plt.legend()
    plt.grid(True)
    plt.show()
    if robot.estimator is not None:
        plt.figure(figsize=(8, 8))
        plt.plot(accumulated_time, np.linalg.norm(robot.estimated_poses[:, :2] - robot.poses[:, :2], axis=1), label='Distance between real and estimated pose')
        plt.xlabel("time (sec)")
        plt.ylabel("distance (mm)")
        plt.title("Evolution of distance vs time")
        plt.legend()
        plt.grid(True)
        plt.show()