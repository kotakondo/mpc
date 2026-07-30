# mpc

Model-predictive path-tracking controller for differential-drive ground robots,
as a ROS 2 (Humble) Python package.

The node subscribes to a waypoint path and the robot's pose, solves a
short-horizon MPC problem with [do-mpc](https://www.do-mpc.com/), and publishes
`geometry_msgs/Twist` velocity commands. It is the ground-robot controller used
by [MIGHTY](https://github.com/mit-acl/mighty).

## Installation

Normally you don't install this directly — MIGHTY pulls it in via `mighty.repos`:

```bash
vcs import src < src/mighty/mighty.repos
colcon build --packages-select mpc
```

Standalone:

```bash
cd ~/your_ws/src
git clone https://github.com/kotakondo/mpc.git
cd .. && colcon build --packages-select mpc
```

Requires `do-mpc` (`pip install do-mpc`) plus `rclpy`, `tf2_ros`, `nav_msgs`,
`geometry_msgs`, `sensor_msgs`, `message_filters`, and `dynus_interfaces`.

## Usage

```bash
ros2 launch mpc mpc.launch.py
```

MIGHTY launches it automatically for ground robots — see
`onboard_mighty.launch.py`, which selects `config/mpc_sim.yaml` in simulation and
`config/mpc.yaml` on hardware.

## Configuration

| Parameter | Meaning |
|-----------|---------|
| `path_topic`, `cmd_vel_topic` | input path and output velocity topics |
| `tracking_frame`, `base_frame` | TF frames used for pose lookup |
| `use_tf_pose` | take pose from TF instead of a pose topic |
| `control_rate_hz`, `dt` | controller rate and MPC step |
| `v_min`, `v_max`, `w_max` | linear and angular velocity limits |
| `max_linear_accel` | linear acceleration limit |
| `goal_tolerance` | distance at which the goal counts as reached |

## Acknowledgments

The MPC controller was originally developed by
**[Lucas Jia (@lucas-yyy000)](https://github.com/lucas-yyy000)**, who wrote the
initial implementation. Later contributions came from
[@kotakondo](https://github.com/kotakondo), Sera Ham, and Elon Raya.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
