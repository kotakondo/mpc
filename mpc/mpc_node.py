#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from dynus_interfaces.msg import SpeedyPath

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose_stamped

import casadi as ca
import do_mpc


YELLOW = "\033[93m"
RESET  = "\033[0m"


def wrap_angle(a):
    return ca.atan2(ca.sin(a), ca.cos(a))


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ─────────────────────────────────────────────────────────────────────────────
# Arclength utilities (inlined from reference code)
# ─────────────────────────────────────────────────────────────────────────────

def polyline_arclength(path: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a polyline (N x 2)."""
    diffs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.zeros(len(path))
    s[1:] = np.cumsum(seg_lens)
    return s


def interp_along_path(path: np.ndarray, s_grid: np.ndarray, s: float):
    """Interpolate position and yaw at arc-length s along path."""
    s = float(np.clip(s, 0.0, s_grid[-1]))
    idx = np.searchsorted(s_grid, s, side='right') - 1
    idx = int(np.clip(idx, 0, len(path) - 2))
    ds = s_grid[idx + 1] - s_grid[idx]
    if ds < 1e-9:
        t = 0.0
    else:
        t = (s - s_grid[idx]) / ds
    pos = (1.0 - t) * path[idx] + t * path[idx + 1]
    diff = path[idx + 1] - path[idx]
    yaw = math.atan2(diff[1], diff[0])
    return pos, yaw


def speed_interp(v_des: np.ndarray, s_grid: np.ndarray, s: float) -> float:
    """Interpolate desired speed at arc-length s."""
    s = float(np.clip(s, 0.0, s_grid[-1]))
    idx = np.searchsorted(s_grid, s, side='right') - 1
    idx = int(np.clip(idx, 0, len(v_des) - 2))
    ds = s_grid[idx + 1] - s_grid[idx]
    if ds < 1e-9:
        return float(v_des[idx])
    t = (s - s_grid[idx]) / ds
    return float((1.0 - t) * v_des[idx] + t * v_des[idx + 1])


def project_pose_to_arclength(path: np.ndarray, s_grid: np.ndarray, xy: np.ndarray) -> float:
    """Find the arc-length on path closest to point xy."""
    best_s = 0.0
    best_d2 = float('inf')
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-12:
            t = 0.0
        else:
            t = float(np.clip(np.dot(xy - a, ab) / ab_len2, 0.0, 1.0))
        proj = a + t * ab
        d2 = float(np.sum((xy - proj) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_s = s_grid[i] + t * (s_grid[i + 1] - s_grid[i])
    return best_s


# ─────────────────────────────────────────────────────────────────────────────
# Reference trajectory generator (from reference code)
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceTrajectoryGenerator:
    """Horizon provider for MPC (position, yaw, v) based on path arclength."""
    def __init__(self):
        self.path = None
        self.v_des = None
        self.s_grid = None
        self.s_now = 0.0

    def update_path(self, path, v_des, keep_progress=False, x_cur=None):
        path = np.asarray(path, dtype=float)
        v_des = np.asarray(v_des, dtype=float)
        if len(path) < 2:
            return
        if path.shape[0] != v_des.shape[0]:
            v_des = np.ones(len(path), dtype=float) * v_des[0] if len(v_des) > 0 else np.ones(len(path))
        self.path = path
        self.v_des = v_des
        self.s_grid = polyline_arclength(self.path)

        if keep_progress:
            self.s_now = float(np.clip(self.s_now, 0.0, self.s_grid[-1]))
        elif x_cur is not None:
            self.s_now = project_pose_to_arclength(self.path, self.s_grid, x_cur[:2])
        else:
            self.s_now = 0.0

    def step_forward(self, v_nom, dt):
        if self.s_grid is not None:
            self.s_now = min(float(self.s_grid[-1]), self.s_now + max(0.0, float(v_nom)) * dt)

    def has_path(self):
        return self.path is not None and len(self.path) >= 2

    def window(self, N, dt):
        """Return arrays over horizon k=0..N: pos_ref[k](2,), yaw_ref[k], v_ref[k]."""
        pos_list, yaw_list, v_list = [], [], []
        s_k = float(self.s_now)
        for _ in range(N + 1):
            vloc = speed_interp(self.v_des, self.s_grid, s_k)
            pos_k, yaw_k = interp_along_path(self.path, self.s_grid, s_k)
            pos_list.append(pos_k)
            yaw_list.append(yaw_k)
            v_list.append(vloc)
            s_k = min(float(self.s_grid[-1]), s_k + vloc * dt)
        return np.array(pos_list), np.array(yaw_list), np.array(v_list)


# ─────────────────────────────────────────────────────────────────────────────
# MPC Node
# ─────────────────────────────────────────────────────────────────────────────

class MPCNode(Node):
    def __init__(self) -> None:
        super().__init__('mpc')

        # --- Parameters ---
        self.declare_parameter('use_tf_pose', False)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tracking_frame', 'odom')

        self.declare_parameter('pose_topic', 'pose')
        self.declare_parameter('path_topic', 'mpc_waypoints')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')

        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('control_rate_hz', 30.0)

        self.declare_parameter('dt', 0.05)
        self.declare_parameter('N_horizon', 20)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('v_max', 1.0)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('max_linear_accel', 0.5)  # [m/s^2] ramp-up/down acceleration limit
        self.declare_parameter('Q_pos', 5.0)
        self.declare_parameter('Q_yaw', 1.0)
        self.declare_parameter('R_v', 0.1)
        self.declare_parameter('R_w', 0.05)
        self.declare_parameter('dR_v', 0.01)
        self.declare_parameter('dR_w', 0.01)

        # --- Load parameters ---
        self.use_tf_pose = self.get_parameter('use_tf_pose').get_parameter_value().bool_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value.lstrip('/')
        self.tracking_frame = self.get_parameter('tracking_frame').get_parameter_value().string_value.lstrip('/')

        # Auto-prefix TF frames with namespace (ROS 2 namespacing doesn't affect TF frame IDs)
        ns = self.get_namespace().strip('/')
        if ns and '/' not in self.base_frame:
            self.base_frame = f'{ns}/{self.base_frame}'
        if ns and '/' not in self.tracking_frame:
            self.tracking_frame = f'{ns}/{self.tracking_frame}'

        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.goal_tol = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.rate_hz = self.get_parameter('control_rate_hz').get_parameter_value().double_value

        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.N = int(self.get_parameter('N_horizon').get_parameter_value().integer_value)
        self.v_min = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.w_max = self.get_parameter('w_max').get_parameter_value().double_value
        self.max_linear_accel = self.get_parameter('max_linear_accel').get_parameter_value().double_value
        self.Q_pos = self.get_parameter('Q_pos').get_parameter_value().double_value
        self.Q_yaw = self.get_parameter('Q_yaw').get_parameter_value().double_value
        self.R_v = self.get_parameter('R_v').get_parameter_value().double_value
        self.R_w = self.get_parameter('R_w').get_parameter_value().double_value
        self.dR_v = self.get_parameter('dR_v').get_parameter_value().double_value
        self.dR_w = self.get_parameter('dR_w').get_parameter_value().double_value

        self.get_logger().info(f'MPC params: dt={self.dt}, N={self.N}, v=[{self.v_min},{self.v_max}], w_max={self.w_max}')

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Subscribers / Publishers ---
        if not self.use_tf_pose:
            self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 10)
        self.path_sub = self.create_subscription(SpeedyPath, self.path_topic, self._path_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # --- State ---
        self.latest_pose: Optional[PoseStamped] = None
        self.goal_reached = False
        self._mpc_ready = False
        self.ref_gen = ReferenceTrajectoryGenerator()
        self._dist_traveled = 0.0  # accumulated distance since goal received
        self._last_xy = None       # for tracking distance traveled
        self._prev_v_cmd = 0.0     # for rate-limiting output velocity
        self._prev_w_cmd = 0.0     # for rate-limiting output angular velocity

        # Clear debug log
        with open('/tmp/mpc_debug.log', 'w') as f:
            f.write(f"MPC started: tracking_frame={self.tracking_frame} pose_topic={self.pose_topic} "
                    f"path_topic={self.path_topic} use_tf_pose={self.use_tf_pose}\n")

        # --- Build MPC ---
        self._build_mpc()

        # --- Control timer ---
        period = 1.0 / max(1e-3, self.rate_hz)
        self.timer = self.create_timer(period, self._control_step)
        self.get_logger().info('MPC controller started.')

    # ──────────────── Callbacks ────────────────

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = yaw_from_quat(msg.pose.orientation)
        frame = msg.header.frame_id.lstrip('/') if msg.header.frame_id else ''
        self._log_debug(f"ODOM: frame={frame} pos=({x:.4f},{y:.4f}) yaw={yaw:.4f}")

    def _path_cb(self, msg: SpeedyPath) -> None:
        """Receive path_msgs/SpeedyPath from mighty's publishMpcPath."""
        if len(msg.poses) < 2:
            return

        path_frame = msg.header.frame_id.lstrip('/') if msg.header.frame_id else ''
        path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=float)
        # Use velocity references from mighty if available, otherwise fall back to v_max
        if len(msg.speeds) == len(path):
            v_des = np.array(msg.speeds, dtype=float)
        else:
            v_des = np.ones(len(path), dtype=float) * self.v_max

        # Get current state for arclength projection
        cur = self._current_pose_tracking()
        if cur is not None:
            x_cur = np.array([cur.pose.position.x, cur.pose.position.y, yaw_from_quat(cur.pose.orientation)])
        else:
            x_cur = np.array([path[0, 0], path[0, 1], 0.0])

        # Always re-project from current robot position
        first_path = not self.ref_gen.has_path()
        self.ref_gen.update_path(path, v_des, keep_progress=False, x_cur=x_cur)
        self.goal_reached = False

        # Only reinitialize warm start on the very first path
        if first_path:
            self._mpc_ready = False
            self._dist_traveled = 0.0
            self._last_xy = None
            self.get_logger().info(f'MPC: first path with {len(path)} waypoints.')

        # Debug log
        self._log_debug(f"PATH_CB: frame={path_frame} tracking_frame={self.tracking_frame} "
                        f"n_pts={len(path)} path[0]=({path[0,0]:.3f},{path[0,1]:.3f}) "
                        f"path[-1]=({path[-1,0]:.3f},{path[-1,1]:.3f}) "
                        f"x_cur=({x_cur[0]:.3f},{x_cur[1]:.3f},{x_cur[2]:.3f})")

    # ──────────────── Pose helpers ────────────────

    def _current_pose_tracking(self) -> Optional[PoseStamped]:
        if self.use_tf_pose:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=self.tracking_frame,
                    source_frame=self.base_frame,
                    time=rclpy.time.Time())
                base_in_base = PoseStamped()
                base_in_base.header.stamp = self.get_clock().now().to_msg()
                base_in_base.header.frame_id = self.base_frame
                base_in_base.pose.orientation.w = 1.0
                pose = do_transform_pose_stamped(base_in_base, tf)
                pose.header.frame_id = self.tracking_frame
                return pose
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
                return None
        else:
            if self.latest_pose is None:
                return None
            ps = self.latest_pose
            source_frame = ps.header.frame_id.lstrip('/') if ps.header.frame_id else ''
            if source_frame and source_frame != self.tracking_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        target_frame=self.tracking_frame,
                        source_frame=source_frame,
                        time=rclpy.time.Time())
                    ps = do_transform_pose_stamped(ps, tf)
                    ps.header.frame_id = self.tracking_frame
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    self.get_logger().warn(f'Pose transform failed: {e}')
                    return None
            return ps

    # ──────────────── MPC build ────────────────

    def _build_mpc(self) -> None:
        model = do_mpc.model.Model('discrete')

        pos = model.set_variable('_x', 'pos', (2, 1))
        yaw = model.set_variable('_x', 'yaw')
        v = model.set_variable('_u', 'v')
        w = model.set_variable('_u', 'w')

        p_ref = model.set_variable('_tvp', 'p_ref', (2, 1))
        yaw_ref = model.set_variable('_tvp', 'yaw_ref')
        v_ref = model.set_variable('_tvp', 'v_ref')

        pos_next = pos + self.dt * ca.vertcat(ca.cos(yaw) * v, ca.sin(yaw) * v)
        yaw_next = yaw + self.dt * w
        model.set_rhs('pos', pos_next)
        model.set_rhs('yaw', yaw_next)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.N, t_step=self.dt,
            n_robust=0, open_loop=0,
            state_discretization='collocation',
            collocation_type='radau', collocation_deg=3, collocation_ni=1,
            store_full_solution=False,
            nlpsol_opts={
                'ipopt.linear_solver': 'mumps',
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
                'print_time': 0,
            },
        )

        pos_err = pos - p_ref
        yaw_err = wrap_angle(yaw - yaw_ref)
        lterm = (
            self.Q_pos * ca.sumsqr(pos_err)
            + self.Q_yaw * ca.sumsqr(yaw_err)
            + self.R_v * (v - v_ref) ** 2
            + self.R_w * w ** 2
        )
        mterm = 2.0 * (self.Q_pos * ca.sumsqr(pos_err) + self.Q_yaw * ca.sumsqr(yaw_err))
        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(v=self.dR_v, w=self.dR_w)

        mpc.bounds['lower', '_u', 'v'] = self.v_min
        mpc.bounds['upper', '_u', 'v'] = self.v_max
        mpc.bounds['lower', '_u', 'w'] = -self.w_max
        mpc.bounds['upper', '_u', 'w'] = self.w_max

        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(self.N + 1):
                tvp_template['_tvp', k, 'p_ref'] = np.zeros((2, 1))
                tvp_template['_tvp', k, 'yaw_ref'] = 0.0
                tvp_template['_tvp', k, 'v_ref'] = 0.0

            if not self.ref_gen.has_path():
                return tvp_template

            pos_ref, yaw_ref_arr, v_ref_arr = self.ref_gen.window(self.N, self.dt)
            for k in range(self.N + 1):
                tvp_template['_tvp', k, 'p_ref'] = pos_ref[k].reshape(2, 1)
                tvp_template['_tvp', k, 'yaw_ref'] = float(yaw_ref_arr[k])
                tvp_template['_tvp', k, 'v_ref'] = float(v_ref_arr[k])
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()

        self.model = model
        self.mpc = mpc
        self.get_logger().info(f'{YELLOW}MPC initialized (path tracking).{RESET}')

    # ──────────────── Debug logging ────────────────

    _debug_log_count = 0

    def _log_debug(self, msg: str) -> None:
        with open('/tmp/mpc_debug.log', 'a') as f:
            f.write(f"[{MPCNode._debug_log_count}] {msg}\n")
        MPCNode._debug_log_count += 1

    # ──────────────── Control loop ────────────────

    def _control_step(self) -> None:
        cur = self._current_pose_tracking()
        if cur is None or not self.ref_gen.has_path() or self.goal_reached:
            self._publish_cmd(0.0, 0.0)
            return

        # Extract current state
        x = float(cur.pose.position.x)
        y = float(cur.pose.position.y)
        yaw = yaw_from_quat(cur.pose.orientation)

        # Track accumulated distance traveled
        cur_xy = np.array([x, y])
        if self._last_xy is not None:
            self._dist_traveled += float(np.linalg.norm(cur_xy - self._last_xy))
        self._last_xy = cur_xy.copy()

        # Apply trapezoidal speed profile per-waypoint
        a = self.max_linear_accel
        dist_to_end = np.linalg.norm(cur_xy - self.ref_gen.path[-1])
        v_ramp_up = math.sqrt(2.0 * a * self._dist_traveled) if self._dist_traveled > 0 else 0.0
        if self.ref_gen.v_des is not None and self.ref_gen.s_grid is not None:
            s_total = self.ref_gen.s_grid[-1]
            for i in range(len(self.ref_gen.v_des)):
                s_to_end = s_total - self.ref_gen.s_grid[i]
                v_down = math.sqrt(2.0 * a * s_to_end) if s_to_end > 0 else 0.0
                self.ref_gen.v_des[i] = min(self.v_max, v_ramp_up, v_down)
        if dist_to_end <= self.goal_tol:
            if not self.goal_reached:
                self.get_logger().info(f'{YELLOW}MPC: goal reached.{RESET}')
                self.goal_reached = True
            self._publish_cmd(0.0, 0.0)
            return

        x_vec = np.array([[x], [y], [yaw]], dtype=float)

        # Log current state vs reference every 30 steps (~1s)
        if MPCNode._debug_log_count % 30 == 0:
            pose_frame = cur.header.frame_id if cur else 'N/A'
            if self.ref_gen.has_path():
                pos_ref, yaw_ref, v_ref = self.ref_gen.window(min(3, self.N), self.dt)
                self._log_debug(
                    f"CTRL: pose_frame={pose_frame} "
                    f"robot=({x:.3f},{y:.3f},yaw={yaw:.3f}) "
                    f"ref[0]=({pos_ref[0][0]:.3f},{pos_ref[0][1]:.3f},yaw={yaw_ref[0]:.3f}) "
                    f"ref[1]=({pos_ref[1][0]:.3f},{pos_ref[1][1]:.3f}) "
                    f"s_now={self.ref_gen.s_now:.3f}/{self.ref_gen.s_grid[-1]:.3f} "
                    f"dist_to_end={dist_to_end:.3f} "
                    f"err=({pos_ref[0][0]-x:.3f},{pos_ref[0][1]-y:.3f})")

        if not self._mpc_ready:
            self.mpc.x0 = x_vec
            self.mpc.set_initial_guess()
            self._mpc_ready = True
            self._publish_cmd(0.0, 0.0)
            return

        try:
            t0 = time.time()
            u = self.mpc.make_step(x_vec)
            t1 = time.time()
            # self.get_logger().info(f'{YELLOW}MPC solve: {(t1 - t0) * 1000:.1f} ms{RESET}')
        except Exception as e:
            self.get_logger().warn(f'MPC solver failed: {e}')
            self._publish_cmd(0.0, 0.0)
            return

        v_cmd = float(u[0])
        w_cmd = float(u[1])

        # Log every control step for velocity analysis
        self._log_debug(f"CMD: v={v_cmd:.4f} w={w_cmd:.4f} v_ramp_up={v_ramp_up:.3f} "
                        f"dist_traveled={self._dist_traveled:.3f} dist_to_end={dist_to_end:.3f} "
                        f"v_des[0]={self.ref_gen.v_des[0]:.3f} "
                        f"robot=({x:.3f},{y:.3f},yaw={yaw:.3f})")

        self._publish_cmd(v_cmd, w_cmd)
        # Re-project arclength from current robot position (keeps reference anchored to robot)
        self.ref_gen.s_now = project_pose_to_arclength(
            self.ref_gen.path, self.ref_gen.s_grid, cur_xy)

    def _publish_cmd(self, v: float, w: float) -> None:
        # Rate-limit: max acceleration per control step
        period = 1.0 / max(1e-3, self.rate_hz)
        max_dv = self.max_linear_accel * period   # max velocity change per step
        max_dw = 2.0 * self.w_max * period        # max angular velocity change per step

        v = float(np.clip(v, self._prev_v_cmd - max_dv, self._prev_v_cmd + max_dv))
        w = float(np.clip(w, self._prev_w_cmd - max_dw, self._prev_w_cmd + max_dw))

        v = float(np.clip(v, self.v_min, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))

        self._prev_v_cmd = v
        self._prev_w_cmd = w

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
