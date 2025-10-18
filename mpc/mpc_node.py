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
from nav_msgs.msg import Odometry

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose

import casadi as ca
import do_mpc


YELLOW = "\033[93m"
RESET  = "\033[0m"


def wrap_angle(a):
    return ca.atan2(ca.sin(a), ca.cos(a))


def yaw_from_quat(q) -> float:
    # x,y,z,w
    siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MPCNode(Node):
    def __init__(self) -> None:
        super().__init__('mpc')

        ### Params
        self.declare_parameter('use_tf_pose', False)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tracking_frame', 'odom')

        self.declare_parameter('pose_topic', '/pose')
        self.declare_parameter('goal_topic', '/goal')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('control_rate_hz', 30.0)

        # MPC params
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('N_horizon', 20)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('v_max', 1.0)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('Q_pos', 5.0)
        self.declare_parameter('Q_yaw', 1.0)
        self.declare_parameter('R_v', 0.1)
        self.declare_parameter('R_w', 0.05)
        self.declare_parameter('dR_v', 0.01)
        self.declare_parameter('dR_w', 0.01)

        ### Load parameters
        self.use_tf_pose: bool = self.get_parameter('use_tf_pose').get_parameter_value().bool_value
        self.base_frame: str   = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tracking_frame: str = self.get_parameter('tracking_frame').get_parameter_value().string_value
        self.pose_topic: str   = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.goal_topic: str   = self.get_parameter('goal_topic').get_parameter_value().string_value
        self.cmd_vel_topic: str = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.goal_tol: float = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.rate_hz: float  = self.get_parameter('control_rate_hz').get_parameter_value().double_value

        self.dt: float = self.get_parameter('dt').get_parameter_value().double_value
        self.N:  int   = int(self.get_parameter('N_horizon').get_parameter_value().integer_value)
        self.v_min: float = self.get_parameter('v_min').get_parameter_value().double_value
        self.v_max: float = self.get_parameter('v_max').get_parameter_value().double_value
        self.w_max: float = self.get_parameter('w_max').get_parameter_value().double_value
        self.Q_pos: float = self.get_parameter('Q_pos').get_parameter_value().double_value
        self.Q_yaw: float = self.get_parameter('Q_yaw').get_parameter_value().double_value
        self.R_v: float   = self.get_parameter('R_v').get_parameter_value().double_value
        self.R_w: float   = self.get_parameter('R_w').get_parameter_value().double_value
        self.dR_v: float  = self.get_parameter('dR_v').get_parameter_value().double_value
        self.dR_w: float  = self.get_parameter('dR_w').get_parameter_value().double_value

        ### TF
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ### IO 
        if not self.use_tf_pose:
            self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 10)
        else:
            self.pose_sub = None
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self._goal_cb, 10)
        self.cmd_pub  = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        ### State 
        self.latest_pose: Optional[PoseStamped] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.cur_pose_tracking: Optional[PoseStamped] = None
        self.goal_reached: bool = False
        self._mpc_ready: bool = False

        ### Build MPC after params are known
        self._build_mpc()

        ### Control timer
        period = 1.0 / max(1e-3, self.rate_hz)
        self.timer = self.create_timer(period, self._control_step)
        self.get_logger().info('MPC controller started.')

    # ---------------- Callbacks ----------------
    def _pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def _goal_cb(self, msg: PoseStamped) -> None:
        # Transform goal into tracking frame at receipt time
        g = msg
        if g.header.frame_id and g.header.frame_id != self.tracking_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame=self.tracking_frame,
                    source_frame=g.header.frame_id,
                    time=rclpy.time.Time())
                g = do_transform_pose(g, tf)
                g.header.frame_id = self.tracking_frame
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f'Goal transform failed: {e}')
                return
        self.goal_pose = g
        self.goal_reached = False
        self._mpc_ready = False  # will reinitialize at next tick
        self.get_logger().info('Received new goal.')

    # ---------------- Pose helpers ----------------
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
                pose = do_transform_pose(base_in_base, tf)
                pose.header.frame_id = self.tracking_frame
                return pose
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
                return None
        else:
            if self.latest_pose is None:
                return None
            ps = self.latest_pose
            if ps.header.frame_id and ps.header.frame_id != self.tracking_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        target_frame=self.tracking_frame,
                        source_frame=ps.header.frame_id,
                        time=rclpy.time.Time())
                    ps = do_transform_pose(ps, tf)
                    ps.header.frame_id = self.tracking_frame
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    self.get_logger().warn(f'Pose transform failed: {e}')
                    return None
            return ps

    ### MPC build 
    def _build_mpc(self) -> None:
        model = do_mpc.model.Model('discrete')

        # States
        pos = model.set_variable('_x', 'pos', (2,1))
        yaw = model.set_variable('_x', 'yaw')

        # Inputs
        v = model.set_variable('_u', 'v')
        w = model.set_variable('_u', 'w')

        # Time-varying parameters (references)
        p_ref   = model.set_variable('_tvp', 'p_ref', (2,1))
        yaw_ref = model.set_variable('_tvp', 'yaw_ref')

        # Discrete dynamics
        pos_next = pos + self.dt * ca.vertcat(ca.cos(yaw)*v, ca.sin(yaw)*v)
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

        # Costs
        pos_err = pos - p_ref
        yaw_err = wrap_angle(yaw - yaw_ref)
        lterm = (
            self.Q_pos * ca.sumsqr(pos_err)
            + self.Q_yaw * ca.sumsqr(yaw_err)
            + self.R_v * v**2
            + self.R_w * w**2
        )
        mterm = 2.0 * (self.Q_pos * ca.sumsqr(pos_err) + self.Q_yaw * ca.sumsqr(yaw_err))
        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(v=self.dR_v, w=self.dR_w)

        # Bounds
        mpc.bounds['lower', '_u', 'v'] = self.v_min
        mpc.bounds['upper', '_u', 'v'] = self.v_max
        mpc.bounds['lower', '_u', 'w'] = -self.w_max
        mpc.bounds['upper', '_u', 'w'] =  self.w_max

        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            # Default zeros
            for k in range(self.N+1):
                tvp_template['_tvp', k, 'p_ref']   = np.zeros((2,1))
                tvp_template['_tvp', k, 'yaw_ref'] = 0.0

            if self.goal_pose is None:
                return tvp_template

            gx = float(self.goal_pose.pose.position.x)
            gy = float(self.goal_pose.pose.position.y)
            # If goal orientation is provided, use it; otherwise keep yaw_ref free
            q = self.goal_pose.pose.orientation
            # If the quaternion is near-zero, keep yaw_ref=0
            if abs(q.x) + abs(q.y) + abs(q.z) + abs(q.w) < 1e-6:
                yaw_goal = 0.0
            else:
                yaw_goal = yaw_from_quat(q)

            for k in range(self.N+1):
                tvp_template['_tvp', k, 'p_ref']   = np.array([[gx],[gy]], dtype=float)
                tvp_template['_tvp', k, 'yaw_ref'] = float(yaw_goal)
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()

        self.model = model
        self.mpc = mpc
        self.get_logger().info(f"{YELLOW}MPC initialized (goal tracking).{RESET}")

    ### Control loop 
    def _control_step(self) -> None:
        # Current pose
        cur = self._current_pose_tracking()
        if cur is None:
            self._publish_cmd(0.0, 0.0)
            return
        self.cur_pose_tracking = cur

        if self.goal_pose is None:
            self._publish_cmd(0.0, 0.0)
            return

        # Stop near goal (position only; yaw can settle through cost)
        dx = self.goal_pose.pose.position.x - cur.pose.position.x
        dy = self.goal_pose.pose.position.y - cur.pose.position.y
        dist = math.hypot(dx, dy)
        if dist <= self.goal_tol:
            self._publish_cmd(0.0, 0.0)
            if not self.goal_reached:
                self.get_logger().info('MPC: goal reached.')
                self.goal_reached = True
            return
        else:
            self.goal_reached = False

        # State vector
        x = float(cur.pose.position.x)
        y = float(cur.pose.position.y)
        yaw = yaw_from_quat(cur.pose.orientation)
        x_vec = np.array([[x], [y], [yaw]], dtype=float)

        if not self._mpc_ready:
            self.mpc.x0 = x_vec
            self.mpc.set_initial_guess()
            self._mpc_ready = True
            self._publish_cmd(0.0, 0.0)
            return

        # Solve MPC
        try:
            t0 = time.time()
            u = self.mpc.make_step(x_vec)  # shape (2,1)
            t1 = time.time()
            self.get_logger().debug(f'MPC solve: {(t1-t0)*1000:.1f} ms')
        except Exception as e:
            self.get_logger().warn(f'MPC solver failed: {e}')
            self._publish_cmd(0.0, 0.0)
            return

        v_cmd = float(u[0])
        w_cmd = float(u[1])

        # Simple taper as we approach the goal (helps avoid overshoot)
        v_cmd *= max(0.1, min(1.0, dist / (self.goal_tol + 1.0)))

        self._publish_cmd(v_cmd, w_cmd)

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x  = float(np.clip(v, self.v_min, self.v_max))
        msg.angular.z = float(np.clip(w, -self.w_max, self.w_max))
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
