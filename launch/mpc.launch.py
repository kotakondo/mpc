from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    namespace = LaunchConfiguration('namespace')
    hardware = LaunchConfiguration('hardware')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('mpc'), 'config', 'mpc.yaml'
        ]),
        description='Path to the YAML file with node parameters.'
    )

    declare_namespace = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the MPC node (topics auto-resolve under this namespace).'
    )

    declare_hardware = DeclareLaunchArgument(
        'hardware',
        default_value='false',
        description='Hardware mode: publishes to cmd_vel_auto instead of cmd_vel.'
    )

    # Choose cmd_vel topic based on hardware flag
    cmd_vel_topic = PythonExpression([
        "'cmd_vel_auto' if '", hardware, "' == 'true' else 'cmd_vel'"
    ])

    controller_node = Node(
        package='mpc',
        executable='mpc_node',
        name='mpc',
        namespace=namespace,
        output='screen',
        parameters=[params_file, {'cmd_vel_topic': cmd_vel_topic}]
    )

    return LaunchDescription([
        declare_params,
        declare_namespace,
        declare_hardware,
        controller_node
    ])
