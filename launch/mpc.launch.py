from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    namespace = LaunchConfiguration('namespace')

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

    controller_node = Node(
        package='mpc',
        executable='mpc_node',
        name='mpc',
        namespace=namespace,
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        declare_params,
        declare_namespace,
        controller_node
    ])
