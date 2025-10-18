from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('mpc'), 'config', 'mpc.params.yaml'
        ]),
        description='Path to the YAML file with node parameters.'
    )


    controller_node = Node(
        package='mpc',
        executable='mpc_node',
        name='mpc',
        output='screen',
        parameters=[params_file]
    )


    return LaunchDescription([
                                declare_params,
                                controller_node
                            ])