#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Get Directories
    # We need stage_ros2 for Rviz, but dqn_project for the WORLD file
    stage_pkg_dir = get_package_share_directory('stage_ros2')
    dqn_pkg_dir = get_package_share_directory('dqn_project')
    
    stage_launch_dir = os.path.join(stage_pkg_dir, 'launch')

    # 2. Define Arguments (Exactly as you had them)
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    enforce_prefixes = LaunchConfiguration('enforce_prefixes')
    declare_prefixes = DeclareLaunchArgument(
        'enforce_prefixes', default_value='false',
        description='on true a prefixes are used for a single robot environment')
    
    use_stamped_velocity = LaunchConfiguration('use_stamped_velocity')
    declare_stamped = DeclareLaunchArgument(
        'use_stamped_velocity', default_value='false',
        description='on true stage will accept TwistStamped command messages')

    one_tf_tree = LaunchConfiguration('one_tf_tree')
    declare_one_tf = DeclareLaunchArgument(
        'one_tf_tree', default_value='false',
        description='on true all tfs are published with a namespace on /tf')
   
    namespace = LaunchConfiguration('namespace')
    declare_namespace = DeclareLaunchArgument(
        'namespace', default_value='', description='Top-level namespace')

    # Rviz Argument (Defaulted to False since you said you don't need it)
    rviz = LaunchConfiguration('rviz')
    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='False', description='Whether run rviz')

    # --- CRITICAL CHANGE: Point to YOUR world file ---
    # We construct the absolute path to 'clean_cave.world' in your package
    world_path = os.path.join(dqn_pkg_dir, 'worlds', 'clean_cave.world')
    
    # 3. Define Nodes

    # Your Reset Wrapper
    reset_node = Node(
        package='dqn_project',
        executable='reset_stage',
        name='odom_reset_wrapper',
        output='screen'
    )

    # Stage Node 
    # (We define this DIRECTLY instead of using IncludeLaunchDescription)
    # (This is the only way to force it to use a custom world path)
    stage_node = Node(
        package='stage_ros2',
        executable='stage_ros2',
        name='stage',
        output='screen',
        arguments=[world_path],
        parameters=[{'use_sim_time': True}]
    )

    # 4. Return Launch Description
    return LaunchDescription([
        declare_namespace,
        declare_stamped,
        declare_rviz,
        declare_prefixes,
        declare_one_tf,
        
        # We don't need 'declare_world' anymore because we hardcoded the path 
        # to your specific clean_cave file above.
        
        stage_node,
        reset_node,
        
        # Include standard Rviz (Optional, conditioned on 'rviz' argument)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(stage_launch_dir, 'rviz.launch.py')),
            condition=IfCondition(rviz),
            launch_arguments={'namespace': namespace,
                              'use_sim_time': use_sim_time,
                              'config': 'cave'}.items()), # Rviz config matches cave layout
    ])