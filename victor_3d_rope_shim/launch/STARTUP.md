# Repos

## 18.04: ~/catkin_ws

```
 Localname                S SCM Version (Spec)                         UID  (Spec)  URI  (Spec) [http(s)://...]
 ---------                - --- --------------                         -----------  ---------------------------
 sdf_tools                  git melodic  (-)                           4b7a0ab808d0 git@github.com:UM-ARM-Lab/sdf_tools.git
 link_bot                   git kinematic_victor_rope_integration  (-) 45945173c12d git@github.com:UM-ARM-Lab/link_bot.git
 lightweight_vicon_bridge   git master  (-)                            6ed19db45b9b git@github.com:UM-ARM-Lab/lightweight_vicon_bridge.git
 kuka_iiwa_interface        git peter_3d_rope  (-)                     d75e01421417 git@github.com:UM-ARM-Lab/kuka_iiwa_interface
 arc_utilities              git no_tests  (-)                          63dcf2e063f0 git@github.com:UM-ARM-Lab/arc_utilities.git

```

## 16.04 virtual: ~/catkin_ws
```
 or_urdf                     git master  (-)                    2ec13fcd536e git@github.com:UM-ARM-Lab/or_urdf.git
 or_ros_plugin_initializer   git master  (-)                    eaa0f06cb532 git@github.com:UM-ARM-Lab/or_ros_plugin_initializer.git
 or_plugin                   git master  (-)                    5b5739f5d1e6 git@github.com:personalrobotics/or_plugin.git
 openrave_catkin             git master  (-)                    601c5bcb84c0 git@github.com:UM-ARM-Lab/openrave_catkin.git
 kuka_iiwa_interface         git master  (-)                    5a052e3a5791 git@github.com:UM-ARM-Lab/kuka_iiwa_interface.git
 comps                       git master  (-)                    205f7eccd597 git@github.com:UM-ARM-Lab/comps.git
 arm_or_robots               git TrajForwardGeneralization  (-) 1c6635ad4493 git@github.com:UM-ARM-Lab/arm_or_robots.git
 arc_utilities               git no_tests  (-)                  077c9e7d0005 git@github.com:UM-ARM-Lab/arc_utilities.git
```

# Nodes/etc open for Gazebo Victor
```
roscore
roslaunch link_bot_gazebo victor.launch pause:=true world_name:=victor_table_rope --screen
roslaunch victor_3d_rope_shim victor_shim.launch --screen
rviz
rosrun victor_3d_rope_shim test_move.py
```

# Nodes/etc open for Fake Victor
```
roscore
roslaunch victor_fake_hardware_interface fake_dual_arm_lcm_bridge.launch --screen
roslaunch victor_3d_rope_shim vicon.launch live:=false --screen
rcnova && rosrun arm_or_robots ros_trajectory_forwarder.py _world_frame:="world_origin"
roslaunch victor_3d_rope_shim victor_shim.launch --screen
rviz
rosrun victor_3d_rope_shim test_move.py
```

# Nodes/etc open for Real Victor
```
roscore
rcnova && roslaunch victor_hardware_interface dual_arm_lcm_bridge.launch --screen
roslaunch victor_3d_rope_shim vicon.launch live:=false --screen
rcnova && rosrun arm_or_robots ros_trajectory_forwarder.py _world_frame:="world_origin"
roslaunch victor_3d_rope_shim victor_shim.launch gazebo:=false --screen
rviz
rosrun victor_3d_rope_shim test_move.py
```