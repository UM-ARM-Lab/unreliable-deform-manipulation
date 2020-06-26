# Repos

## 18.04

```
 Localname                S SCM Version (Spec)     UID  (Spec)  URI  (Spec) [http(s)://...]
 ---------                - --- --------------     -----------  ---------------------------
 sdf_tools                  git melodic  (-)                        4b7a0ab808d0 git@github.com:UM-ARM-Lab/sdf_tools.git
 link_bot                   git victor_shim_gazebo_integration  (-) 025c46236972 git@github.com:UM-ARM-Lab/link_bot.git
 lightweight_vicon_bridge   git master  (-)                         6ed19db45b9b git@github.com:UM-ARM-Lab/lightweight_vicon_bridge.git
 kuka_iiwa_interface        git peter_3d_rope  (-)                  d75e01421417 git@github.com:UM-ARM-Lab/kuka_iiwa_interface
 arc_utilities              git master  (-)                         63dcf2e063f0 git@github.com:UM-ARM-Lab/arc_utilities.git

```

## 16.04 - not sure when things were last pulled from master
```
 or_urdf                     git master  (-)                    2ec13fcd536e git@github.com:UM-ARM-Lab/or_urdf.git
 or_ros_plugin_initializer   git master  (-)                    eaa0f06cb532 git@github.com:UM-ARM-Lab/or_ros_plugin_initializer.git
 or_plugin                   git master  (-)                    5b5739f5d1e6 git@github.com:personalrobotics/or_plugin.git
 openrave_catkin             git master  (-)                    601c5bcb84c0 git@github.com:UM-ARM-Lab/openrave_catkin.git
 kuka_iiwa_interface         git master  (-)                    5a052e3a5791 git@github.com:UM-ARM-Lab/kuka_iiwa_interface.git
 comps                       git master  (-)                    205f7eccd597 git@github.com:UM-ARM-Lab/comps.git
 arm_or_robots               git TrajForwardGeneralization  (-) 1c6635ad4493 git@github.com:UM-ARM-Lab/arm_or_robots.git
 arc_utilities               git master  (-)                    077c9e7d0005 git@github.com:UM-ARM-Lab/arc_utilities.git
```

# Nodes/etc open for Fake Victor
```
roscore
roslaunch victor_fake_hardware_interface fake_dual_arm_lcm_bridge.launch --screen
roslaunch victor_3d_rope_shim static_transforms.launch --screen
roslaunch victor_3d_rope_shim vicon_transform_replacements.launch --screen
rcnew && rosrun arm_or_robots ros_trajectory_forwarder.py _world_frame:="world_origin"
rviz
watch 'cat /proc/cpuinfo | grep MHz | sort -r'
roslaunch victor_3d_rope_shim victor_shim.launch --screen
```

# Nodes/etc open for Gazebo Victor
```
roscore
roslaunch link_bot_gazebo world.launch world_name:=victor_table_rope pause:=false --screen
rviz
watch 'cat /proc/cpuinfo | grep MHz | sort -r'
```