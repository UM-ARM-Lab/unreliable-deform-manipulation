#!/usr/bin/env python

import argparse
import pathlib

from lxml import etree

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.inertia_matrices import box


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('mass', type=float, help='mass')
    parser.add_argument('width', type=float, help='x')
    parser.add_argument('depth', type=float, help='y')
    parser.add_argument('height', type=float, help='z')
    parser.add_argument('outdir', type=pathlib.Path, help='mass')

    args = parser.parse_args()

    args.outdir.mkdir(exist_ok=True, parents=False)

    # x size is width, y size is depth, z size is height
    model_name = f"box_{int(args.width * 100)}_{int(args.depth * 100)}_{int(args.height * 100)}"
    sdf_filename = args.outdir / (model_name + ".sdf")
    config_filename = args.outdir / "model.config"

    write_model_config(config_filename, model_name)
    write_model_sdf(sdf_filename, args, model_name)


def write_model_sdf(sdf_filename, args, model_name):
    sdf_config_str = """
    <sdf version="1.6">
        <model name="moving_box">
            <link name="link_1">
                <pose>0 0 0 0 0 0</pose>
                <inertial>
                    <inertia>
                        <ixx>0</ixx>
                        <ixy>0</ixy>
                        <ixz>0</ixz>
                        <iyy>0</iyy>
                        <iyz>0</iyz>
                        <izz>0</izz>
                    </inertia>
                    <mass>0</mass>
                </inertial>
                <visual name="visual">
                    <geometry>
                        <box>
                            <size>0 0 0</size>
                        </box>
                    </geometry>
                </visual>
                <collision name="box_collision">
                    <geometry>
                        <box>
                            <size>0 0 0</size>
                        </box>
                    </geometry>
                    <surface>
                        <friction>
                            <ode>
                                <mu>5.0</mu>
                                <mu2>5.0</mu2>
                            </ode>
                        </friction>
                    </surface>
                </collision>
            </link>
            <plugin name='position_3d_plugin' filename='libposition_3d_plugin.so'>
                <kP_pos>5</kP_pos>
                <kI_pos>0.0</kI_pos>
                <kD_pos>0.0</kD_pos>
                <kP_vel>100.0</kP_vel>
                <kI_vel>0.0</kI_vel>
                <kD_vel>0.0</kD_vel>
                <max_force>15.0</max_force>
                <max_vel>0.5</max_vel>
                <kP_rot>0.1</kP_rot>
                <kI_rot>0.0</kI_rot>
                <kD_rot>0.0</kD_rot>
                <kP_rot_vel>0.0</kP_rot_vel>
                <kI_rot_vel>0.0</kI_rot_vel>
                <kD_rot_vel>0.0</kD_rot_vel>
                <max_torque>0.0</max_torque>
                <max_rot_vel>0.0</max_rot_vel>
                <link>link_1</link>
            </plugin>
        </model>
    </sdf>
    """
    ixx, iyy, izz = box(mass=args.mass, x=args.width, y=args.depth, z=args.height)
    sdf = etree.fromstring(sdf_config_str)

    model = sdf.find('model')
    model.set('name', model_name)
    link = model.find('link')
    pose = link.find('pose')
    pose.text = f"0 0 {args.height / 2} 0 0 0 0"
    inertial = link.find('inertial')
    inertia = inertial.find('inertia')
    mass = inertial.find('mass')
    mass.text = f"{args.mass}"
    ixx_element = inertia.find('ixx')
    ixx_element.text = f"{ixx}"
    iyy_element = inertia.find('iyy')
    iyy_element.text = f"{iyy}"
    izz_element = inertia.find('izz')
    izz_element.text = f"{izz}"
    visual_box_size = link.find('visual').find("geometry").find("box").find("size")
    visual_box_size.text = f"{args.width} {args.depth} {args.height}"
    collision_box_size = link.find('collision').find("geometry").find("box").find("size")
    collision_box_size.text = f"{args.width} {args.depth} {args.height}"

    with sdf_filename.open("w") as f:
        f.write(etree.tostring(sdf, pretty_print=True, encoding='unicode'))


def write_model_config(config_filename, model_name):
    model_config_str = "<model><name>small_moving_box</name><version>1.0</version><sdf version='1.6'>model.sdf</sdf></model>"
    model_xml = etree.fromstring(model_config_str)
    name = model_xml.find('name')
    name.text = model_name
    sdf = model_xml.find('sdf')
    sdf.text = f"{model_name}.sdf"
    with config_filename.open("w") as f:
        f.write(etree.tostring(model_xml, pretty_print=True, encoding='unicode'))


if __name__ == '__main__':
    main()
