#ifndef LBV_EIGEN_ROS_CONVERSIONS_HPP
#define LBV_EIGEN_ROS_CONVERSIONS_HPP

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <victor_hardware_interface/JointValueQuantity.h>

////////////////////////////////////////////////////////////////////////////////

template<typename Output>
Output ConvertTo(geometry_msgs::Point const& geo);

template<typename Output>
Output ConvertTo(geometry_msgs::Vector3 const& eig);

template<typename Output>
Output ConvertTo(Eigen::Vector3d const& eig);

template<>
inline Eigen::Vector3d ConvertTo<Eigen::Vector3d>(geometry_msgs::Point const& geo)
{
    return Eigen::Vector3d(geo.x, geo.y, geo.z);
}

template<>
inline Eigen::Vector3d ConvertTo<Eigen::Vector3d>(geometry_msgs::Vector3 const& eig)
{
    return Eigen::Vector3d(eig.x, eig.y, eig.z);
}

template<>
inline geometry_msgs::Point ConvertTo<geometry_msgs::Point>(Eigen::Vector3d const& eig)
{
    geometry_msgs::Point geo;
    geo.x = eig.x();
    geo.y = eig.y();
    geo.z = eig.z();
    return geo;
}

template<>
inline geometry_msgs::Vector3 ConvertTo<geometry_msgs::Vector3>(Eigen::Vector3d const& eig)
{
    geometry_msgs::Vector3 geo;
    geo.x = eig.x();
    geo.y = eig.y();
    geo.z = eig.z();
    return geo;
}

////////////////////////////////////////////////////////////////////////////////

template<typename Output>
Output ConvertTo(geometry_msgs::Pose const& pose);

template<typename Output>
Output ConvertTo(Eigen::Affine3d const& transform);

template<typename Output>
Output ConvertTo(Eigen::Isometry3d const& transform);

template<typename Output>
Output ConvertTo(Eigen::Translation3d const& translation);

template<>
inline Eigen::Isometry3d ConvertTo<Eigen::Isometry3d>(geometry_msgs::Pose const& pose)
{
    Eigen::Translation3d const trans(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Quaterniond const quat(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    return trans * quat;
}

template<>
inline Eigen::Affine3d ConvertTo<Eigen::Affine3d>(geometry_msgs::Pose const& pose)
{
    Eigen::Translation3d const trans(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Quaterniond const quat(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    return trans * quat;
}

template<>
inline geometry_msgs::Transform ConvertTo<geometry_msgs::Transform>(Eigen::Affine3d const& transform)
{
    geometry_msgs::Transform geo_transform;
    geo_transform.translation.x = transform.translation().x();
    geo_transform.translation.y = transform.translation().y();
    geo_transform.translation.z = transform.translation().z();
    const Eigen::Quaterniond quat(transform.rotation());
    geo_transform.rotation.x = quat.x();
    geo_transform.rotation.y = quat.y();
    geo_transform.rotation.z = quat.z();
    geo_transform.rotation.w = quat.w();
    return geo_transform;
}

template<>
inline geometry_msgs::Transform ConvertTo<geometry_msgs::Transform>(Eigen::Isometry3d const& transform)
{
    geometry_msgs::Transform geo_transform;
    geo_transform.translation.x = transform.translation().x();
    geo_transform.translation.y = transform.translation().y();
    geo_transform.translation.z = transform.translation().z();
    const Eigen::Quaterniond quat(transform.rotation());
    geo_transform.rotation.x = quat.x();
    geo_transform.rotation.y = quat.y();
    geo_transform.rotation.z = quat.z();
    geo_transform.rotation.w = quat.w();
    return geo_transform;
}

template<>
inline geometry_msgs::Transform ConvertTo<geometry_msgs::Transform>(Eigen::Translation3d const& translation)
{
    geometry_msgs::Transform geo_transform;
    geo_transform.translation.x = translation.translation().x();
    geo_transform.translation.y = translation.translation().y();
    geo_transform.translation.z = translation.translation().z();
    geo_transform.rotation.x = 0;
    geo_transform.rotation.y = 0;
    geo_transform.rotation.z = 0;
    geo_transform.rotation.w = 1;
    return geo_transform;
}

////////////////////////////////////////////////////////////////////////////////

template<typename Output>
Output ConvertTo(geometry_msgs::Transform const& transform);

template<>
inline Eigen::Isometry3d ConvertTo<Eigen::Isometry3d>(geometry_msgs::Transform const& transform)
{
    Eigen::Translation3d const trans(transform.translation.x, transform.translation.y, transform.translation.z);
    Eigen::Quaterniond const quat(transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
    return trans * quat;
}

template<>
inline Eigen::Affine3d ConvertTo<Eigen::Affine3d>(geometry_msgs::Transform const& transform)
{
    Eigen::Translation3d const trans(transform.translation.x, transform.translation.y, transform.translation.z);
    Eigen::Quaterniond const quat(transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
    return trans * quat;
}

////////////////////////////////////////////////////////////////////////////////

template<typename Output>
Output ConvertTo(Eigen::VectorXd const& eig);

template<>
inline std::vector<double> ConvertTo<std::vector<double>>(Eigen::VectorXd const& eig)
{
    std::vector<double> vec((size_t)eig.size());
    for (size_t idx = 0; idx < (size_t)eig.size(); idx++)
    {
        const double val = eig[(ssize_t)idx];
        vec[idx] = val;
    }
    return vec;
}

////////////////////////////////////////////////////////////////////////////////

template<typename Output>
Output ConvertTo(victor_hardware_interface::JointValueQuantity const& jvq);

template<>
inline Eigen::VectorXd ConvertTo<Eigen::VectorXd>(victor_hardware_interface::JointValueQuantity const& jvq)
{
    Eigen::VectorXd eig(7);
    eig[0] = jvq.joint_1;
    eig[1] = jvq.joint_2;
    eig[2] = jvq.joint_3;
    eig[3] = jvq.joint_4;
    eig[4] = jvq.joint_5;
    eig[5] = jvq.joint_6;
    eig[6] = jvq.joint_7;
    return eig;
}

////////////////////////////////////////////////////////////////////////////////

template<typename Input, typename Output>
std::vector<Output, Eigen::aligned_allocator<Output>> ConvertTo(std::vector<Input> const &geom)
{
    std::vector<Output, Eigen::aligned_allocator<Output>> eig(geom.size());
    for (size_t idx = 0; idx < geom.size(); ++idx)
    {
        eig[idx] = ConvertTo<Output>(geom[idx]);
    }
    return eig;
}

#endif // LBV_EIGEN_ROS_CONVERSIONS_HPP