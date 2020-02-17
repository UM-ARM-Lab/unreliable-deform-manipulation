#!/usr/bin/env python
import rospy

from link_bot_gazebo.srv import GetObjects, GetObject, GetObjectRequest, GetObjectsResponse
from std_msgs.msg import String

object_services = {}


def objects_handler(req):
    global object_services

    res = GetObjectsResponse()
    for service in object_services.values():
        object_req = GetObjectRequest()
        object_res = service.call(object_req)
        res.objects.objects.append(object_res.object)

    return res


def register_object_handler(msg):
    global object_services

    new_object_service = rospy.ServiceProxy(msg.data, GetObject)
    object_services[msg.data] = new_object_service


if __name__ == '__main__':
    rospy.init_node('objects_server')

    objects_service = rospy.Service("/objects", GetObjects, objects_handler)
    register_object = rospy.Subscriber("/register_object", String, register_object_handler)
    rospy.spin()
