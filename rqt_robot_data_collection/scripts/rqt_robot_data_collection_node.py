#!/usr/bin/env python

import sys

from rqt_gui.main import Main

plugin = 'rqt_robot_data_collection'
main = Main(filename=plugin)
sys.exit(main.main(standalone=plugin))
