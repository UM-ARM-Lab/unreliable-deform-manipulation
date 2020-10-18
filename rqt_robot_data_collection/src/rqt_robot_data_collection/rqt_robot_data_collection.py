import pathlib
from typing import Optional, Dict

import hjson
import rospkg
from python_qt_binding.QtWidgets import QWidget

import rospy
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import scenario_map, get_scenario
from python_qt_binding import loadUi
from qt_gui.plugin import Plugin


class RobotDataCollection(Plugin):

    def __init__(self, context):
        super().__init__(context)

        # Give QObjects reasonable names
        self.setObjectName('MyPlugin')

        # Process standalone plugin command-line arguments
        from argparse import ArgumentParser
        parser = ArgumentParser()
        # Add argument(s) to the parser.
        parser.add_argument("-q", "--quiet", action="store_true",
                            dest="quiet",
                            help="Put plugin in silent mode")
        args, unknowns = parser.parse_known_args(context.argv())
        if not args.quiet:
            print('arguments: ', args)
            print('unknowns: ', unknowns)

        # Create QWidget
        self._widget = QWidget()

        # Get path to UI file which should be in the "resource" folder of this package
        ui_file = pathlib.Path(rospkg.RosPack().get_path('rqt_robot_data_collection')) / 'resource' / 'RobotDataCollection.ui'

        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)

        # Give QObjects reasonable names
        self._widget.setObjectName('RobotDataCollection')
        # Show _widget.windowTitle on left-top of each plugin (when
        # it's set in _widget). This is useful when you open multiple
        # plugins at once. Also if you open multiple instances of your
        # plugin at once, these lines add number to make it easy to
        # tell from pane to pane.
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._widget)

        # Setup QT Connections
        self._widget.reset_button.clicked.connect(self.reset_clicked)

        self._widget.scenario_combobox.currentTextChanged = self.scenario_changed
        self._widget.scenario_combobox.addItems(scenario_map.keys())

        params_dir = pathlib.Path(rospkg.RosPack().get_path('link_bot_data')) / 'collect_dynamics_params'
        params_filenames = [p.as_posix() for p in params_dir.glob("*json")]
        self._widget.params_combobox.currentTextChanged = self.params_changed
        self._widget.params_combobox.addItems(params_filenames)

        if len(params_filenames) > 0:
            self._widget.params_edit.setText(params_filenames[0])
        else:
            self.params: Optional[Dict] = None

    def shutdown_plugin(self):
        # TODO unregister all publishers here
        pass

    def save_settings(self, plugin_settings, instance_settings):
        # TODO save intrinsic configuration, usually using:
        # instance_settings.set_value(k, v)
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # TODO restore intrinsic configuration, usually using:
        # v = instance_settings.value(k)
        pass

    # def trigger_configuration(self):
    # Comment in to signal that the plugin has a way to configure
    # This will enable a setting button (gear icon) in each dock widget title bar
    # Usually used to open a modal configuration dialog

    def reset_clicked(self):
        if not self.scenario or not self.params:
            return

        self.scenario.on_before_data_collection(self.params)

    def scenario_changed(self, scenario_name):
        rospy.logwarn("scneario changed")
        self.scenario = get_scenario(scenario_name)

    def params_changed(self, params_name):
        rospy.logwarn("params changed")
        self._widget.params_edit.setText(params_name)
        with open(params_name, 'r') as params_file:
            params_str = params_file.read()
        self.params = hjson.loads(params_str)
