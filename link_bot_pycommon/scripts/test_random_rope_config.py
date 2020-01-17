#!/usr/bin/env python3

from link_bot_pycommon.link_bot_pycommon import make_random_rope_configuration
import matplotlib.pyplot as plt
from link_bot_data.visualization import plot_rope_configuration

c = make_random_rope_configuration([-2, 2, -2, 2], 22, 0.5, max_angle_rad=2)

plt.figure()
plt.axis("equal")
ax = plt.gca()
plot_rope_configuration(ax, c)
plt.show()
