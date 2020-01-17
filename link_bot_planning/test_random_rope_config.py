# coding: utf-8
from link_bot_pycommon.link_bot_pycommon import make_random_rope_configuration
make_random_rope_configuration([-2,2,-2,2], 22, 0.5)
import matplotlib.pyplot as plt
from link_bot_data.visualization import plot_rope_configuration
get_ipython().run_line_magic('pinfo', 'plot_rope_configuration')
plt.figure()
ax = plt.gca()
get_ipython().run_line_magic('pinfo', 'plot_rope_configuration')
plot_rope_configuration(ax, c)
c=make_random_rope_configuration([-2,2,-2,2], 22, 0.5)
plot_rope_configuration(ax, c)
plt.axis("equal")
plot_rope_configuration(ax, c)
plt.clear()
