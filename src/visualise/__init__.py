"""What a CardioConv or CardioHydra run looked at, drawn from the backbone's
``captured`` intermediates one window at a time; ``scripts/visualise.py``
drives it.

``scores.py`` turns a window's capture into rows of numbers (the
pulsatility map's health, the kinematics pooled under each trace's mask,
the label levels and, on a synthetic store, the mass on the true vessels);
``panels.py`` draws the masks beside and over the mean frame;
``kinematics.py`` the field's amplitude, wavelength, speed and heading and
the speed-against-pressure scatter; ``propagation.py`` the arrows that
say which way the pulse travels; ``gifs.py`` the moving pictures;
``window.py`` scores and draws one window with all of them.
"""

import matplotlib

matplotlib.use("Agg")            # write files; never open a window on a cluster

VISUALS_DIR = "visuals"
