""" GREEN
Verkruysse, W., Svaasand, L. O. & Nelson, J. S.
Remote plethysmographic imaging using ambient light.
Optical. Express 16, 21434–21445 (2008).
"""

from einops import rearrange

from unsupervised_methods import utils


def GREEN(frames):
    rgb = utils.process_video(frames)                     # (1, 3, T)
    return rearrange(rgb[:, 1, :], "b t -> (b t)")        # the green trace
