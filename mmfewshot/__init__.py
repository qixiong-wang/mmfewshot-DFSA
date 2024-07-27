# Copyright (c) OpenMMLab. All rights reserved.

import mmcv

from .mmseg import *

from .utils import *  # noqa: F401, F403
from .version import __version__, short_version


def digit_version(version_str):
    digit_version_ = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version_.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version_.append(int(patch_version[0]) - 1)
            digit_version_.append(int(patch_version[1]))
    return digit_version_


mmcv_minimum_version = '1.3.12'
mmcv_maximum_version = '1.6.0'
mmcv_version = digit_version(mmcv.__version__)


assert (digit_version(mmcv_minimum_version) <= mmcv_version
        <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'


__all__ = ['__version__', 'short_version']
