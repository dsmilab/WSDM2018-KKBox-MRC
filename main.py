import pandas as pd
import numpy as np
import lightgbm as lgb

import gc


def transform_isrc_to_year(isrc):
    if type(isrc) != str:
        return np.nan
    # this year 2017
    suffix = int(isrc[5:7])
    return 1900 + suffix if suffix > 17 else 2000 + suffix



