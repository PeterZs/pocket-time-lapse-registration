import pandas as pd;
import numpy as np
import pandas as pd
import string

import itertools
def alphalist(n=10):
    def iter_all_strings():
        for size in itertools.count(1):
            for s in itertools.product(string.ascii_lowercase, repeat=size):
                yield "".join(s)

    rlist = [];
    for s in iter_all_strings():
        rlist.append(s);
        if (len(rlist) >= n):
            return rlist;


def RANDOM_DATAFRAME(size=None):
    if(size is None):
        size = (10,3);
    return pd.DataFrame(np.random.rand(*size), columns=alphalist(size[1]));

