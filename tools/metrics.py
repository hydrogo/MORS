import numpy as np
def NS(Qobs, Qsim):
    return 1 - np.sum((Qobs-Qsim)**2) / (np.sum((Qobs-Qobs.mean())**2))