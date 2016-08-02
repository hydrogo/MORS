import numpy as np
import pandas as pd
import scipy.optimize as so
import sys
sys.path.append('../models')

def calibrate(data, simulation_routine='HBV-s', objective_function='NS', method='DE'):
    '''
    Inputs:
    1. data - meteorological forcing data packed in pandas dataframe:
        'Temp' - daily temperature (Celsium degrees)
        'Prec' - daily precipitation (mm/day)
        'Evap' - daily potential evapotranspiration (mm/day)
    2.  'Qobs' - daily river runoff (mm\day)
    3. simulation routines (default='HBV')
    'HBV'
    'HBV-s'
    'GR4J'
    'GR4J-Cema-Neige'
    'SIMHYD'
    'SIMHYD-Cema-Neige'
    4. objective function for optimization:
        'NS'   - Nash-Sutcliffe model efficiency criterion
    5. optimization method:
        'DE' - differential evolution (global optimization routine)
    Output:
        list of optimal model parameters
    '''
    ### data preparation
    Qobs = data['Qobs']
    Qm = Qobs.mean()

    ### initialize the model used for parameters optimization
    if simulation_routine == 'HBV':
        import hbv
        model = hbv.simulation
        bnds = hbv.bounds()

    elif simulation_routine == 'HBV-s':
        import hbv_s
        model = hbv_s.simulation
        bnds = hbv_s.bounds()

    elif simulation_routine == 'GR4J':
        import gr4j
        model = gr4j.simulation
        bnds = gr4j.bounds()

    elif simulation_routine == 'GR4J-Cema-Neige':
        import gr4j_cemaneige
        model = gr4j_cemaneige.simulation
        bnds = gr4j_cemaneige.bounds()

    elif simulation_routine == 'SIMHYD':
        import simhyd
        model = simhyd.simulation
        bnds = simhyd.bounds()

    elif simulation_routine == 'SIMHYD-Cema-Neige':
        import simhyd_cemaneige
        model = simhyd_cemaneige.simulation
        bnds = simhyd_cemaneige.bounds()

    else:
        print("Incorrect simulation routine name, try one of:\
        'HBV', 'HBV-s', 'GR4J', 'GR4J-Cema-Neige', 'SIMHYD', 'SIMHYD-Cema-Neige' ")

    ### initialize objective function for optimization
    def obj_func_calc(params):
            # simulate hydrograph
            Qsim = model(data, params)
            # calculate objective function value
            return ((Qobs-Qsim)**2).sum()/((Qobs-Qm)**2).sum()

    if objective_function == 'NS':
        pass
    else:
        print("Incorrect objective function name, only 'NS' is available ")

    ### initialize optimization algorithm
    if method == 'DE':
        optimizer = so.differential_evolution
    else:
        print("Incorrect optimization method name, only 'DE' is available ")

    result = optimizer(obj_func_calc, bnds, maxiter=5, polish=False, disp=True)

    opt_param = result.x

    return opt_param
