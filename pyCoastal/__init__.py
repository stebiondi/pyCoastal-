"""
pycoastal - A Python toolkit for coastal engineering computations.
Stefano Biondi, UF
"""
import sys
import inspect
import pyCoastal

def __getattr__(name):
    # If they tried to access a function that doesn't exist:
    # Build a categorized list of all available callables
    groups = {
        "Wave Tools": pyCoastal.wave_tools,
        "Structure": pyCoastal.structural,
        "Morphodynamics": pyCoastal.morphodynamics,
        "Sediment Transport": pyCoastal.sediment_transport,
    }

    available = []
    for grp, mod in groups.items():
        funcs = [f for f, obj in inspect.getmembers(mod, inspect.isfunction)]
        if funcs:
            available.append(f"\n{grp}:\n  " + "\n  ".join(funcs))

    message = (
        f"'{name}' not found in pyCoastal. Available functions:" +
        "".join(available) +
        "\n\nUse cs.<function_name>(...) to call."
    )
    raise AttributeError(message)
    
from .wave_tools import (
    dispersion,
    wave_number,
    surf_similarity,
    breaker_type,
    ursell_number,
    wave_setup
)

from .structural import (
    hudson_dn50,
    vandermeer_dn50,
    goda_wave_force,
    hunt_runup,
    stockdon_runup,
    iribarren_stability
)

from .morphodynamics import (
    bruuns_rule,
    exner_change
)

from .sediment_transport import (
    shields_parameter,
    van_rijn_bedload,
    van_rijn_suspended,
    bijker_bedload,
    bagnold_sediment,
    cerc_transport,
    einstein_bedload,
    izbash_current   
)
