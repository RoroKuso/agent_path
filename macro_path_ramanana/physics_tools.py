"""
File containing constants and tools to compute power and speed from ground data.
"""

import numpy as np
from typing import Tuple

"""Environement's constants"""
g = 9.81
pixel_in_meter = 4.25
max_altitude = 418
cols = 964

"""Agent's constants"""
mass = 70 # mass
L = 1 # leg length
T =  2 * np.pi * (2 * L / 3 / g) ** 0.5 # natural period  of the leg
T_power = T/2 # period used in power computations
P_tot = 100 # total power spent by the agent
foot_surface = 0.02 # surface of the foot
v_max = 2 # the maximum speed reachable by the agent. Going any faster will set the speed to 0
v_precautious = 0.2 # the speed the agent will reach when there's a steep slope
v_min = 0.005 # the minimum speed reachable by the agent


"""PHYSICAL CONVERSION FUNCTIONS"""

def power_to_walk(v: float) -> float:
    """returns the power required to walk at a given speed"""
    if (v * T / 8 / L)**2 < 1:
        return mass * g * L / T_power * (1 - np.sqrt(1 - (v * T / 8 / L)**2))
    return np.Infinity

def walking_speed(P: float) -> float:
    """returns the walking speed corresponding to a given power"""
    if P < 0:
        return v_min
    
    if 1 > (1 - P * T_power / mass / g / L)**2:
        v = 8 * L / T * np.sqrt(1 - (1 - P * T_power / mass / g / L)**2)
        return max(v, v_min) if v <= v_max else v_precautious
 
    return v_precautious

def power_to_deform_ground(ground: Tuple[float, float]) -> float:
    """returns dissipated power from ground deformation.
    
    Parameters
    ----------
    ground: Tuple[float, float]
        Tuple containing ground caracteristics (respectively Young's modulus, max deformation depth).

    """
    E, layer_thickness = ground[0], ground[1]
    return (mass * g)**2 * layer_thickness / (E * foot_surface * T)

def power_from_com_height_variation(delta_z: float, d: float) -> float:
    """returns dissipated power from COM height variation.
    
    Parameters
    ----------
    delta_z: float
        height variation
    d: float
        vertical variation (along walking direction)
    """
    v = walking_speed(0.5 * P_tot) # We use an abitrary speed for the walking speed.
    walking_com_dz = L - np.sqrt(L**2 - (v * T / 8)**2)
    
    if np.abs(delta_z) > 1.25:
        return np.Infinity
    
    if delta_z > walking_com_dz:
        delta_z -= walking_com_dz
    elif delta_z < 0:
        delta_z = delta_z
    else:
        delta_z = 0

    restitution_coef = 0.8 # @MYLENE: Ce coefficient doit être entre 0 et 1. On pourra être amené à le modifier si on n'arrive pas à converger
    
    # Thales' theorem
    return restitution_coef * mass * g *  delta_z * v / d

def compute_walking_speed(delta_z: float, d: float, ground: Tuple[float, float]) -> float:
    """returns walking speed from walking power `P_{walk} = P_{tot} - P_{dissipated}`.
    
    Parameters
    ----------
    delta_z: float
        Height variation
    d: float
        vertical variation (along walking direction)
    ground: Tuple[float, float]
        Tuple containing ground caracteristics (respectively Young's modulus, max deformation depth).
    """
    # Compute the different dissipated powers
    P_ground = power_to_deform_ground(ground)
    
    if P_ground < 0:
        return 0
    
    P_com_variation = power_from_com_height_variation(delta_z, d)

    if P_com_variation == np.Infinity:
        return 0
    
    # Compute the power left to walk
    P_walk = P_tot - P_ground - P_com_variation

    return walking_speed(P_walk)

def true_height(value: int, scale: int) -> float:
    """Converts pixel height to true height"""
    return value / scale * max_altitude