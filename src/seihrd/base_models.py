from typing import Mapping
from pydantic import BaseModel


class A:
    noop = 'noop'
    vaccine = 'vaccine'
    mask = 'mask'
    lockdown = 'lockdown'
    social_distancing = 'social_distancing'


i2a = [
    A.noop,
    A.vaccine,
    A.mask,
    A.lockdown,
    A.social_distancing,
]
a2i = {a: i for i, a in enumerate(i2a)}


class SimHyperParams(BaseModel):
    action_durations: Mapping[str, int]
    max_steps: int
    initial_population: int = 1000


class SubCompartmentPopulations(BaseModel):
    uv: int = 0
    fv: int = 0
    b: int = 0


class Populations(BaseModel):
    susceptible: SubCompartmentPopulations
    exposed1: SubCompartmentPopulations
    exposed2: SubCompartmentPopulations
    infected: SubCompartmentPopulations
    recovered: SubCompartmentPopulations
    hospitalized: SubCompartmentPopulations
    deceased: SubCompartmentPopulations


class Probs(BaseModel):
    vfv: float
    vb: float
    s_e1: float
    e1_s: float
    e1_i: float
    i_r: float
    i_h: float
    i_d: float
    e2_i: float
    h_r: float
    h_d: float
    r_e2: float
    e2_r: float


class State(BaseModel):
    populations: Populations
    probs: Probs
    action_residue: Mapping[str, int]
    epp: float
    hyper_parameters: SimHyperParams
    action_mask: Mapping[str, int]
    step_count: int
    is_done: bool

