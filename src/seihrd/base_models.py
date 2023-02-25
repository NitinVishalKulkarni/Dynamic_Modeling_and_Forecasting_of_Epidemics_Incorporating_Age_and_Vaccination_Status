from dataclasses import dataclass, fields
from typing import Mapping
from numpy.random import normal


class A:
    noop = 'noop'
    social_distancing = 'social_distancing'
    vaccine = 'vaccine'
    mask = 'mask'
    lockdown = 'lockdown'


i2a = [
    # A.noop,
    A.social_distancing,
    A.lockdown,
    A.mask,
    A.vaccine,
]
a2i = {a: i for i, a in enumerate(i2a)}


class DictLike:
    def __getitem__(self, field):
        return getattr(self, field)

    def __setitem__(self, field, value):
        setattr(self, field, value)


@dataclass
class SimHyperParams:
    action_durations: Mapping[str, int]
    action_cool_downs: Mapping[str, int]
    max_steps: int
    initial_population: int = 1000


@dataclass
class SubCompPopulations(DictLike):
    uv: int = 0
    fv: int = 0
    b: int = 0

    def total(self):
        return self.uv + self.fv + self.b


@dataclass
class Populations(DictLike):
    susceptible: SubCompPopulations
    exposed: SubCompPopulations
    infected: SubCompPopulations
    recovered: SubCompPopulations
    hospitalized: SubCompPopulations
    deceased: SubCompPopulations

    def total(self):
        return sum([self[c.name].total() for c in fields(self)])

    def vax(self, comp: str):
        return sum([self[field.name][comp] for field in fields(self)])


def noisy(param):
    return normal(param, param * 0.05, 1)


@dataclass
class SubCompParams:
    uv: float = 0
    fv: float = 0
    b: float = 0

    def noisy(self):
        self.uv = noisy(self.uv)
        self.fv = noisy(self.fv)
        self.b = noisy(self.b)


@dataclass
class Params:
    vfv: float
    vb: float
    alpha: float
    beta: float
    e_s: SubCompParams
    e_i: SubCompParams
    i_r: SubCompParams
    i_h: SubCompParams
    i_d: SubCompParams
    e2_i: SubCompParams
    h_r: SubCompParams
    h_d: SubCompParams
    e_r: SubCompParams

    def noisy(self):
        for f in fields(self):
            field = getattr(self, f.name)
            if isinstance(field, SubCompParams):
                field.noisy()
            else:
                setattr(self, f.name, noisy(field))


@dataclass
class State:
    populations: Populations
    params: Params
    action_residue: Mapping[str, int]
    epp: float
    hyper_parameters: SimHyperParams
    action_mask: Mapping[str, int]
    step_count: int
    is_done: bool

