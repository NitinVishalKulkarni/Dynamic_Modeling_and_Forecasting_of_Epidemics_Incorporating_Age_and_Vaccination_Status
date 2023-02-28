from typing import Sequence
from numpy.random import normal
from pydantic import BaseModel


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


class SimHyperParams(BaseModel):
    action_durations: Sequence[int]
    action_cool_downs: Sequence[int]
    max_steps: int
    initial_population: int = 1000


class SubCompPopulations(BaseModel, DictLike):
    uv: int = 0
    fv: int = 0
    b: int = 0

    def total(self):
        return self.uv + self.fv + self.b


class Populations(BaseModel, DictLike):
    susceptible: SubCompPopulations
    exposed: SubCompPopulations
    infected: SubCompPopulations
    hospitalized: SubCompPopulations
    recovered: SubCompPopulations
    deceased: SubCompPopulations

    def total(self):
        return sum([self[c].total() for c in self.__fields__])

    def vax(self, comp: str):
        return sum([self[field][comp] for field in self.__fields__])

    def to_list(self):
        return [[self[f].uv, self[f].fv, self[f].b] for f in self.__fields__]


def noisy(param):
    return normal(param, param * 0.05, 1)


class SubCompParams(BaseModel, DictLike):
    uv: float = 0
    fv: float = 0
    b: float = 0

    def noisy(self):
        self.uv = noisy(self.uv)
        self.fv = noisy(self.fv)
        self.b = noisy(self.b)


class Params(BaseModel, DictLike):
    vfv: float
    vb: float
    alpha: float
    beta: float
    e_s: SubCompParams  # sigma_s
    e_i: SubCompParams  # zeta_s
    i_r: SubCompParams  # gamma_i
    i_h: SubCompParams  # delta
    i_d: SubCompParams  # mu_i
    e2_i: SubCompParams  # zeta_r
    h_r: SubCompParams  # gamma_h
    h_d: SubCompParams  # mu_h
    e_r: SubCompParams  # sigma_r

    def noisy(self):
        for f in self.__fields__:
            if isinstance(self[f], SubCompParams):
                self[f].noisy()
            else:
                self[f] = noisy(self[f])


class State(BaseModel):
    populations: Populations
    params: Params
    epp: float
    hyper_parameters: SimHyperParams
    action_in_effect: Sequence[int]
    action_cool_down: Sequence[int]
    action_mask: Sequence[int]
    time_step: int
    is_done: bool

