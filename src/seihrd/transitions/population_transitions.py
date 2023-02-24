from seihrd.base_models import (
    Populations,
    SubCompartmentPopulations,
)
from random import random


class PopulationTransitions:
    def __init__(self):
        self.s = None

    def __call__(self, state):
        self.s = state.copy(deep=True)
        self.vax()

        transitions_list = [
            ('s_e1', 'susceptible', 'exposed1'),
            ('e1_s', 'exposed1', 'susceptible'),
            ('e1_i', 'exposed1', 'infected'),
            ('i_h', 'infected', 'hospitalized'),
            ('i_r', 'infected', 'recovered'),
            ('i_d', 'infected', 'deceased'),
            ('h_r', 'hospitalized', 'recovered'),
            ('h_d', 'hospitalized', 'deceased'),
            ('r_e2', 'recovered', 'exposed2'),
            ('e2_i', 'exposed2', 'infected'),
            ('e2_r', 'exposed2', 'recovered'),
        ]

        for prob, population_from, population_to in transitions_list:
            self.transition_compartment(
                population_from=getattr(self.s.populations, population_from),
                population_to=getattr(self.s.populations, population_to),
                prob=getattr(self.s.probs, prob),
            )

        return self.s

    def vax(self):
        for c in Populations.__fields__:
            population = getattr(self.s.populations, c)

            # uv to fv
            for i in range(population.uv):
                if random() < self.s.probs.vfv:
                    population.uv -= 1
                    population.fv += 1

            # fv to b
            for i in range(population.fv):
                if random() < self.s.probs.vb:
                    population.fv -= 1
                    population.b += 1

    @staticmethod
    def transition_compartment(
        population_from: SubCompartmentPopulations,
        population_to: SubCompartmentPopulations,
        prob: float,
    ):
        # uv
        for i in range(population_from.uv):
            if random() < prob:
                population_from.uv -= 1
                population_to.uv += 1

        # fv
        for i in range(population_from.fv):
            if random() < prob:
                population_from.fv -= 1
                population_to.fv += 1

        # b
        for i in range(population_from.b):
            if random() < prob:
                population_from.b -= 1
                population_to.b += 1

