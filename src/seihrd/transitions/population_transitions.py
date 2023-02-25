from copy import deepcopy
from seihrd.base_models import (
    Populations,
    SubCompPopulations,
    State,
)
from random import random


class PopulationTransitions:
    def __init__(self):
        pass

    def __call__(self, state: State):
        s = deepcopy(state)
        po = deepcopy(s.populations)
        pa = deepcopy(s.params)
        pa.noisy()

        i = (po.infected.total() ** pa.alpha) / po.total()

        s.populations.susceptible = SubCompPopulations(
            uv=int(
                po.susceptible.uv
                - (pa.beta * po.susceptible.uv * i)
                + (pa.e_s.uv * po.exposed.uv)
                - (pa.vfv * po.susceptible.uv)
            ),
            fv=int(
                po.susceptible.fv
                - (pa.beta * po.susceptible.fv * i)
                + (pa.e_s.fv * po.exposed.fv)
                + (pa.vfv * po.susceptible.uv)
                - (pa.vb * po.susceptible.fv)
            ),
            b=int(
                po.susceptible.b
                - (pa.beta * po.susceptible.b * i)
                + (pa.e_s.b * po.exposed.b)
                + (pa.vb * po.susceptible.fv)
            )
        )

        s.populations.exposed = SubCompPopulations(
            uv=int(
                po.exposed.uv
                + (pa.beta * po.susceptible.uv * i)
                + (pa.beta * po.recovered.uv * i)
                - (pa.e_i.uv * po.exposed.uv)
                - (pa.e2_i.uv * po.exposed.uv)
                - (pa.e_s.uv * po.exposed.uv)
                - (pa.e_r.uv * po.exposed.uv)
                - (pa.vfv * po.exposed.uv)
            ),
            fv=int(
                po.exposed.fv
                + (pa.beta * po.susceptible.fv * i)
                + (pa.beta * po.recovered.fv * i)
                - (pa.e_i.fv * po.exposed.fv)
                - (pa.e2_i.fv * po.exposed.fv)
                - (pa.e_s.fv * po.exposed.fv)
                - (pa.e_r.fv * po.exposed.fv)
                + (pa.vfv * po.exposed.uv)
                - (pa.vb * po.exposed.fv)
            ),
            b=int(
                po.exposed.b
                + (pa.beta * po.susceptible.b * i)
                + (pa.beta * po.recovered.b * i)
                - (pa.e_i.b * po.exposed.b)
                - (pa.e2_i.b * po.exposed.b)
                - (pa.e_s.b * po.exposed.b)
                - (pa.e_r.b * po.exposed.b)
                + (pa.vb * po.exposed.fv)
            )
        )

        s.populations.infected = SubCompPopulations(
            uv=self._updated_infected('uv', po, pa),
            fv=self._updated_infected('fv', po, pa),
            b=self._updated_infected('b', po, pa),
        )

        s.populations.hospitalized = SubCompPopulations(
            uv=self._updated_hospitalized('uv', po, pa),
            fv=self._updated_hospitalized('fv', po, pa),
            b=self._updated_hospitalized('b', po, pa),
        )

        s.populations.recovered = SubCompPopulations(
            uv=int(
                po.recovered.uv
                - (pa.beta * po.recovered.uv * i)
                + (pa.e_r.uv * po.exposed.uv)
                + (pa.i_r.uv * po.infected.uv)
                + (pa.h_r.uv * po.hospitalized.uv)
                - (pa.vfv * po.recovered.uv)
            ),
            fv=int(
                po.recovered.fv
                - (pa.beta * po.recovered.fv * i)
                + (pa.e_r.fv * po.exposed.fv)
                + (pa.i_r.fv * po.infected.fv)
                + (pa.h_r.fv * po.hospitalized.fv)
                + (pa.vfv * po.recovered.uv)
                - (pa.vb * po.recovered.fv)
            ),
            b=int(
                po.recovered.b
                - (pa.beta * po.recovered.b * i)
                + (pa.e_r.b * po.exposed.b)
                + (pa.i_r.b * po.infected.b)
                + (pa.h_r.b * po.hospitalized.b)
                + (pa.vb * po.recovered.fv)
            )
        )

        s.populations.deceased = SubCompPopulations(
            uv=self._updated_deceased('uv', po, pa),
            fv=self._updated_deceased('fv', po, pa),
            b=self._updated_deceased('b', po, pa),
        )

        return s

    @staticmethod
    def _updated_infected(comp, po, pa):
        return int(
            po.infected[comp]
            + (pa.e_i[comp] * po.exposed[comp])
            + (pa.e2_i[comp] * po.exposed[comp])
            - (pa.i_h[comp] * po.infected[comp])
            - (pa.i_r[comp] * po.infected[comp])
            - (pa.i_d[comp] * po.infected[comp])
            # AD: Why no vaccination conversions?
        )

    @staticmethod
    def _updated_hospitalized(comp, po, pa):
        return int(
            po.hospitalized[comp]
            + (pa.i_h[comp] * po.infected[comp])
            - (pa.h_r[comp] * po.hospitalized[comp])
            - (pa.h_d[comp] * po.hospitalized[comp])
            # AD: Why no vaccination conversions?
        )

    @staticmethod
    def _updated_deceased(comp, po, pa):
        return int(
            po.deceased[comp]
            + (pa.i_d[comp] * po.infected[comp])
            + (pa.h_d[comp] * po.hospitalized[comp])
        )


class PopulationTransitionsOld:
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
                prob=getattr(self.s.params, prob),
            )

        return self.s

    def vax(self):
        from dataclasses import fields
        for c in fields(Populations):
            population = getattr(self.s.populations, c.name)

            # uv to fv
            for i in range(population.uv):
                if random() < self.s.params.vfv:
                    population.uv -= 1
                    population.fv += 1

            # fv to b
            for i in range(population.fv):
                if random() < self.s.params.vb:
                    population.fv -= 1
                    population.b += 1

    @staticmethod
    def transition_compartment(
            population_from: SubCompPopulations,
            population_to: SubCompPopulations,
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
