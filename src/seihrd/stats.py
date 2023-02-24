class Stats:
    def __init__(self):
        self.states = []

    def __call__(self, state):
        self.states.append(state)

