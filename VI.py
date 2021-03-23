import copy


class ValueIteration:
    def __init__(self):
        self.states = []
        self.discount_factor = 0.2
        self.iteration = 0

    def iterate(self):
        self.iteration += 1
        print(f"Iteration : {self.iteration} " + "-" * 50)
        new_states = []
        for state in copy.deepcopy(self.states):
            if len(state.actions) == 0:
                new_states.append(state)
                continue
            print(f"State: {state.name}")
            state.value = max([self.action_value(action) for action in state.actions])
            new_states.append(state)
            print()
        self.states = new_states

    def action_value(self, action):
        val = 0
        print("Action", action.name)
        for i in range(len(action.resultant_states)):
            s_val = action.transition_prob[i] * (action.transition_costs[i] + self.discount_factor * self.get_state(
                action.resultant_states[i]).value)
            print(
                f"{action.transition_prob[i]} * ({action.transition_costs[i]} + {self.discount_factor} * {self.get_state(action.resultant_states[i]).value}) = {s_val}")
            val += s_val
        print(f"Total = {val}")
        return val

    def get_state(self, state_name):
        for state in self.states:
            if state.name == state_name:
                return state
        return None

    def __str__(self):
        s = ""
        for state in self.states:
            s += f"State {state.name} has value {state.value}\n"
        return s


class State:
    def __init__(self, name, value):
        self.value = value
        self.name = name
        self.actions = []


class Action:
    def __init__(self, name, probs, resultant_states, costs):
        self.name = name
        self.transition_costs = costs
        self.resultant_states = resultant_states
        self.transition_prob = probs


vi = ValueIteration()
state_A = State("A", 0)
state_A.actions = [Action("Right", [0.8, 0.2], ["B", "A"], [-1, -1]),
                   Action("Up", [0.8, 0.2], ["C", "A"], [-1, -1])]
state_B = State("B", 0)
state_B.actions = [Action("Left", [0.8, 0.2], ["A", "B"], [-1, -1]),
                   Action("Up", [0.8, 0.2], ["R", "B"], [-4, -1])]
state_C = State("C", 0)
state_C.actions = [Action("Right", [0.25, 0.75], ["R", "C"], [-3, -1]),
                   Action("Down", [0.8, 0.2], ["A", "C"], [-1, -1])]
state_R = State("R", 16.5)

vi.states = [state_A, state_B, state_C, state_R]

print(vi)
for _ in range(5):
    vi.iterate()
    print(vi)

A_R = -1.1708300339200002 + 16.6
print(A_R - 0.7881754375)
print(A_R - 0.7749999206399998)
