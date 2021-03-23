from copy import deepcopy
from enum import Enum

HEALTH = "HEALTH"
POSITION = "POSITION"
MMSTATE = "MMSTATE"
ARROWS = "ARROWS"
ACTIONS = "ACTIONS"
MATERIALS = "MATERIALS"


class Health(Enum):
    H_0, H_25, H_50, H_75, H_100 = range(5)


class Materials(Enum):
    M_0, M_1, M_2 = range(3)


class Positions(Enum):
    W, N, E, S, C = range(5)


class MMState(Enum):
    D, R = range(2)


class Arrows(Enum):
    A_0, A_1, A_2, A_3 = range(4)


X = 5
arr = [1 / 2, 1, 2]
Y = arr[X % 3]
STEP_COST = -10 / Y
GAMMA = 0.999
ERROR = 0.001


class Actions(Enum):
    UP, LEFT, DOWN, RIGHT, STAY, SHOOT, HIT, CRAFT, GATHER, NONE = range(10)


class State:
    def __init__(self, value, health, arrows, materials, mm_state, position):
        self.value = value
        self.actions = []
        self.health = Health(health)
        self.arrows = Arrows(arrows)
        self.materials = Materials(materials)
        self.mm_state = MMState(mm_state)
        self.pos: Positions = Positions(position)

    def __str__(self):
        # return f"Pos: {self.pos.name}  Mat: {self.materials.value} Arrow: {self.arrows.value} MM: {self.mm_state.name} MM_health: {self.health.value} Value = " + "{:0.2f}".format(
        #     self.value)
        return f"({self.pos.name},{self.materials.value},{self.arrows.value},{self.mm_state.name},{self.health.value * 25})"

    def get_info(self):
        return {
            POSITION: self.pos,
            MATERIALS: self.materials,
            ARROWS: self.arrows,
            MMSTATE: self.mm_state,
            HEALTH: self.health,
        }


class Action:
    def __init__(self, action_type: Actions):
        self.type = action_type


class ValueIteration:
    def __init__(self):
        self.states: [State] = []
        self.discount_factor: float = 0.2
        self.iteration: int = 0

    def iterate(self):
        self.iteration += 1
        print(f"Iteration : {self.iteration} " + "-" * 50)
        new_states = []
        for state in deepcopy(self.states):
            action_values = [self.action_value(action, state) for action in state.actions]
            state.value = max(action_values)
            # print(state, ":", state.actions[action_values.index(state.value)].type.name,
            #       "[{:0.2f}]".format(state.value),
            #       end="\n")
            new_states.append(state)
        stop = True
        total_diff = 0
        for idx in range(len(new_states)):
            diff = self.states[idx].value - new_states[idx].value
            if abs(diff) > ERROR:
                stop = False
            total_diff += abs(diff)
        print(total_diff)
        if stop:
            return -1
        self.states = new_states
        return 0

    def action_value(self, action: Action, state: State):
        results = []
        # result[0] is unsuccessful state, result[1:] are successful
        new_state_info = state.get_info()
        if action.type == Actions.NONE:
            return state.value
        if state.pos == Positions.C:
            # failed case for movements
            new_state_info[POSITION] = Positions.E
            results.append((0.15, deepcopy(new_state_info)))
            if action.type == Actions.UP:
                new_state_info[POSITION] = Positions.N
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.DOWN:
                new_state_info[POSITION] = Positions.S
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.LEFT:
                new_state_info[POSITION] = Positions.W
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.RIGHT:
                new_state_info[POSITION] = Positions.E
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.STAY:
                new_state_info[POSITION] = Positions.C
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.SHOOT:
                results = []
                new_state_info[ARROWS] = Arrows(max(0, new_state_info[ARROWS].value - 1))
                results.append((0.5, deepcopy(new_state_info)))
                results.append((0.5, deepcopy(new_state_info)))
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 1))
            elif action.type == Actions.HIT:
                results = [(0.9, deepcopy(new_state_info))]
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 2))
                results.append((0.1, deepcopy(new_state_info)))

        elif state.pos == Positions.N:
            if action.type == Actions.DOWN:
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                new_state_info[POSITION] = Positions.C
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.STAY:
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                new_state_info[POSITION] = Positions.N
                results.append((0.85, deepcopy(new_state_info)))

            elif action.type == Actions.CRAFT:
                if new_state_info[MATERIALS].value > 0:
                    results.append((0, deepcopy(new_state_info)))
                    new_state_info[ARROWS] = Arrows(min(new_state_info[ARROWS].value + 1, len(Arrows) - 1))
                    results.append((0.5, deepcopy(new_state_info)))
                    new_state_info = state.get_info()
                    new_state_info[ARROWS] = Arrows(min(new_state_info[ARROWS].value + 2, len(Arrows) - 1))
                    results.append((0.35, deepcopy(new_state_info)))
                    new_state_info = state.get_info()
                    new_state_info[ARROWS] = Arrows(min(new_state_info[ARROWS].value + 3, len(Arrows) - 1))
                    results.append((0.15, deepcopy(new_state_info)))
                else:
                    results.append((1.0, deepcopy(new_state_info)))

        elif state.pos == Positions.S:
            if action.type == Actions.UP:
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                new_state_info[POSITION] = Positions.C
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.STAY:
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                new_state_info[POSITION] = Positions.S
                results.append((0.85, deepcopy(new_state_info)))
            elif action.type == Actions.GATHER:
                results.append((0.25, deepcopy(new_state_info)))
                new_state_info[MATERIALS] = Materials(min(new_state_info[MATERIALS].value + 1, len(Materials) - 1))
                results.append((0.75, deepcopy(new_state_info)))

        elif state.pos == Positions.E:

            if action.type == Actions.LEFT:
                new_state_info[POSITION] = Positions.C
                results.append((1.0, deepcopy(new_state_info)))
            elif action.type == Actions.STAY:
                new_state_info[POSITION] = Positions.E
                results.append((1.0, deepcopy(new_state_info)))
            elif action.type == Actions.SHOOT:
                results = []
                new_state_info[ARROWS] = Arrows(max(0, new_state_info[ARROWS].value - 1))
                results.append((0.1, deepcopy(new_state_info)))
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 1))
                results.append((0.9, deepcopy(new_state_info)))
            elif action.type == Actions.HIT:
                results.append((0.2, deepcopy(new_state_info)))
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 2))
                results.append((0.8, deepcopy(new_state_info)))

        elif state.pos == Positions.W:

            if action.type == Actions.RIGHT:
                new_state_info[POSITION] = Positions.C
                results.append((1.0, deepcopy(new_state_info)))
            elif action.type == Actions.STAY:
                new_state_info[POSITION] = Positions.W
                results.append((1.0, deepcopy(new_state_info)))
            elif action.type == Actions.SHOOT:
                results = []
                new_state_info[ARROWS] = Arrows(max(0, new_state_info[ARROWS].value - 1))
                results.append((0.75, deepcopy(new_state_info)))
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 1))
                results.append((0.25, deepcopy(new_state_info)))

        final_results = []
        got_hit = -1
        if state.get_info()[MMSTATE] == MMState.D:
            for result in results:
                new_state = result[1]
                new_state[MMSTATE] = MMState.D
                final_results.append((0.8 * result[0], deepcopy(new_state)))
                new_state[MMSTATE] = MMState.R
                final_results.append((0.2 * result[0], deepcopy(new_state)))

        elif state.get_info()[MMSTATE] == MMState.R:
            for result in results:
                final_results.append((0.5 * result[0], result[1]))  # MM just remains ready
            if state.get_info()[POSITION] == Positions.C or state.get_info()[POSITION] == Positions.E:
                # attack and got hit
                un_result = results[0]
                new_state = un_result[1]
                new_state[MMSTATE] = MMState.D
                got_hit = len(final_results)
                final_results.append((0.5, deepcopy(new_state)))
            else:
                for result in results:
                    result_state = deepcopy(result[1])
                    result_state[MMSTATE] = MMState.D
                    final_results.append((0.5 * result[0], deepcopy(result_state)))

        value = 0
        total = 0
        for result in final_results:
            total += result[0]
        # print(total)
        assert (0.99 < total < 1.01)
        for idx, result in enumerate(final_results):
            reward = 0
            if got_hit == idx:
                reward = -30
            # print("value is " + str(self.getvalue(result[1])))
            STEP = STEP_COST
            if action.type == Actions.STAY:
                STEP = 0
            value += result[0] * (STEP + reward + GAMMA * self.getvalue(result[1]))
        return value

    def getvalue(self, result):
        # print(result)
        idx = result[POSITION].value * (len(Materials) * len(Arrows) * len(MMState) * len(Health)) + result[
            MATERIALS].value * (len(Arrows) * len(MMState) * len(Health)) + result[ARROWS].value * len(MMState) * len(
            Health) + result[MMSTATE].value * len(Health) + result[HEALTH].value
        assert(self.states[idx].get_info() == result)
        return self.states[idx].value

    def __str__(self):
        s = ""
        for ste in self.states:
            s += str(ste)
        return s


vi = ValueIteration()
states_init = []
for pos in range(len(Positions)):
    for mat in range(len(Materials)):
        for arrow in range(len(Arrows)):
            for mmst in range(len(MMState)):
                for health in range(len(Health)):
                    state_1 = State(0, health, arrow, mat, mmst, pos)
                    state_1.actions.append(Action(Actions.STAY))
                    if state_1.pos == Positions.C:
                        state_1.actions.append(Action(Actions.DOWN))
                        state_1.actions.append(Action(Actions.UP))
                        state_1.actions.append(Action(Actions.LEFT))
                        state_1.actions.append(Action(Actions.RIGHT))
                        state_1.actions.append(Action(Actions.HIT))
                        state_1.actions.append(Action(Actions.SHOOT))
                    if state_1.pos == Positions.N:
                        state_1.actions.append(Action(Actions.DOWN))
                        state_1.actions.append(Action(Actions.CRAFT))
                    if state_1.pos == Positions.S:
                        state_1.actions.append(Action(Actions.UP))
                        state_1.actions.append(Action(Actions.GATHER))
                    if state_1.pos == Positions.E:
                        state_1.actions.append(Action(Actions.LEFT))
                        state_1.actions.append(Action(Actions.SHOOT))
                        state_1.actions.append(Action(Actions.HIT))
                    if state_1.pos == Positions.W:
                        state_1.actions.append(Action(Actions.RIGHT))
                        state_1.actions.append(Action(Actions.SHOOT))
                    if state_1.health.value == 0:
                        state_1.actions = [Action(Actions.NONE)]
                        state_1.value = 50
                    states_init.append(state_1)

vi.states = states_init

for _ in range(1000):
    if vi.iterate() == -1:
        break
