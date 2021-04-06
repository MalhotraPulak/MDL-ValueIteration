from copy import deepcopy
from enum import Enum
from typing import List
import json
import sys
import numpy as np
import cvxpy as cp

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


debug = False
if len(sys.argv) == 2 and sys.argv[1] == "d":
    debug = True


class Actions(Enum):
    UP, LEFT, DOWN, RIGHT, STAY, SHOOT, HIT, CRAFT, GATHER, NONE = range(10)


class State:
    def __init__(self, value, health, arrows, materials, mm_state, position):
        self.value: float = value
        self.actions: List[Actions] = []
        self.health: Health = Health(health)
        self.arrows: Arrows = Arrows(arrows)
        self.materials: Materials = Materials(materials)
        self.mm_state: MMState = MMState(mm_state)
        self.pos: Positions = Positions(position)
        self.favoured_action: Actions = Actions.NONE

    def __str__(self):
        return f"( {self.pos.name} , {self.materials.value} , {self.arrows.value} , {self.mm_state.name} , {self.health.value} )"

    def get_tuple(self):
        return self.pos.name, self.materials.value, self.arrows.value, self.mm_state.name, self.health.value * 25

    def get_info(self):
        return {
            POSITION: self.pos,
            MATERIALS: self.materials,
            ARROWS: self.arrows,
            MMSTATE: self.mm_state,
            HEALTH: self.health,
        }

    def get_number(self):
        idx = self.pos.value * (len(Materials) * len(Arrows) * len(MMState) * len(Health)) + self.materials.value * (
                len(Arrows) * len(MMState) * len(Health)) + self.arrows.value * len(MMState) * len(
            Health) + self.mm_state.value * len(Health) + self.health.value
        return idx

    def filter(self):
        self.actions = [action for action in self.actions if self.filter_action(action)]

    def filter_action(self, action: Actions):
        if action == Actions.SHOOT:
            return self.arrows.value > 0
        elif action == Actions.CRAFT:
            return self.materials.value > 0
        elif action == Actions.NONE:
            return self.health == Health.H_0
        else:
            return True


class LPP:
    def __init__(self, states):
        self.states: [State] = states
        self.discount_factor: float = GAMMA
        self.iteration: int = -1
        self.dim = sum([len(st.actions) for st in self.states])
        print("Dim is", self.dim)
        self.num_states = len(self.states)
        self.r = None
        self.a = None
        self.alpha = None
        self.solution = None
        self.x = None
        self.policy = None
        self.initialize_r()
        self.initialize_a()
        self.initialize_alpha()
        self.run_LP()
        self.get_solution()
        self.make_dict()

    def initialize_r(self):
        r = np.zeros((1, self.dim))
        count = 0
        for i, state in enumerate(self.states):
            for action in state.actions:
                got_hit, results = self.action_value(action, state)
                for idx, (pr, st) in enumerate(results):
                    # action -> results
                    if action != Actions.NONE:
                        if idx == got_hit:
                            r[0][count] += pr * (-40)
                        r[0][count] += pr * STEP_COST
                    else:
                        r[0][count] += 0
                count += 1
        self.r = r

    def initialize_a(self):
        a = np.zeros((self.num_states, self.dim))
        action_no = 0
        for state_no, cur_state in enumerate(self.states):
            for action in cur_state.actions:
                print(cur_state, action.name)
                got_hit, results = self.action_value(action, cur_state)
                if len(results) <= 0:
                    print(cur_state, action)
                assert len(results) > 0
                for idx, (pr, next_state) in enumerate(results):
                    a[state_no][action_no] += pr  # outflow
                    if action != Actions.NONE:
                        a[self.getState(next_state).get_number()][action_no] -= pr  # inflow
                action_no += 1
                assert state_no == cur_state.get_number()
        self.a = a

    def initialize_alpha(self):
        # starting probability is equal
        alpha = np.zeros((1, self.num_states))
        start_state = State(materials=Materials.M_2, arrows=Arrows.A_3, mm_state=MMState.R,
                            health=Health.H_100, value=0, position=Positions.C)
        print("Start state: ", start_state, start_state.get_number())
        alpha[0][start_state.get_number()] = 1
        self.alpha = alpha.T

    def run_LP(self):
        x = cp.Variable((self.dim, 1), 'x')
        print(x.shape, self.a.shape, self.alpha.shape, self.r.shape)
        constraints = [
            cp.matmul(self.a, x) == self.alpha,
            x >= 0
        ]

        objective = cp.Maximize(cp.matmul(self.r, x))
        problem = cp.Problem(objective, constraints)

        solution = problem.solve(verbose=True)
        self.solution = solution
        self.x = x

    def get_solution(self):
        xs = self.x.value.tolist()
        count = 0
        # xs = [1.2, 1.1, 1.4]
        self.policy = []
        for state in self.states:
            action_len = len(state.actions)
            options = xs[count: count + action_len]
            max_arg = np.argmax(np.array(options))
            state.favoured_action = state.actions[max_arg]
            self.policy.append([state.get_tuple(), state.favoured_action.name])
            count += action_len

    def make_dict(self):
        d = {
            "a": self.a.tolist(),
            "r": self.r.tolist(),
            "x": self.x.value.tolist(),
            "alpha": self.alpha.tolist(),
            "policy": self.policy,
            "objective": self.solution
        }
        with open("outputs/part_3_output.json", "w") as f:
            json.dump(d, f, indent=2)

    def action_value(self, action: Actions, state: State):
        results = []
        if debug:
            print(action.name)
        # result[0] is unsuccessful state, result[1:] are successful
        new_state_info = state.get_info()
        if action == Actions.NONE:
            return -1, [(1, deepcopy(new_state_info))]
        if state.pos == Positions.C:
            if action == Actions.UP:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.N
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.DOWN:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.S
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.LEFT:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.W
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.RIGHT:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.E
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.STAY:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.C
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.SHOOT:
                if state.get_info()[ARROWS].value > 0:
                    # unsuccessful
                    new_state_info[ARROWS] = Arrows(new_state_info[ARROWS].value - 1)
                    results.append((0.5, deepcopy(new_state_info)))
                    # successful
                    new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 1))
                    results.append((0.5, deepcopy(new_state_info)))
                else:
                    assert False
                    # unsuccessful
            elif action == Actions.HIT:
                # unsuccessful
                results.append((0.9, deepcopy(new_state_info)))
                # successful
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 2))
                results.append((0.1, deepcopy(new_state_info)))

        elif state.pos == Positions.N:
            if action == Actions.DOWN:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.C
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.STAY:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.N
                results.append((0.85, deepcopy(new_state_info)))

            elif action == Actions.CRAFT:
                if new_state_info[MATERIALS].value > 0:
                    new_state_info[MATERIALS] = Materials(new_state_info[MATERIALS].value - 1)
                    new_state_info[ARROWS] = Arrows(min(new_state_info[ARROWS].value + 1, len(Arrows) - 1))
                    results.append((0.5, deepcopy(new_state_info)))
                    new_state_info = state.get_info()
                    new_state_info[MATERIALS] = Materials(new_state_info[MATERIALS].value - 1)
                    new_state_info[ARROWS] = Arrows(min(new_state_info[ARROWS].value + 2, len(Arrows) - 1))
                    results.append((0.35, deepcopy(new_state_info)))
                    new_state_info = state.get_info()
                    new_state_info[MATERIALS] = Materials(new_state_info[MATERIALS].value - 1)
                    new_state_info[ARROWS] = Arrows(min(new_state_info[ARROWS].value + 3, len(Arrows) - 1))
                    results.append((0.15, deepcopy(new_state_info)))
                else:
                    assert False
                    # unsuccessful

        elif state.pos == Positions.S:
            if action == Actions.UP:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.C
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.STAY:
                # unsuccessful
                new_state_info[POSITION] = Positions.E
                results.append((0.15, deepcopy(new_state_info)))
                # successful
                new_state_info[POSITION] = Positions.S
                results.append((0.85, deepcopy(new_state_info)))
            elif action == Actions.GATHER:
                # this might be wrong
                # unsuccessful
                results.append((0.25, deepcopy(new_state_info)))
                # successful
                new_state_info[MATERIALS] = Materials(min(new_state_info[MATERIALS].value + 1, len(Materials) - 1))
                results.append((0.75, deepcopy(new_state_info)))

        elif state.pos == Positions.E:

            if action == Actions.LEFT:
                # task 2 1
                new_state_info[POSITION] = Positions.C
                results.append((1.0, deepcopy(new_state_info)))
            elif action == Actions.STAY:
                new_state_info[POSITION] = Positions.E
                results.append((1.0, deepcopy(new_state_info)))
            elif action == Actions.SHOOT:
                results = []
                if state.get_info()[ARROWS].value > 0:
                    new_state_info[ARROWS] = Arrows(new_state_info[ARROWS].value - 1)
                    results.append((0.1, deepcopy(new_state_info)))
                    # successful
                    new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 1))
                    results.append((0.9, deepcopy(new_state_info)))
                else:
                    assert False
            elif action == Actions.HIT:
                # unsuccessful
                results.append((0.8, deepcopy(new_state_info)))  # miss with high prob
                # successful
                new_state_info[HEALTH] = Health(max(0, new_state_info[HEALTH].value - 2))
                results.append((0.2, deepcopy(new_state_info)))  # hit

        elif state.pos == Positions.W:
            if action == Actions.RIGHT:
                # successful
                new_state_info[POSITION] = Positions.C
                results.append((1.0, deepcopy(new_state_info)))
            elif action == Actions.STAY:
                # successful
                new_state_info[POSITION] = Positions.W
                results.append((1.0, deepcopy(new_state_info)))
            elif action == Actions.SHOOT:
                results = []
                if state.get_info()[ARROWS].value > 0:
                    # unsuccessful
                    new_state_info[ARROWS] = Arrows(new_state_info[ARROWS].value - 1)
                    results.append((0.75, deepcopy(new_state_info)))
                    # successful
                    new_state_info[HEALTH] = Health(new_state_info[HEALTH].value - 1)
                    results.append((0.25, deepcopy(new_state_info)))
                else:
                    assert False

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

                new_state = deepcopy(state.get_info())  # get the new state for unsuccessful action
                new_state[MMSTATE] = MMState.D  # new MM state is dormant
                new_state[ARROWS] = Arrows.A_0  # new Arrow state is 0
                new_state[HEALTH] = Health(
                    min(new_state[HEALTH].value + 1, len(Health) - 1))  # new health state is + 25
                got_hit = len(final_results)  # index of action where you got hit
                final_results.append((0.5, deepcopy(new_state)))  # this new state has 0.5 probability
            else:
                for result in results:
                    result_state = deepcopy(result[1])
                    result_state[MMSTATE] = MMState.D
                    final_results.append((0.5 * result[0], deepcopy(result_state)))

        final_final_results = []
        for prob, res in final_results:
            if res == state.get_info():
                pass
                # print(action)
                # print(res)
            else:
                final_final_results.append((prob, res))
        assert len(final_final_results) > 0
        return got_hit, final_final_results

    @classmethod
    def getIdx(cls, info):
        idx = info[POSITION].value * (len(Materials) * len(Arrows) * len(MMState) * len(Health)) + info[
            MATERIALS].value * (len(Arrows) * len(MMState) * len(Health)) + info[ARROWS].value * len(MMState) * len(
            Health) + info[MMSTATE].value * len(Health) + info[HEALTH].value
        return idx

    def getState(self, info) -> State:
        idx = self.getIdx(info)
        assert (self.states[idx].get_info() == info)
        return self.states[idx]

    def __str__(self):
        s = ""
        for ste in self.states:
            s += str(ste)
        return s


if __name__ == "__main__":
    debug = False
    if len(sys.argv) == 2 and sys.argv[1] == "d":
        debug = True

    X = 5  # TODO change this for final_results
    arr = [1 / 2, 1, 2]
    Y = arr[X % 3]
    STEP_COST = -10 / Y
    # GAMMA = 0.25
    GAMMA = 0.999
    ERROR = 0.001

    states_init = []
    for pos in range(len(Positions)):
        for mat in range(len(Materials)):
            for arrow in range(len(Arrows)):
                for mmst in range(len(MMState)):
                    for health in range(len(Health)):
                        state_1 = State(0, health, arrow, mat, mmst, pos)
                        if state_1.pos == Positions.C:
                            state_1.actions.append(Actions.UP)
                            state_1.actions.append(Actions.DOWN)
                            state_1.actions.append(Actions.LEFT)
                            state_1.actions.append(Actions.RIGHT)
                            state_1.actions.append(Actions.HIT)
                            if arrow > 0:
                                state_1.actions.append(Actions.SHOOT)
                            state_1.actions.append(Actions.STAY)
                        if state_1.pos == Positions.N:
                            state_1.actions.append(Actions.DOWN)
                            if mat > 0:
                                state_1.actions.append(Actions.CRAFT)
                            state_1.actions.append(Actions.STAY)
                        if state_1.pos == Positions.S:
                            state_1.actions.append(Actions.UP)
                            state_1.actions.append(Actions.GATHER)
                            state_1.actions.append(Actions.STAY)
                        if state_1.pos == Positions.E:
                            state_1.actions.append(Actions.STAY)
                            state_1.actions.append(Actions.LEFT)
                            if arrow > 0:
                                state_1.actions.append(Actions.SHOOT)
                            state_1.actions.append(Actions.HIT)
                        if state_1.pos == Positions.W:
                            state_1.actions.append(Actions.STAY)
                            state_1.actions.append(Actions.RIGHT)
                            if arrow > 0:
                                state_1.actions.append(Actions.SHOOT)
                        if state_1.health.value == 0:
                            state_1.actions = [Actions.NONE]
                            state_1.value = 0
                        state_1.filter()
                        states_init.append(state_1)
    LPP(states_init)
