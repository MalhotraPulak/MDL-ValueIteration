from copy import deepcopy
from enum import Enum
from typing import List
import random
import json
import sys

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


class ValueIteration:
    def __init__(self):
        self.states: [State] = []
        self.discount_factor: float = GAMMA
        self.iteration: int = -1

    def iterate(self):
        self.iteration += 1
        print(f"iteration={self.iteration}")
        new_states = []
        for state in deepcopy(self.states):
            if debug:
                print("Deciding optimal action for", str(state))
            action_values = [self.action_value(action, state)[0] for action in state.actions]
            state.value = max(action_values)
            state.favoured_action = state.actions[action_values.index(state.value)]
            print(str(state) + ":" + state.favoured_action.name +
                  "=[{:0.3f}]".format(state.value),
                  end="\n")
            new_states.append(state)
        stop = True
        max_diff = 0
        for idx in range(len(new_states)):
            diff = self.states[idx].value - new_states[idx].value
            if abs(diff) > ERROR:
                stop = False
            max_diff = max(max_diff, abs(diff))
        print(max_diff, file=sys.stderr)
        self.states = new_states
        if stop:
            return -1
        return 0

    def action_value(self, action: Actions, state: State):
        results = []
        if debug:
            print(action.name)
        # result[0] is unsuccessful state, result[1:] are successful
        new_state_info = state.get_info()
        if action == Actions.NONE:
            return 50, []
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
                # new_state_info[POSITION] = Positions.W
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

        value: float = 0
        total_prob: float = 0.0
        for result in final_results:
            total_prob += result[0]
        # print(total)
        assert (0.99 < total_prob < 1.01)

        for idx, result in enumerate(final_results):
            reward = 0

            # print("value is " + str(self.getvalue(result[1])))
            STEP = STEP_COST
            # for the other task
            # if action == Actions.STAY:
            #     STEP = 0
            # print(result[1])
            if got_hit == idx:
                reward = -40
                # print(result[1])
                # STEP = -40
            # if result[1][HEALTH].value == 0:
            #     reward = 50
            #     STEP = 0

            if debug:
                print("{:0.4f}".format(
                    result[0]) + f", state={self.getState(result[1])} value={self.getState(result[1]).value}")
            value += result[0] * (STEP + GAMMA * (reward + self.getState(result[1]).value))
        if debug:
            print(value)
        return value, final_results

    @classmethod
    def getIdx(cls, info):
        idx = info[POSITION].value * (len(Materials) * len(Arrows) * len(MMState) * len(Health)) + info[
            MATERIALS].value * (len(Arrows) * len(MMState) * len(Health)) + info[ARROWS].value * len(MMState) * len(
            Health) + info[MMSTATE].value * len(Health) + info[HEALTH].value
        return idx

    def getState(self, info) -> State:
        # print(result)
        idx = self.getIdx(info)
        assert (self.states[idx].get_info() == info)
        return self.states[idx]

    def simulate(self, init_state):
        current_state = self.getState(init_state.get_info())
        print("Now:", current_state, current_state.favoured_action)
        while current_state.health.value != 0:
            optimal_action = current_state.favoured_action
            possible_outcomes = self.action_value(optimal_action, current_state)[1]
            # print(optimal_action, current_state, possible_outcomes)
            actual_outcome = random.random()
            total_prob = 0
            print("Possible outcomes")
            for out in possible_outcomes:
                print("{:0.3f}".format(out[0]), self.getState(out[1]))
            for idx, outcome in enumerate(possible_outcomes):
                total_prob += outcome[0]
                if total_prob > actual_outcome:
                    current_state = self.getState(outcome[1])
                    print("Selected Outcome:", idx, "Rolled", "{:0.3f}".format(actual_outcome))
                    print(current_state, current_state.favoured_action)
                    break

    def train(self, max_iter):
        while self.iterate() != -1 and self.iteration < max_iter - 1:
            pass
        print(f"iteration={self.iteration}", file=sys.stderr)
        self.dump_states()

    def dump_states(self):
        with open("trained_states.txt", "w") as f:
            sts = []
            for idx, state in enumerate(self.states):
                sts.append({"id": idx, "action": state.favoured_action.value, "value": state.value})
            json.dump(sts, f)

    def load_states(self):
        with open("trained_states.txt", "r") as f:
            sts = json.load(f)
            for a_state in sts:
                ste = self.states[a_state["id"]]
                ste.value = a_state["value"]
                ste.favoured_action = Actions(a_state["action"])

    def __str__(self):
        s = ""
        for ste in self.states:
            s += str(ste)
        return s

    def do(self):
        for state in self.states:
            if state.pos == Positions.W and state.mm_state == MMState.R:
                print(state, state.value, state.favoured_action)
        action_array = {}
        for action in Actions:
            action_array[action.name] = 0
        for state in self.states:
            action_array[state.favoured_action.name] += 1
        print(action_array)


if __name__ == "__main__":
    debug = False
    if len(sys.argv) == 2 and sys.argv[1] == "d":
        debug = True

    X = 0  # TODO change this for final_results
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
                        if state_1.pos == Positions.N:
                            state_1.actions.append(Actions.DOWN)
                            if mat > 0:
                                state_1.actions.append(Actions.CRAFT)
                        if state_1.pos == Positions.S:
                            state_1.actions.append(Actions.UP)
                            state_1.actions.append(Actions.GATHER)
                        if state_1.pos == Positions.E:
                            state_1.actions.append(Actions.LEFT)
                            if arrow > 0:
                                state_1.actions.append(Actions.SHOOT)
                            state_1.actions.append(Actions.HIT)
                        if state_1.pos == Positions.W:
                            state_1.actions.append(Actions.RIGHT)
                        state_1.actions.append(Actions.STAY)
                        if state_1.pos == Positions.W:
                            if arrow > 0:
                                state_1.actions.append(Actions.SHOOT)

                        if state_1.health.value == 0:
                            state_1.actions = [Actions.NONE]
                            state_1.value = 0
                        states_init.append(state_1)
    vi = ValueIteration()
    vi.states = states_init
    vi.train(500)
    # vi.load_states()
    # vi.do()
    # vi.load_states()
    #
    # total: List[float] = [0, 0, 0, 0, 0]
    # for st in vi.states:
    #     total[st.pos.value] += st.value
    #
    # print("Total values by position")
    # for i in range(len(Positions)):
    #     print(Positions(i))
    # print(total)
    #
    # initial_state = State(value=0, position=Positions.W.value, materials=0, arrows=0, mm_state=MMState.D.value,
    #                       health=Health.H_100.value)
    # initial_state = State(value=0, position=Positions.C.value, materials=2, arrows=0, mm_state=MMState.R.value,
    #                       health=Health.H_100.value)
    # vi.simulate(initial_state)
