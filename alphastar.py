import numpy as np

NODE_STATE_INDEX = 0
NODE_G_INDEX = 1
NODE_H_INDEX = 2
NODE_PARENT_INDEX = 3
NODE_ACTION_INDEX = 4


class OpenSet:

    def __init__(self, threshold):
        self.diff_threshold = np.array(threshold)
        self.points = np.empty([0, threshold.shape[0]])
        self.nodes = []

    def append(self, node):
        self.nodes.append(node)
        self.points = np.append(self.points, np.atleast_2d(node[NODE_STATE_INDEX]), axis=0)

    def contains(self, node):
        diffs = np.abs(self.points - node[NODE_STATE_INDEX])
        comps = diffs < self.diff_threshold
        if np.any(np.logical_and.reduce(comps, axis=1)):
            return True
        return False

    def pop(self, i):
        node = self.nodes.pop(i)
        np.delete(self.points, i)
        return node

    def __len__(self):
        return len(self.nodes)


class CloseSet:

    def __init__(self, threshold):
        self.diff_threshold = np.array(threshold)

        self.points = np.empty([0, threshold.shape[0]])
        self.nodes = []

    def append(self, node):
        self.nodes.append(node)
        self.points = np.append(self.points, np.atleast_2d(node[NODE_STATE_INDEX]), axis=0)

    def contains(self, node):
        diffs = np.abs(self.points - node[NODE_STATE_INDEX])
        comps = diffs < self.diff_threshold
        if np.any(np.logical_and.reduce(comps, axis=1)):
            return True
        return False


def solve(start_state, goal_state, h_func, is_end_state_func, next_actions_func, next_states_func,
          state_similarity, g_func=lambda x, y: np.linalg.norm(x-y), max_iters=1000):
    def node(state, parent, action=None):  # !
        # node tuple: state, g, h, parent, action
        if parent is not None:
            p_state, p_g, p_h, p_parent, p_action = parent
        else:
            p_state, p_g, p_h, p_parent = state, 0., 0., None
        return state, g_func(state, p_state) + p_g, h_func(state, goal_state), parent, action

    state_len = len(start_state)

    assert state_len == len(goal_state)

    assert state_similarity
    if isinstance(state_similarity, float) or isinstance(state_similarity, int):
        state_similarity = np.ones(state_len)*state_similarity

    start = node(start_state, None)
    goal = node(goal_state, None)

    open_set = OpenSet(state_similarity)
    close_set = CloseSet(state_similarity)

    open_set.append(start)

    iters = 0

    num_nodes_searched = 0

    while len(open_set) > 0 and iters < max_iters:
        fs = [node[NODE_STATE_INDEX] + node[NODE_H_INDEX] for node in open_set.nodes]  # !
        lowest_f_index = np.argmin(fs)

        current = open_set.pop(lowest_f_index)
        num_nodes_searched += 1

        close_set.append(current)
        if is_end_state_func(current[NODE_STATE_INDEX]):
            return current, num_nodes_searched, open_set.nodes, close_set.nodes

        next_actions = next_actions_func(current[NODE_STATE_INDEX])
        next_states = next_states_func(current[NODE_STATE_INDEX], next_actions)
        for state, action in zip(next_states, next_actions):

            successor = node(state, current, action)
            if close_set.contains(successor):
                continue

            if not open_set.contains(successor):  # !
                open_set.append(successor)

        iters += 1

    return start, num_nodes_searched, open_set.nodes, close_set.nodes


def path(result):
    states, actions = [], []
    current = result
    while current is not None:
        states.append(current[NODE_STATE_INDEX])
        actions.append(current[NODE_ACTION_INDEX])

        current = current[NODE_PARENT_INDEX]

    states.reverse()
    actions.reverse()
    return states, actions
