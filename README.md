# alphastar-search
Python implementation of A* search algorithm

To use this module you have to call `solve` function.

Solve function takes the following arguments:
*  start_state        : Starting point
*  goal_state         : Ending point
*  h_func             : Heuristic function h(current_state, goal_state)
*  next_actions_func  : A function that takes a state and produces all the possible actions from this state
*  is_end_state_func  : A function that takes a state and returns whether this state is end state or not. 
The default argument for this function is a function that checks if the current state is the goal state
*  next_states_func   : A function that produces all the next states given the current state and all the possible actions.
The default argument for this function just adds the actions to the state
*  state_similarity   : A vector describing the maximum absolute distance for each dimension that 
makes two states "equal". The defualt value is a zero vector. If given a scalar value, it will create
a uniform vector with this value.
*  g_func             : Cost function g(state_1, state_2). The default argument for this function is the euclidean
 distance between the two states.
*  max_iters=1000         : An upper limit on the number of iterations

The result of the `solve` function can be passed to the `path` function to generate the whole 
path from start to end, and the corresponding actions of each step.

```python
result, n, open_set, close_set = solve_maze(maze, start)
states, actions = path(result)
```

