from typing import List, Dict, Tuple, Literal, Callable
import numpy as np
import numpy.typing as npt
import autograd.numpy as anp
from autograd import grad
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


'''
A State is the abstract concept of circumstance. Intuitively, an agent exists in a state and can travel between states via actions taken. A state may be terminal; in this case, if the agent traverses it, the episode is halted prematurely.
'''
class State:
    '''
    Constructor for objects of type State
    '''
    def __init__(self, isTerminal:bool):
        self._isTerminal = isTerminal

    '''
    Returns true if this State is terminal; false, otherwise
    '''
    def isTerminal(self) -> bool:
        return self._isTerminal

'''
An Action is the abstract concept of behavior. Intuitively, an agent takes an action in an attempt to travel between states and obtain a reward.
'''
class Action:
    '''
    Constructor for objects of type Action
    '''
    def __init__(self, func:Dict[State,State], name:str):
        self._func = func
        self._name = name

    '''
    Returns the resulting State following taking this action in the given State
    '''
    def __call__(self, state:State) -> State:
        return self._func[state]

    '''
    Returns the name of this Action
    '''
    def __str__(self) -> str:
        return self._name

    '''
    Overwrites the destination State after taking this action in the given State
    '''
    def __setitem__(self, source, destination):
        self._func[source] = destination
        
'''
An Environment is the abstract concept of setting. Intuitively, it is a collection of States equipped with rewards and links such that an agent can travel through it.
'''
class Environment:
    '''
    Constructor for objects of type Environment
    '''
    def __init__(self, states:list, rewards:Dict[Tuple[State,Action],float], links:Dict[Tuple[State,Action],State]):
        self._states = states
        self._rewards = rewards
        
        # Overwrite actions to act on non-empty links associated with each cell
        for link in list(links.keys()):
            if not links[(link[0],link[1])] is None: link[1][link[0]] = links[(link[0],link[1])]

    '''
    Returns the reward associated with the given State-Action pair
    '''
    def get_reward(self, state:State, action:Action) -> float:
        return self._rewards[(state,action)]

    '''
    Returns the State at the given index
    '''
    def __getitem__(self, key:int):
        return self._states[key]

    '''
    Iterates over all States in this Environment
    '''
    def __iter__(self):
        for state in self._states:
            yield state

'''
A Policy is an abstract map from states to actions
'''
class Policy:
    '''
    Constructor for objects of type Policy
    '''
    def __init__(self, env:Environment, policy:Callable[State,Action]):
        self._env = env
        self._states = env._states
        self._actions = env._actions
        self._policy = policy

    '''
    Returns the Action given by taking this Policy in the given State
    '''
    def __call__(self, state) -> Action:
        return self._policy(state)


class ValueFunction:
    '''
    Constructor for objects of type ValueFunction
    '''
    def __init__(self, v:Callable[[State, List[float]], float]):
        self._v = v

    '''
    Evaluate this function at the given State-weight pair
    '''
    def __call__(self, s:State, w:List[float]=None):
        if w is None:
            new = lambda w: self._v(s,w)
            return lambda *args: new(list(args))
        return self._v(s,w)

    '''
    Returns the partial derivative of v(s) wrt to the variable at the given index, evaluated at the given weights
    '''
    def partial_derivative(self, s:State, index:int, weights:Tuple[float]) -> float:
        self.gradient(s, weights)[index]

    '''
    Returns the gradient of v(s) evaluated at the given weights
    '''
    def gradient(self, s:State, weights:Tuple[float]) -> npt.ArrayLike:
        f = lambda w: self._v(s,w)
        def wrapper(w):
            return f(w)
        grad_f = grad(wrapper)
        w = anp.array(weights)
        return grad_f(w)
        

'''
A Cell is a State which exists in a Cartesian plane and is given a colour. By default, a Cell is white and non-terminal.
'''
class Cell(State):
    '''
    Constructor for objects of type Cell
    '''
    def __init__(self, x:int, y:int, isTerminal:bool=False, colour:str='white'):
        super().__init__(isTerminal)
        self._colour = colour
        self._x = x
        self._y = y

    '''
    Returns the x coordinate of this cell
    '''
    def getX(self) -> int:
        return self._x

    '''
    Returns the y coordinate of this cell
    '''
    def getY(self) -> int:
        return self._y

    '''
    Returns the colour of this cell
    '''
    def get_colour(self) -> str:
        return self._colour

'''
A Direction is an action which moves an agent north, east, west, or south. A direction must be provided an Environment of cells in a Cartesian plane in order for the concept to make sense.
'''
class Direction(Action):
    def __init__(self, env:Environment, name:Literal['north','east','south','west']):
        self._env = env
        match name:
            case 'north':
                func={state:env[state.getY()-1][state.getX()] if state.getY() != 0 else state for state in env}
            case 'east':
                func={state:env[state.getY()][state.getX()+1] if state.getX() != env._width - 1 else state for state in env}
            case 'west':
                func={state:env[state.getY()][state.getX()-1] if state.getX() != 0 else state for state in env}
            case 'south':
                func={state:env[state.getY()+1][state.getX()] if state.getY() != env._height - 1 else state for state in env}
            case _:
                raise ValueError('Invalid Direction')
        super().__init__(func, name)

class Example1(Environment):    
    def __init__(self):
        self._width = 5
        self._height = 5
        self._states = [[Cell(x=0, y=0, isTerminal=True, colour='black'), Cell(x=1, y=0), Cell(x=2, y=0), Cell(x=3, y=0), Cell(x=4, y=0, isTerminal=True, colour='black')],
               [Cell(x=0, y=1), Cell(x=1, y=1), Cell(x=2, y=1), Cell(x=3, y=1), Cell(x=4, y=1)],
               [Cell(x=0, y=2, colour='red'), Cell(x=1, y=2, colour='red'), Cell(x=2, y=2), Cell(x=3, y=2,colour='red'), Cell(x=4, y=2, colour='red')],
               [Cell(x=0, y=3), Cell(x=1, y=3), Cell(x=2, y=3), Cell(x=3, y=3), Cell(x=4, y=3)],
               [Cell(x=0, y=4, colour='blue'), Cell(x=1, y=4), Cell(x=2, y=4), Cell(x=3, y=4), Cell(x=4, y=4)]]

        self._actions = [Direction(self, 'north'), Direction(self, 'east'), Direction(self, 'west'), Direction(self, 'south')]

        self._rewards = {(state,action):-1 if action(state) == state else
                        -20 if action(state).get_colour() == 'red' else
                        0 if state.get_colour() == 'black' else
                        -1 for action in self._actions for state in self}
        
        self._links={(state,action):self[4][0] if state.get_colour == 'red' else None for state in self for action in self._actions}

        super().__init__(self._states, self._rewards, self._links)

    def __iter__(self):
        for row in self._states:
            for cell in row:
                yield cell

    def __len__(self):
        return 25

class Example2(Environment):
    def __init__(self):
        self._width = 7
        self._height = 7
        self._states = [[Cell(x=0, y=0), Cell(x=1, y=0), Cell(x=2, y=0), Cell(x=3, y=0), Cell(x=4, y=0), Cell(x=5, y=0), Cell(x=6, y=0, isTerminal=True, colour='black')],
               [Cell(x=0, y=1), Cell(x=1, y=1), Cell(x=2, y=1), Cell(x=3, y=1), Cell(x=4, y=1), Cell(x=5, y=1), Cell(x=6, y=1)],
               [Cell(x=0, y=2), Cell(x=1, y=2), Cell(x=2, y=2), Cell(x=3, y=2), Cell(x=4, y=2), Cell(x=5, y=2), Cell(x=6, y=2)],
               [Cell(x=0, y=3), Cell(x=1, y=3), Cell(x=2, y=3), Cell(x=3, y=3), Cell(x=4, y=3), Cell(x=5, y=3), Cell(x=6, y=3)],
               [Cell(x=0, y=4), Cell(x=1, y=4), Cell(x=2, y=4), Cell(x=3, y=4), Cell(x=4, y=4), Cell(x=5, y=4), Cell(x=6, y=4)],
               [Cell(x=0, y=5), Cell(x=1, y=5), Cell(x=2, y=5), Cell(x=3, y=5), Cell(x=4, y=5), Cell(x=5, y=5), Cell(x=6, y=5)],
               [Cell(x=0, y=6, isTerminal=True, colour='black'), Cell(x=1, y=6), Cell(x=2, y=6), Cell(x=3, y=6), Cell(x=4, y=6), Cell(x=5, y=6), Cell(x=6, y=6)]]
    
        self._actions = [Direction(self, 'north'), Direction(self, 'east'), Direction(self, 'west'), Direction(self, 'south')]

        self._rewards = {(state,action):-1 if (state._x==0 and state._y==6) else
                         1 if (state._x==6 and state._y==0) else
                         0 for action in self._actions for state in self}
        
        self._links={}

        super().__init__(self._states, self._rewards, self._links)

    def reset(self):
        return self._states[3][3]

    def __len__(self):
        return 49

    def __iter__(self):
        for row in self._states:
            for cell in row:
                yield cell

class RandomPolicy(Policy):
    def __init__(self):
        env=Example2()
        policy = lambda state: random.choice(env._actions)
        super().__init__(env._states, env._actions, policy)

def sarsa(env:Environment=Example1(), epsilon=lambda t: 1/t, alpha=0.5, gamma=0.9, loops=1000, showResult:bool=False):
    random.seed(0)
    if type(epsilon) in [int, float]:
        temp = epsilon
        epsilon = lambda t: temp
    graph = []

    Q = {(state,action):0 if state.isTerminal() else random.random() for state in env for action in env._actions}

    count = 0
    while count < loops:
        
        count += 1
        S = env[4][0]
        if random.random() < epsilon(count):
            A = random.choice(env._actions)
        else:
            A = max(env._actions, key=lambda action: Q[S,action])
        total = 0
        while not S.isTerminal():
            S_prime = A(S)
            R = env._rewards[(S,A)]
            total += R
            if random.random() < epsilon(count):
                A_prime = random.choice(env._actions)
            else:
                A_prime = max(env._actions, key=lambda action: Q[S_prime,action])
            Q[(S,A)] = Q[(S,A)] + alpha * (R + gamma * Q[(S_prime,A_prime)] - Q[(S,A)])
            S = S_prime
            A = A_prime
        graph.append(total)

    if showResult:
        for n in range(5):print([str(max(env._actions, key=lambda action:Q[state,action])) for state in env][5*n:(5*n)+5])
    return graph
    

def qlearning(env:Environment=Example1(), epsilon=lambda t: 1/t, alpha=0.1, gamma=0.9, loops=1000, showResult:bool=False):
    random.seed(0)
    if type(epsilon) in [int, float]:
        temp = epsilon
        epsilon = lambda t: temp
    graph=[]
    env = Example1()
    Q = {(state,action):0 if state.isTerminal() else 0 for state in env for action in env._actions}
    
    count=0
    while count < loops:
        count += 1
        S = env[4][0]
        total = 0
        while not S.isTerminal():
            if random.random() < epsilon(count):
                A = random.choice(env._actions)
            else:
                A = max(env._actions, key=lambda action: Q[S,action])
            S_prime = A(S)
            R = env._rewards[(S,A)]
            total += R
            Q[(S,A)] = Q[(S,A)] + alpha * (R + gamma * max([Q[S_prime, action] for action in env._actions]) - Q[(S,A)])
            S = S_prime
        graph.append(total)

    if showResult:
        for n in range(5):print([str(max(env._actions, key=lambda action:Q[state,action])) for state in env][5*n:(5*n)+5])   
    return graph
    

def gradient_mc(policy: Policy, value_function: ValueFunction, alpha: float = 0.1, gamma:float=0.9, loops: int = 100):
    random.seed(0)
    w = [0.0] * len(policy._env)
    
    count = 0
    while count < loops:
        count += 1
        
        episode = []
        S = policy._env.reset()
        
        while not S.isTerminal():
            A = policy(S)
            next_state = A(S)
            R = policy._env._rewards[(next_state, A)]
            episode.append((S, A, R))
            S = next_state

        G = 0
        for S, A, R in episode:
            G = gamma * G + R
            value_estimate = value_function(S,w)
            gradient = value_function.gradient(S,w)
            w = [a + alpha * (G - value_estimate) * b for a,b in zip(w, gradient)]
        
    
    for n in range(7):print([w[7*n + i] for i in range(7)])    
            

def semi_gradient_td_0(policy:Policy, value_function:ValueFunction, alpha:float=0.1, gamma:float=0.9, loops:int=100):
    random.seed(0)
    w = [0.0] * len(policy._env)
    
    count = 0
    while count < loops:
        count += 1

        episode = []
        S = policy._env.reset()

        while not S.isTerminal():
            A = policy(S)
            next_state = A(S)
            R = policy._env._rewards[(next_state, A)]
            episode.append((S,A,R))
            S = next_state

        for (S, A, R) in episode:
            S_prime = A(S)
            w = [a + alpha * (R + gamma * value_function(S_prime,w) - value_function(S,w)) * b for a,b in zip(w, value_function.gradient(S,w))]

    for n in range(7):print([round(w[7*n + i],5) for i in range(7)])    

def graph(iterations:int=1000, graph_type:Literal['sarsa','q']="sarsa", loops:int=100):
    random.seed(0)
    if not graph_type in ['sarsa','q']:
        raise ValueError("graph_type must be \'sarsa\' or \'q\'")
    data = []
    graph = []
    for i in range(iterations):
        if graph_type == 'sarsa':
            data.append(sarsa(loops=loops))
        else:
            data.append(qlearning(loops=loops))
    for i in range(len(data[0])):
        graph.append(1/iterations * sum([iteration[i] for iteration in data]) )
        
    if graph_type == 'sarsa':
        plt.plot(graph, color='red', linewidth=2.5)
    else:
        plt.plot(graph, color='green', linewidth=2.5)

    plt.grid(True)
    if graph_type == 'sarsa':
        plt.title("SARSA")
    else:
        plt.title("Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

def display_grid1():
    fig, ax = plt.subplots()

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    for i in range(5):
        for j in range(5):
            if (i == 0 and j == 0) or (i == 0 and j == 4):
                color = 'black'
            elif (i == 2 and j == 0) or (i == 2 and j == 1) or (i == 2 and j == 3) or (i == 2 and j == 4):
                color = 'red'
            elif (i == 4 and j == 0):
                color = 'blue'
            else:
                color = 'white'
        
            rect = patches.Rectangle((j, 4-i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

def display_grid2():
    fig, ax = plt.subplots()

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)

    for i in range(7):
        for j in range(7):
            if (i == 0 and j == 6) or (i == 6 and j == 0):
                color = 'black'
            elif (i == 3 and j == 3):
                color = 'blue'
            else:
                color = 'white'
        
            rect = patches.Rectangle((j, 6-i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

    plt.show()

env = Example2()
policy = Policy(env, policy=lambda s: random.choice(env._actions))
value_function = ValueFunction(v=lambda s,w: w[s.getX() + 7*s.getY()])
gradient_mc(policy, value_function)
print()
semi_gradient_td_0(policy, value_function)
