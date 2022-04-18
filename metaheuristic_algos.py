from cmath import inf
import numpy as np

def metaheuristic(datapoints, algo, F, num_clusters):
    pass

def generate_random_sols(D, low, up, n):
    ''' Randomly create a set of solutions
    Inputs
    D: int - dimensionality of solution space
    low: float - lower bound of solution space
    up: float - upper bound of solution space
    n: int - number of solutions to generate

    Outputs
    output: nxD array - solution vector
    
    '''
    output = np.zeros((n,D))
    for i in range(n):
        output[i,:] = np.random.uniform(low,up,size=(1,D))
    return output

def create_first_point(D, low, up):
    ''' Randomly create a solution vector
    Inputs
    D: int - dimensionality of solution space
    low: float - lower bound of solution space
    up: float - upper bound of solution space

    Outputs
    output: 1xD array - solution vector
    
    '''
    # Creates a random vector of dimensionality D with uniform distribution
    # bounded by low and up values.
    # I use uniform distribution because every point in the solution space
    # should have the same probability of being picked.
    output = np.random.uniform(low,up,size=(1,D))
    return output

def simulated_annealing(f,lower_bound,upper_bound,D,N,To,beta,step):

    ''' Simulated Annealing
    Inputs
    f: funct - Cost function to be optimized
    lower_bound: lower bound of the solution space
    upper_bound: upper bound of the solution space
    D: int - dimensionality of solution space
    N: number of iterations
    To: initial temperature
    beta: cooling schedule
    step: float - step size for neighbors

    Outputs
    bestsol: vector - best solution vector in the solution space
    '''

    ### create an initial solution
    currentsol = create_first_point(D,lower_bound,upper_bound)
    bestsol, bestfvalue = currentsol, f(currentsol)

    for t in range(N):
        ### Get temperature
        T = To*(beta**t)

        ### Perform random walk
        epsilon = np.random.normal(loc=0,scale=step,size=currentsol.shape)
        candidatesol = currentsol + epsilon
        # make sure the candidate does not go over bounds
        while candidatesol[0][0]<lower_bound or candidatesol[0][0]>upper_bound or \
            candidatesol[0][1]<lower_bound or candidatesol[0][1]>upper_bound:
            # continue replacing the generated candiate until it meets the
            # solution space bound criteria
            epsilon = np.random.normal(loc=0,scale=step,size=currentsol.shape)
            candidatesol = currentsol + epsilon

        ### Update current solution
        diff = f(candidatesol) - f(currentsol)
        if diff<0: # it improved
            currentsol = candidatesol
        else: # it didn't improve
            r = np.random.uniform(low=0,high=1)
            p = np.exp(-diff/T)
            if p>r: # accept bad solution
                currentsol = candidatesol

        ### Update best solution found until now
        if f(currentsol)<bestfvalue:
            bestsol = currentsol
            bestfvalue = f(currentsol)

    return bestsol

def differential_evolution(f,low,up,D,N,F,Cr,n):
    ''' Differential Evolution
    Inputs
    f: funct - Cost function to be optimized
    low: lower bound of the solution space
    up: upper bound of the solution space
    D: int - dimensionality of solution space
    N: number of iterations
    F: Differential weight
    Cr: Crossover probability
    n: int - solution population size

    Outputs
    bestsol: vector - best solution vector in the solution space
    '''
    
    # Initialize the population x with randomly generated solutions
    X = generate_random_sols(D,low,up,n)

    t=0
    while t<N:
        for i in range(n): # For each solution
            # Randomly choose 3 distinct vectors (each is 1xD)
            rand_index = np.random.choice(n,size=3,replace=False)
            xp = X[[rand_index[0]]]
            xq = X[[rand_index[1]]]
            xr = X[[rand_index[2]]]
            # Generate donor vector
            v = xp + F*(xq-xr)
            v = np.clip(v,low,up) # make sure it stay in bounds
            # Generate random index
            Jr = np.random.choice(D)
            # Generate random distributed number r
            r = np.random.uniform(0,1)
            u = np.zeros((1,D))
            for j in range(D):
                if r <= Cr or j==Jr:
                    u[0,j] = v[0,j]
                else:
                    u[0,j] = X[i,j]
            # Select and update the solution
            if f(u) < f(X[[i]]):
                X[i] = u[0]
        t = t +1
    # Return the best solution in the population
    bestf = np.inf
    for i in range(n):
        if f(X[[i]])<bestf:
            bestf = f(X[[i]])
            bestsol = X[[i]]
    return bestsol


def PSO(f,low,up,D,N,alpha,beta,n):
    ''' Particle Swarm Optimization (PSO)
    Inputs
    f: funct - Cost function to be optimized
    low: lower bound of the solution space
    up: upper bound of the solution space
    D: int - dimensionality of solution space
    N: number of iterations
    alpha: acceleration constant for social component
    beta: acceleration constant for cognitive component
    n: int - solution population size

    Outputs
    bestsol: vector - best solution vector in the solution space
    '''

    # Generate particles (random initial solutions) 
    X = generate_random_sols(D,low,up,n)
    # Generate intial velocities as zeros
    V = np.zeros(X.shape)
    # Find best current global solution
    bestf = np.inf
    for i in range(n):
        if f(X[[i]])<bestf:
            bestf = f(X[[i]])
            g = X[[i]]
    # Initialize current best for each particle 
    # as their own position (X current best: Xcb)
    Xcb = np.array(X)

    t=0
    while t<N:
        for i in range(n): # For each particle
            # Generate noise vectors
            e1 = np.random.rand(D)
            e2 = np.random.rand(D)
            # Generate new velocities
            social = alpha*e1*(g[0]-X[i])
            cognitive = beta*e2*(Xcb[i] -X[i])
            V[i] = V[i] + social + cognitive
            V[i] = np.clip(V[i],low/20,up/20) # clip velocity (small steps)
            # Calculate new locations
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i],low,up) # make sure it stay in bounds
            # Find the current best of the particle
            if f(X[[i]]) < f(Xcb[[i]]):
                Xcb[i] = X[i]
        # Find best current global solution
        bestf = np.inf
        for i in range(n):
            if f(X[[i]])<bestf:
                bestf = f(X[[i]])
                g = X[[i]]
        t = t +1
    # Return the best solution in the population
    bestf = np.inf
    for i in range(n):
        if f(X[[i]])<bestf:
            bestf = f(X[[i]])
            bestsol = X[[i]]
    return bestsol
