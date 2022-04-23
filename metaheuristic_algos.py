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

def distance(firefly1, firefly2):
    """
    Calculates the distance between two fireflies
    :param firefly1: solution vector 1
    :param firefly2: solution vector 2
    :return: distance between firefly1 and firefly2
    """
    return np.linalg.norm(firefly1 - firefly2)


def intensity(beta_0, gamma, distance, firefly1, firefly2):
    """
    Calculates the intensity of a firefly
    :param beta_0: initial intensity
    :param gamma: light absorption coefficient
    :param distance: distance between firefly1 and firefly2
    :param firefly1: solution vector 1
    :param firefly2: solution vector 2
    :return: intensity of firefly 2 wrt firefly 1
    """
    return beta_0 / (1 + gamma * distance(firefly1, firefly2) ** 2)


def find_best_solution(f, population):
    """
    Finds the best solution in a population
    :param f: objective function
    :param population: solution vectors
    :return: the best fitness, the best solution, fitness of population
    """
    population_fitness = f(population)
    best_ind = np.argmax(population_fitness)
    return population_fitness[best_ind], population[best_ind], population_fitness

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

def firefly_optimization(f, D, lower_bound, upper_bound, pop_size, hyperparams, max_iter):
    """
    Implements the Firefly Algorithm.
    :param f: objective function
    :param D: dimension of solution vectors
    :param lower_bound: lower bound of solution vectors
    :param upper_bound: upper bound of solution vectors
    :param pop_size: population size
    :param hyperparams: list containing hyperparameters: gamma, lambda_0, beta_0
    :param max_iter: maximum number of iterations
    :return: best solution found, best fitness
    """
    [gamma, lambda_0, beta_0] = hyperparams
    population = np.clip(np.random.normal(0, np.abs(upper_bound - lower_bound) / 4, (pop_size, D)),
                         lower_bound, upper_bound)
    global_best_fitness, global_best_solution, population_fitness = find_best_solution(f, population)

    idx = 0
    while idx < max_iter:
        for i in range(pop_size):
            for j in range(pop_size):
                j_intensity = intensity(population_fitness[j] * beta_0, gamma, distance, population[i], population[j])
                if population_fitness[i] < j_intensity:
                    population[i] = population[i] + intensity(beta_0, gamma, distance, population[i], population[j]) * \
                                    (population[j] - population[i]) + \
                                    (lambda_0 ** idx) * np.random.normal(0, 1, D)

        population = np.clip(population, lower_bound, upper_bound)
        cur_best_fitness, cur_best_solution, population_fitness = find_best_solution(f, population)
        if cur_best_fitness > global_best_fitness:
            global_best_fitness = cur_best_fitness
            global_best_solution = cur_best_solution

        idx += 1

    return global_best_solution, global_best_fitness


def bat_optimization(f, D, lower_bound, upper_bound, pop_size, hyperparams, max_iter):
    """
    Implements the Bat Algorithm.
    :param f: objective function
    :param D: dimension of solution vectors
    :param lower_bound: lower bound of solution vectors
    :param upper_bound: upper bound of solution vectors
    :param pop_size: population size
    :param hyperparams: list containing hyperparameters: f_max, alpha, gamma
    :param max_iter: maximum number of iterations
    :return: best solution found, best fitness
    """
    [f_max, alpha, gamma] = hyperparams
    f_min = 0
    sigma = 0.01
    rf = 1

    population = np.clip(np.random.normal(0, np.abs(upper_bound - lower_bound) / 4, (pop_size, D)),
                         lower_bound, upper_bound)
    global_best_fitness, global_best_solution, population_fitness = find_best_solution(f, population)
    velocity = np.zeros((pop_size, D))
    loudness = np.ones(pop_size)
    pulse_rate = np.ones(pop_size) * 0.5

    idx = 0
    while idx < max_iter:
        for i in range(pop_size):
            freq_i = f_min + (f_max - f_min) * np.random.uniform(0, 1)
            velocity[i] = velocity[i] + (population[i] - global_best_solution) * freq_i
            temp_bat = population[i] + velocity[i]

            if np.random.uniform(0, 1) > pulse_rate[i]:
                temp_bat = global_best_solution + sigma * np.random.normal(0, 1, D)

            fitness = f(np.expand_dims(temp_bat, 0))
            if fitness > population_fitness[i] and np.random.uniform(0, 1) < loudness[i]:
                population[i] = temp_bat
                population_fitness[i] = fitness

            loudness[i] = alpha * loudness[i]
            pulse_rate[i] = rf * (1 - np.exp(-gamma * idx))

        population = np.clip(population, lower_bound, upper_bound)
        cur_best_fitness, cur_best_solution, population_fitness = find_best_solution(f, population)
        if cur_best_fitness > global_best_fitness:
            global_best_fitness = cur_best_fitness
            global_best_solution = cur_best_solution

        idx += 1

    return global_best_solution, global_best_fitness

