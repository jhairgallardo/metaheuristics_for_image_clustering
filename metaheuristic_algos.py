import numpy as np
from scipy.spatial import distance_matrix


def metaheuristic(datapoints, algo, num_clusters):
    ### Define fitness function
    def fitness_function(centroids):
        '''
        Implements Fitness function based on the quantization error
        https://ieeexplore.ieee.org/document/1299577
        :param centroids: set of clusters solution
        Note: We want to minimize this function
        '''
        # Reshape centroids
        pop_size = len(centroids)
        centroids = centroids.reshape(pop_size, num_clusters, datapoints.shape[1])
        dists = np.empty((num_clusters, datapoints.shape[0], pop_size))
        for cluster_idx in range(num_clusters):
            dists[cluster_idx] = distance_matrix(datapoints, centroids[:, cluster_idx, :])

        pixel_labels = np.argmin(dists, axis=0).T
        pixel_distances = np.min(dists, axis=0).T

        cluster_sums = np.empty(pop_size)
        for pop_idx in range(pop_size):
            for cluster_idx in range(num_clusters):
                cluster_idxs = pixel_labels[pop_idx] == cluster_idx
                cur_sum = np.sum(pixel_distances[pop_idx, cluster_idxs]) / len(cluster_idxs)
                if np.isnan(cur_sum):
                    # check to penalize empty clusters
                    cluster_sums[pop_idx] = np.inf
                    break
                cluster_sums[pop_idx] += cur_sum
        fitness = cluster_sums / num_clusters
        return fitness

    ## Parameters for all algorithms
    D = num_clusters * datapoints.shape[1]
    # TODO: double check if this idea for low and up is ok.
    low = np.min(datapoints)
    up = np.max(datapoints)

    ## PSO
    if algo['name'] == 'PSO':
        iter = algo['iter']  # number of iterations
        alpha, beta = algo['alpha'], algo['beta']
        n = algo['pop_size']  # number of particles
        centroids, bestf = PSO(fitness_function, low, up, D, iter, alpha, beta, n)

    elif algo['name'] == 'DE':
        iter = algo['iter']  # number of iterations
        F, Cr = algo['F'], algo['Cr']
        n = algo['pop_size']  # population size
        centroids, bestf = differential_evolution(fitness_function, low, up, D, iter, F, Cr, n)

    elif algo['name'] == 'SA':
        iter = algo['iter']  # number of iterations
        To = algo['init_temp']
        beta = algo['beta']
        centroids, bestf = simulated_annealing(fitness_function, low, up, D, iter, To, beta, step=0.1)

    elif algo['name'] == 'FA':
        iter = algo['iter']  # number of iterations
        gamma = algo['gamma']
        alpha = algo['alpha']
        beta = algo['beta']
        n = algo['pop_size']  # number of fireflies
        centroids, bestf = firefly(fitness_function, low, up, D, iter, gamma, alpha, beta, n, flip_f=True)

    elif algo['name'] == 'BA':
        iter = algo['iter']  # number of iterations
        alpha = algo['alpha']
        gamma = algo['gamma']
        n = algo['pop_size']  # number of bats
        centroids, bestf = bats(fitness_function, low, up, D, iter, alpha, gamma, n)

    else:
        return None, None

    # Reshape centroids
    centroids = centroids.reshape(num_clusters, datapoints.shape[1])
    # Get labels
    SAD = []
    for centroid in centroids:
        SAD.append(np.sum(np.abs(datapoints - centroid), axis=1))
    labels = np.argmin(SAD, axis=0)
    return labels, centroids


def find_best_solution(population, fitness_function, maximize=False):
    fitness = fitness_function(population)
    if maximize:
        best_idx = np.argmax(fitness)
    else:
        best_idx = np.argmin(fitness)
    return fitness, fitness[best_idx], population[best_idx]


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
    return np.random.uniform(low, up, size=(n, D))


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
    return np.random.uniform(low, up, size=(1, D))


def simulated_annealing(f, lower_bound, upper_bound, D, N, To, beta, step):
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
    currentsol = create_first_point(D, lower_bound, upper_bound)
    # if initial solution results on inf fitness function
    # (maybe one cluster doesn't have any datapoints)
    # reset init values until it is not inf
    while f(currentsol) == np.inf:
        currentsol = create_first_point(D, lower_bound, upper_bound)
    bestsol, bestfvalue = currentsol, f(currentsol)

    print('\t best fitness:', bestfvalue)

    for t in range(N):
        ### Get temperature
        T = To * (beta ** t)

        ### Perform random walk
        epsilon = np.random.normal(loc=0, scale=step, size=currentsol.shape)
        candidatesol = currentsol + epsilon
        candidatesol = np.clip(candidatesol, lower_bound, upper_bound)  # make sure it stay in bounds

        ### Update current solution
        diff = f(candidatesol) - f(currentsol)
        if diff < 0:  # it improved
            currentsol = candidatesol
        else:  # it didn't improve
            r = np.random.uniform(low=0, high=1)
            p = np.exp(-diff / T)
            if p > r:  # accept bad solution
                currentsol = candidatesol

        ### Update best solution found until now
        if f(currentsol) < bestfvalue:
            bestsol = currentsol
            bestfvalue = f(currentsol)

        print('\t best fitness:', bestfvalue)

    return bestsol, bestfvalue


def differential_evolution(f, low, up, D, N, F, Cr, n):
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
    X = generate_random_sols(D, low, up, n)
    # Find best current global solution
    bestf = np.inf
    for i in range(n):
        if f(X[[i]]) < bestf:
            bestf = f(X[[i]])
            bestsol = X[[i]]
    print('\t best fitness:', bestf)

    t = 0
    while t < N:
        for i in range(n):  # For each solution
            # Randomly choose 3 distinct vectors (each is 1xD)
            rand_index = np.random.choice(n, size=3, replace=False)
            xp = X[[rand_index[0]]]
            xq = X[[rand_index[1]]]
            xr = X[[rand_index[2]]]
            # Generate donor vector
            v = xp + F * (xq - xr)
            v = np.clip(v, low, up)  # make sure it stay in bounds
            # Generate random index
            Jr = np.random.choice(D)
            u = np.zeros((1, D))
            for j in range(D):
                # Generate random distributed number r
                r = np.random.uniform(0, 1)
                if r <= Cr or j == Jr:
                    u[0, j] = v[0, j]
                else:
                    u[0, j] = X[i, j]
            # Select and update the solution
            if f(u) < f(X[[i]]):
                X[i] = u[0]

        # Find best current global solution
        iter_bestf = np.inf
        for i in range(n):
            if f(X[[i]]) < iter_bestf:
                iter_bestf = f(X[[i]])
                iter_bestsol = X[[i]]

        # If iteration best solution is better than previous ones
        # update
        if iter_bestf < bestf:
            bestf = iter_bestf
            bestsol = iter_bestsol

        print('\t best fitness:', bestf)
        t = t + 1
    return bestsol, bestf


def PSO(f, low, up, D, N, alpha, beta, n):
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
    X = generate_random_sols(D, low, up, n)
    # Generate intial velocities as zeros
    V = np.zeros(X.shape)
    # Find best current global solution
    _, bestf, bestsol = find_best_solution(X, f)
    # Initialize current best for each particle
    # as their own position (X current best: Xcb)
    Xcb = np.array(X)

    print('\t best fitness:', bestf)

    t = 0
    while t < N:
        e1 = np.random.uniform(0, 1, (n, D))
        e2 = np.random.uniform(0, 1, (n, D))

        social = alpha * e1 * (bestsol - X)
        cognitive = beta * e2 * (Xcb - X)
        V = V + social + cognitive
        V = np.clip(V, -up / 5, up / 5)
        X = X + V
        X = np.clip(X, low, up)
        Xcb_idxs = np.argmin(np.stack((f(X), f(Xcb)), axis=1), axis=1)
        Xcb = np.array([X[i] if Xcb_idxs[i] == 0 else Xcb[i] for i in range(n)])  # should vectorize this later
        _, iter_bestf, iter_bestsol = find_best_solution(X, f)
        if iter_bestf < bestf:
            bestf = iter_bestf
            bestsol = iter_bestsol

        print('\t best fitness:', bestf)
        t = t + 1
    return bestsol, bestf


def firefly(func, low, up, D, N, gamma, alpha, beta, n, flip_f=False):
    ''' Firefly Algorithm (FA)
    It expects a maximization problem. For functions with global minima,
    use the flip_f argument as True.

    Inputs
    f: funct - Cost function to be optimized
    low: lower bound of the solution space
    up: upper bound of the solution space
    D: int - dimensionality of solution space
    N: number of iterations
    gamma: light absorbtion coefficient
    alpha: exploration coefficient
    beta: attractiveness
    n: int - solution population size
    flip_f: boolean - Flag to flip minimization problems into maximization

    Outputs
    bestsol: vector - best solution vector in the solution space
    '''
    if flip_f:
        f = lambda var: -1 * func(var)
    else:
        f = func

    # Generate fireflies (random initial solutions)
    X = generate_random_sols(D, low, up, n)
    # Find best current global solution
    fitness, bestf, bestsol = find_best_solution(X, f, maximize=True)
    print('\t best fitness:', bestf)

    t = 0
    while t < N:
        for i in range(n):
            for j in range(n):
                if fitness[i] < fitness[j]:
                    rij = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                    eps = np.random.normal(size=(1, D))
                    # Move xi towards xj
                    X[i] = X[i] + (beta / (1 + gamma * rij**2))*(X[j]-X[i]) + alpha * eps
                    # Clip values to boundaries
                    X[i] = np.clip(X[i], low, up)

        # Find best current global solution
        fitness, iter_bestf, iter_bestsol = find_best_solution(X, f, maximize=True)

        # If iteration best solution is better than previous ones
        # update
        if iter_bestf > bestf:
            bestf = iter_bestf
            bestsol = iter_bestsol

        print('\t best fitness:', bestf)
        t = t + 1
    return bestsol, bestf


def bats(f, low, up, D, N, alpha, gamma, n):
    ''' Bat Algorithm (BA)
    Inputs
    f: funct - Cost function to be optimized
    low: lower bound of the solution space
    up: upper bound of the solution space
    D: int - dimensionality of solution space
    N: number of iterations
    alpha: loudness
    gamma: pulse rates
    n: int - solution population size

    Outputs
    bestsol: vector - best solution vector in the solution space
    '''

    # Generate bats (random initial solutions)
    # make sure you don't get a bestf = inf with intial solutions
    bestf = np.inf
    while bestf == np.inf:
        X = generate_random_sols(D, low, up, n)
        for i in range(n):
            if f(X[[i]]) < bestf:
                bestf = f(X[[i]])
                bestsol = X[[i]]
    print('\t best fitness:', bestf)

    # Generate velocities
    V = np.zeros(X.shape)
    # Generate Frequencies
    Qmin = 0
    Qmax = 2
    # Generate pulse rates
    r0 = np.random.uniform(size=(n))  # init pulse rates
    r = np.random.uniform(size=(n))  # each bat pulse rate place holder
    # Generate Loudness
    A = np.ones(n)

    t = 0
    while t < N:
        for i in range(n):
            Q = Qmin + (Qmin - Qmax) * np.random.uniform()
            V[i] = V[i] + (X[i] - bestsol) * Q
            xi = X[i] + V[i]
            xi = np.clip(xi, low, up)
            # V[i] = np.clip(V[i],low/20,up/20) # clip velocity (small steps)

            # Pulse rate
            if np.random.uniform() > r[i]:
                xi = bestsol + 0.01 * np.random.normal(size=(1, D))
                xi = np.clip(xi, low, up)

            # Random walk
            xi = xi + 0.01 * np.random.normal(size=(1, D)) * np.mean(A)
            xi = np.clip(xi, low, up)

            # Evaluate solution and accept it under conditions
            if (f(xi) < bestf and np.random.uniform() < A[i]):
                X[i] = xi
                # Increase ri and reduce Ai
                A[i] = alpha * A[i]
                r[i] = r0[i] * (1 - np.exp(-gamma * t))

        # Find best current global solution
        iter_bestf = np.inf
        for i in range(n):
            if f(X[[i]]) < iter_bestf:
                iter_bestf = f(X[[i]])
                iter_bestsol = X[[i]]

        # If iteration best solution is better than previous ones
        # update
        if iter_bestf < bestf:
            bestf = iter_bestf
            bestsol = iter_bestsol
        print('\t best fitness:', bestf)

        t = t + 1
    return bestsol, bestf
