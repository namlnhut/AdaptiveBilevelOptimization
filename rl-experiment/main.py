import numpy as np
from src.environment import GridWorldEnvironment
from src.MDPsolver import MDPsolver
import argparse
import pickle
import copy
def find_fixed_point(policy_matrix, env):
    P = env.T.transpose((1,0,2)).reshape(env.n_states*env.n_actions, -1)
    P_pi = P.dot(policy_matrix)
    tol=1
    y=np.zeros((state_dim*n_actions, 1))
    while(tol > 1e-10):
        new_y = env.r.reshape(env.n_states*env.n_actions, -1) + env.gamma*P_pi.dot(y)
        tol = np.linalg.norm(new_y - y)
        y = new_y
    return y

def get_policy_matrix(policy,env):
    matrix = np.zeros((env.n_states,env.n_states*env.n_actions))
    for s in range(env.n_states):
        matrix[s][s*env.n_actions:(s+1)*env.n_actions] = policy[s]
    return matrix
def get_state_transitions(policy,env):
    matrix = np.zeros((env.n_states,env.n_states))
    for s in range(env.n_states):
        matrix[s][s*env.n_actions:(s+1)*env.n_actions] = policy[s]
    return matrix
def KL(x,y,rho):
    kl = 0
    for s in range(x.shape[0]):
        kl += rho[s]*x[s].dot(np.log(x[s]/y[s]))
    return  kl
def get_inner_grad(y, policy_matrix, env):
    P = env.T.transpose((1,0,2)).reshape(env.n_states*env.n_actions, -1)
    P_pi = P.dot(policy_matrix)
    bellman_residual = env.r.reshape(env.n_states*env.n_actions, -1) + env.gamma*P_pi.dot(y) - y
    return  (env.gamma*P_pi - np.eye(env.n_states*env.n_actions)).dot(bellman_residual)
    #return - bellman_residual
def compute_hypergrad(policy, y,gridworld,rho):
    """part_1 = y*gridworld.p_in.repeat(n_actions)
    P = gridworld.T.transpose((1,0,2)).reshape(gridworld.n_states*gridworld.n_actions, -1)
    P_pi = P.dot(policy_matrix)
    part_2 = policy.flatten()*gridworld.p_in.repeat(n_actions)
    matrix_1 = 2 * (gridworld.r.sum() + gridworld.gamma*P_pi.dot(y).sum() - y.sum())"""
    # We use the simplified formula of
    #a = rho.repeat(gridworld.n_actions).reshape(gridworld.n_states*gridworld.n_actions, -1)
    #b = policy.reshape(gridworld.n_states*gridworld.n_actions, -1)
    #return (1 - gridworld.gamma)**(-1)*y #*a*b
    return -y/(1 - gridworld.gamma)
parser = argparse.ArgumentParser(description='Primal Dual RL')
parser.add_argument('--n-states', type=int, default=10)
parser.add_argument('--grid-type', type=int, default=0)
parser.add_argument('--adaptive', action="store_true", default=False)
parser.add_argument('--theory-steps', action="store_true", default=False)
args = parser.parse_args()
gridworld = GridWorldEnvironment(args.grid_type, args.n_states, prop=0)
gridworld.gamma = 0.5
state_dim = gridworld.n_states
n_actions = gridworld.n_actions
a_tot = state_dim*n_actions
K = int(3e3) 
policy_s = np.ones(gridworld.n_actions)/gridworld.n_actions
policy = np.vstack(state_dim*[policy_s])
Us=[policy, policy]
policies = [policy,policy]
zs = []
values = []
gammas = [1]
solver = MDPsolver(gridworld)
solver.value_iteration()
v_out = solver.pi_eval(policy)
values.append((solver.v - v_out).dot(gridworld.p_in))

theory_inner_step = 1/(1 + (1 - gridworld.gamma)**2)
theory_outer_step = (1 - gridworld.gamma)**3
for k in range(K):
    #import pdb; pdb.set_trace()
    G=0
    ys = []
    weights = []

    y = np.zeros((state_dim*n_actions,1))       
    s_to_s_transitions = np.zeros((gridworld.n_states,gridworld.n_states))
    for s in range(gridworld.n_states):
        s_to_s_transitions[s,:] = policy[s].dot(gridworld.T[:,s,:])
    rho = (1 - gridworld.gamma)*np.linalg.solve(np.eye(gridworld.n_states) - gridworld.gamma*s_to_s_transitions, gridworld.p_in)
    policy_matrix = get_policy_matrix(policy,gridworld)
    true_avg_y =find_fixed_point(policy_matrix, gridworld)
    for t in range((k+1)*1000):
        inner_grad = get_inner_grad(y,policy_matrix, gridworld)
        if args.adaptive:
            inner_grad_norm = np.linalg.norm(inner_grad)**2
            G += inner_grad_norm
            if G > 0:
                weights.append(1/inner_grad_norm)
                eta = 10/((1 - gridworld.gamma)**2*G)
                y = y - eta*inner_grad #/inner_grad_norm
                y[y > (1 - gridworld.gamma)**(-1)] = (1 - gridworld.gamma)**(-1)
                y[y < - (1 - gridworld.gamma)**(-1)] = - (1 - gridworld.gamma)**(-1)
                ys.append(copy.deepcopy(y))
        else:
            if args.theory_steps:
                y = y - theory_inner_step*inner_grad/inner_grad_norm
            else:
                y = y - 0.9*inner_grad
        
    #rho=np.ones(state_dim)
    #avg_y = (np.array(ys).T.dot(np.array(weights))/np.sum(weights)).T if weights else y
    avg_y = y
    if args.adaptive:
        #print(np.linalg.norm(avg_y - true_avg_y), "error_inner_loop")
        zs.append( (KL(policies[k],policies[k-1],rho) + KL(policies[k-1],policies[k],rho))/gammas[k]**2) #zs squared KL(policies[k],policies[k-1],rho)
        gammas.append(100/np.sqrt(gammas[0] + np.sum(zs)))
    else:
        if args.theory_steps:
            gammas.append(theory_outer_step)
        else:
            gammas.append(0.5)
    hypergrad = compute_hypergrad(policy, avg_y, gridworld,rho)
    #print(hypergrad)
    hypergrad = -hypergrad.reshape(state_dim,n_actions)
    hypergrad -= np.max(hypergrad, axis=1, keepdims=True).repeat(n_actions,axis=1)
    policy_tilde = policy*np.exp(gammas[-1]*hypergrad)
    new_policy = policy_tilde/np.sum(policy_tilde, axis=1, keepdims=True).repeat(n_actions,axis=1)
    


    policies.append(new_policy)
    policy = new_policy
    #print(gammas[-1], "gamma")
    ## Evaluation
    #policy_avg = np.mean(policies, axis=0)
    v_out = solver.pi_eval(policy)
    #print(policies)
    values.append((solver.v - v_out).dot(gridworld.p_in))
    if not k%5:
        print(k, values[-1])
        with open(f"{args.adaptive}{args.n_states}{args.theory_steps}", "wb") as file:
            pickle.dump({"values":values, "policies":policies}, file)
    if np.isnan((solver.v - v_out).dot(gridworld.p_in)):
        break
    

