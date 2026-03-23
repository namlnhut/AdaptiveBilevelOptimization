import numpy as np
from src.environment import GridWorldEnvironment
from src.MDPsolver import MDPsolver
import argparse
import pickle
import copy
import os
import logging
from scipy.optimize import LinearConstraint, minimize, Bounds, linprog

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
def euclidean(policy1, policy2, rho):
    return np.sum([np.linalg.norm(policy1[s] - policy2[s])*rho[s] for s in range(policy1.shape[0])])
def compute_projection(y):
    dist = lambda x : np.linalg.norm(x - y)
    tol = None
    y = y.flatten()
    output = minimize(dist, x0=y, 
                        constraints = [LinearConstraint(A,np.ones(state_dim),np.ones(state_dim))], 
                        bounds= Bounds(0, 1), 
                        method="SLSQP", tol=tol) #, method = "trust-constr", hess= lambda x: np.zeros((g.nbr_edges, g.nbr_edges)))
    if not output.success:
        raise Exception(f"Optimization Problem: {output.message}") 
    return output.x.reshape(state_dim,n_actions)
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
    #return bellman_residual
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
parser = argparse.ArgumentParser(description='Primal Dual RL (Euclidean)')
parser.add_argument('--n-states', type=int, default=10)
parser.add_argument('--grid-type', type=int, default=0)
parser.add_argument('--adaptive', action="store_true", default=False)
parser.add_argument('--theory-steps', action="store_true", default=False)
parser.add_argument('--max-iter', type=int, default=300)
parser.add_argument('--output-dir', type=str, default=None)
args = parser.parse_args()

# ---- output / logging setup ----
_here = os.path.dirname(os.path.abspath(__file__))
_out  = args.output_dir or os.path.join(_here, "results", "euclidean")
_log  = os.path.join(_here, "logs")
os.makedirs(_out, exist_ok=True)
os.makedirs(_log, exist_ok=True)

_mode     = "adaptive" if args.adaptive else ("theory" if args.theory_steps else "fixed")
_run_name = f"euclidean_{_mode}_S{args.n_states}_G{args.grid_type}"
_ckpt     = os.path.join(_out, f"{_run_name}.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(_log, f"{_run_name}.log"), mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Run: {_run_name}  |  output: {_ckpt}")

gridworld = GridWorldEnvironment(args.grid_type, args.n_states, prop=0)
gridworld.gamma = 0.5
state_dim = gridworld.n_states
n_actions = gridworld.n_actions
A = np.zeros((state_dim, state_dim*n_actions))
for s in range(state_dim):
    A[s][s*n_actions:(s+1)*n_actions] = 1
a_tot = state_dim*n_actions
K = args.max_iter
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
    y = np.zeros((state_dim*n_actions,1))
    weights = [] #np.ones((state_dim*n_actions,1))/(1 - gridworld.gamma)
    s_to_s_transitions = np.zeros((gridworld.n_states,gridworld.n_states))
    for s in range(gridworld.n_states):
        s_to_s_transitions[s,:] = policy[s].dot(gridworld.T[:,s,:])
    rho = (1 - gridworld.gamma)*np.linalg.solve(np.eye(gridworld.n_states) - gridworld.gamma*s_to_s_transitions, gridworld.p_in)
    #rho = np.ones(state_dim*n_actions)/state_dim/n_actions
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
                y = y - eta*inner_grad/inner_grad_norm
                y[y > (1 - gridworld.gamma)**(-1)] = (1 - gridworld.gamma)**(-1)
                y[y < - (1 - gridworld.gamma)**(-1)] = - (1 - gridworld.gamma)**(-1)
                ys.append(copy.deepcopy(y))
        else:
            if args.theory_steps:
                y = y - theory_inner_step*inner_grad
            else:
                y = y - 0.9*inner_grad
            y[y > (1 - gridworld.gamma)**(-1)] = (1 - gridworld.gamma)**(-1)
            y[y < - (1 - gridworld.gamma)**(-1)] = - (1 - gridworld.gamma)**(-1)

        
    #rho=np.ones(state_dim)
    avg_y = y #(np.array(ys).T.dot(np.array(weights))/np.sum(weights)).T if weights else y
    #print(np.linalg.norm(avg_y - true_avg_y), "error_inner_loop")
    #import pdb;pdb.set_trace()
    if args.adaptive:
        zs.append(euclidean(policies[k], policies[k-1], rho)/gammas[k]**2) #zs squared KL(policies[k],policies[k-1],rho)
        gammas.append(10/np.sqrt(gammas[0] + np.sum(zs)))
    else:
        if args.theory_steps:
            gammas.append(theory_outer_step)
        else:
            gammas.append(0.5)
    hypergrad = compute_hypergrad(policy, avg_y, gridworld,rho)
    #print(hypergrad)
    new_policy = np.ones_like(policy)
    policy_tilde = policy - gammas[-1]*hypergrad.reshape(state_dim,n_actions)
    new_policy = compute_projection(policy_tilde)
    policies.append(new_policy)
    policy = new_policy
    #print(gammas[-1], "gamma")
    ## Evaluation
    #policy_avg = np.mean(policies, axis=0)
    v_out = solver.pi_eval(policy)
    values.append((solver.v - v_out).dot(gridworld.p_in))
    if not k % 5:
        logger.info(f"k={k:5d}  gap={values[-1]:.6e}  gamma={gammas[-1]:.4f}")
        with open(_ckpt, "wb") as file:
            pickle.dump({"values": values, "policies": policies,
                         "args": vars(args)}, file)

    if np.isnan((solver.v - v_out).dot(gridworld.p_in)):
        logger.warning("NaN detected — stopping early.")
        break

with open(_ckpt, "wb") as file:
    pickle.dump({"values": values, "policies": policies, "args": vars(args)}, file)
logger.info(f"Done. Final gap={values[-1]:.6e}  Saved to {_ckpt}")



