import argparse
import pickle
import numpy as np

g = lambda z,w : (100*z - 10)**2 + (w - 2)**2
f = lambda u, M: u.dot(M.dot(u))

grad_g = lambda z,w: np.array([200*(100*z - 10), 2*(w - 2)])
grad_f = lambda u, M: M.dot(u)

## The matrix M is zero everywhere

parser = argparse.ArgumentParser(description='Primal Dual RL')
parser.add_argument('--adaptive', action="store_true", default=False)
parser.add_argument('--eta', type=float, default=2/20002)
parser.add_argument('--gamma', type=float, default=1/2)
args = parser.parse_args()

K =1000
x_zero =np.array([-7,7])
xs = [x_zero,x_zero]
gammas = [1]
zs=[]
values = []
for k in range(K):
    #import pdb; pdb.set_trace()
    G=0
    ys = []
    weights = []
    y = np.array([1, 3])
    for t in range(k+1):
        inner_grad = grad_g(y[0],y[1])
        if args.adaptive:
            inner_grad_norm = np.linalg.norm(inner_grad)**2
            G += inner_grad_norm
            if G > 0:
                weights.append(1/inner_grad_norm)
                eta = 1/(2*G)
                y = y - eta*inner_grad/inner_grad_norm
                ys.append(y)
        else:
            y = y - args.eta*inner_grad
            #import pdb; pdb.set_trace()

    if args.adaptive:
        avg_y = (np.array(ys).T.dot(np.array(weights))/np.sum(weights)).T if weights else y
        zs.append((np.linalg.norm(xs[k] - xs[k+1])**2)/gammas[k]**2) #zs squared
        gammas.append(1/np.sqrt(1 + np.sum(zs)))
        hypergrad = grad_f(xs[-1], np.diag(avg_y))
        #print(hypergrad)
        xs.append(xs[-1] - gammas[-1]*hypergrad)
    else:
        hypergrad = grad_f(xs[-1], np.diag(y))
        xs.append(xs[-1] - args.gamma*hypergrad)
    if f(xs[-1], np.array([[1/10, 0],[0,2]])) > 1e30:
        break
    else:
        values.append(f(xs[-1], np.array([[1/10, 0],[0,2]])))
        print(values[-1])
file_name = "toy_adaptive" if args.adaptive else f"eta{args.eta}gamma{args.gamma}"
with open(file_name, "wb") as file:
    pickle.dump({"xs":xs, "values": values}, file)

