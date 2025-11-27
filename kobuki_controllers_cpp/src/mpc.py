from casadi import *
import matplotlib.pyplot as plt
import numpy as np
'''
    This script sets up a nonlinear programming problem using CasADi to minimize a function. (gotten from https://www.youtube.com/watch?v=RrnkPrcpyEA)
'''

# x   = SX.sym('w')
# # obj = x*x*x*x - 5*x*x + 4
# obj = exp(0.2*x)*sin(x)

# g = []
# p = []

# OPT_variables = x
# nlp_prob      = {'x': OPT_variables, 'f': obj, 'g': g, 'p': p}
# # nlp_prob['x'] = vertcat(x)
# # nlp_prob['f'] = vertcat(obj)
# # nlp_prob['g'] = vertcat(g)
# # nlp_prob['P'] = vertcat(P)

# opts = {'ipopt': {'print_level': 0, 'max_iter': 100, 'acceptable_tol': 1e-6, 'acceptable_obj_change_tol': 1e-6}, 'print_time': 0}

# solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

# args = {'lbx': 0, 'ubx': 4*pi, 'lbg': -inf, 'ubg': inf, 'p': [], 'x0': 10.0}

# sol = solver(x0 = args['x0'], lbx = args['lbx'], ubx = args['ubx'], lbg = args['lbg'], ubg = args['ubg'], p = args['p'])

# x_sol     = sol['x']
# min_value = sol['f']

# print('x_sol:', x_sol)
# print('min_value:', min_value)

'''
    This script sets up a nonlinear programming problem using CasADi to minimize a sum. (gotten from https://www.youtube.com/watch?v=RrnkPrcpyEA)
'''

# x = [0, 45, 90, 135, 180]
# y = [667, 661, 757, 871, 1210]

# plt.figure()
# plt.plot(x, y, 'ro', label='data points')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data Points')

# m = SX.sym('m')
# c = SX.sym('c')

# obj = 0

# for i in range(len(x)):
#     obj += (y[i] - (m*x[i] + c))**2

# g = []
# p = []

# OPT_variables = vertcat(m, c)
# nlp_prob      = {'x': OPT_variables, 'f': obj, 'g': g, 'p': p}

# opts = {'ipopt': {'print_level': 0, 'max_iter': 100, 'acceptable_tol': 1e-6, 'acceptable_obj_change_tol': 1e-6}, 'print_time': 0}

# solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

# args = {'lbx': [-inf, -inf], 'ubx': [inf, inf], 'lbg': -inf, 'ubg': inf, 'p': [], 'x0': [0.5, 1.0]}

# sol = solver(x0 = args['x0'], lbx = args['lbx'], ubx = args['ubx'], lbg = args['lbg'], ubg = args['ubg'], p = args['p'])

# x_sol     = sol['x']
# min_value = sol['f']
# m_val = float(x_sol[0])
# c_val = float(x_sol[1])
# print('x_sol:', x_sol)
# print('min_value:', min_value)

# plt.plot(x, [m_val*xi + c_val for xi in x], 'b-', label='fitted line')
# plt.grid()
# plt.legend()
# plt.show()

'''
    This script sets up a nonlinear programming problem using CasADi to minimize a sum for point stabilization of a differential drive robot. (gotten from https://www.youtube.com/watch?v=RrnkPrcpyEA)
'''

def shift(T, t0, x0, u, f):
    """
    Avanza el estado del sistema un paso usando el primer control de la secuencia.
    Desplaza la secuencia de controles para el siguiente ciclo MPC.
    """
    st = np.array(x0)
    con = np.array(u[:, 0]).flatten()  # Primer control de la secuencia

    f_value = np.array(f(st, con).full()).flatten()
    st = st + T * f_value
    x0 = st.tolist()
    t0 = t0 + T

    u = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)

    return t0, x0, u

T = 0.2
N = 10

rob_diam = 0.3

v_max = 0.6
v_min = -v_max
w_max = pi/4
w_min = -w_max

x  = SX.sym('x')
y  = SX.sym('y')
th = SX.sym('th')

states   = vertcat(x, y, th)
n_states = states.numel()

v = SX.sym('v')
w = SX.sym('w')
controls   = vertcat(v, w)
n_controls = controls.numel()

rhs = vertcat(v*cos(th), v*sin(th), w)

f = Function('f', [states, controls], [rhs])
U = SX.sym('U', n_controls, N)
p = SX.sym('P', n_states + n_states)

X = SX.sym('X', n_states, N+1)

X[:, 0] = p[:n_states]

for i in range(N):
    st  = X[:, i]
    con = U[:, i]
    f_value = f(st, con)
    st_next = st + (T*f_value)
    X[:, i+1] = st_next

ff = Function('ff', [U, p], [X]) 

obj = 0
g   = []

Q = diag([1, 5, 0.1])
R = diag([0.5, 0.05])

for i in range(N):
    st  = X[:, i]
    con = U[:, i]
    obj += (st - p[n_states:6]).T @ Q @ (st - p[n_states:6]) + con.T @ R @ con

for i in range(N+1):
    g.append(X[0, i])
    g.append(X[1, i])

g = vertcat(*g)

OPT_variables = vertcat(reshape(U, n_controls*N, 1))
nlp_prob      = {'x': OPT_variables, 'f': obj, 'g': g, 'p': p}

opts = {'ipopt': {'print_level': 0, 'max_iter': 100, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6}, 'print_time': 0}

print(type(opts))

solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

args = {
    'lbg': [-2]*(N+1)*2,    # 2 restricciones por paso (x, y)
    'ubg': [2]*(N+1)*2,
    'lbx': [v_min, w_min]*N,  # Repite para cada paso del horizonte
    'ubx': [v_max, w_max]*N
}

t0 = 0
x0 = [0, 0, 0]
xs = [1.5, 1.5, pi/2]
xx = [x0]
t  = [t0]
u0 = [0] * (n_controls * N)
sim_tim = 20
mpciter = 0
xx1 = []
u_cl = []

while np.linalg.norm(np.array(x0) - np.array(xs), 2) > 1e-2 and mpciter < sim_tim/T:
    p = vertcat(x0, xs)
    args['p']  = p
    args['x0'] = vertcat(reshape(u0, n_controls*N, 1))

    sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

    u_opt = reshape(sol['x'], n_controls, N)
    u_opt_np = u_opt.full()
    u0 = u_opt_np.flatten()
    u_cl.append(u_opt_np[:, 0].tolist())

    x1 = ff(u_opt, p).full().flatten()
    xx1.append(x1.tolist())

    t.append(t0)
    t0, x0, u0 = shift(T, t0, x0, u_opt_np, f)
    xx.append(x0 if isinstance(x0, list) else x0.tolist())

    mpciter += 1



xx = np.array(xx)
xs_arr = np.array(xs)

plt.figure()
plt.plot(xx[:, 0], xx[:, 1], 'b-', label='Trayectoria seguida')
plt.plot(xs[0], xs[1], 'ro', label='Referencia')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trayectoria MPC')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()


u_cl = np.array(u_cl)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(u_cl[:, 0], label='v (velocidad lineal)')
plt.ylabel('v [m/s]')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(u_cl[:, 1], label='w (velocidad angular)')
plt.xlabel('Iteración')
plt.ylabel('w [rad/s]')
plt.legend()
plt.grid()

plt.suptitle('Controles aplicados por MPC')
plt.show()