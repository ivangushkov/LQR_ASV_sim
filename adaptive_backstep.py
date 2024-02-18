import numpy as np
import matplotlib.pyplot as plt
from hybridpath import HybridPathGenerator, HybridPathSignals

class AdaptiveBackstep:
    def __init__(self):
        self.init_system()

    def init_system(self):

        #self.bias = np.zeros(3)

        I = np.eye(3)

        K_1 = np.diag([10, 10, 10])
        kappa = 0.5
        self.K_1_tilde = K_1 + kappa*I
        self.K_2 = np.diag([60, 60, 30])
        self.tau_max = np.array([41.0, 50.0, 55.0]) # Må tilpasses thrusterne

        ## Forenklet modell ## Bør også endres
        m = 50
        self.M = np.diag([m, m, m])
        self.D = np.diag([10, 10, 5])

    def control_law(self, eta, nu, w, v_ref, dt_v_ref, dtheta_v_ref, eta_d, dtheta_eta_d, ddtheta_eta_d): # dtheta == ds
        _, R_trps = self.R(eta[2])
        S = self.S(nu[2])

        eta_error = eta - eta_d
        eta_error[2] = self.ssa(eta_error[2])

        z1 = R_trps @ eta_error
        alpha1 = -self.K_1_tilde @ z1 + R_trps @ dtheta_eta_d * v_ref

        z2 = nu - alpha1

        sigma1 = self.K_1_tilde @ (S @ z1) - self.K_1_tilde @ nu - S @ (R_trps @ dtheta_eta_d) * v_ref + R_trps @ dtheta_eta_d * dt_v_ref

        dtheta_alpha1 = self.K_1_tilde @ (R_trps @ dtheta_eta_d) + R_trps @ ddtheta_eta_d * v_ref + R_trps @ dtheta_eta_d * dtheta_v_ref

        # Control law ## Må endres når system-matrisene endres
        tau = -self.K_2 @ z2 + self.D @ nu + self.M @ sigma1 + self.M @ dtheta_alpha1 * (v_ref + w)

        # Add constraints to tau #

        if np.absolute(tau[0]) > self.tau_max[0] or np.absolute(tau[1]) > self.tau_max[1] or np.absolute(tau[2]) > self.tau_max[2]:
            if np.absolute(tau[0]) > self.tau_max[0]:
                tau[2] = np.sign(tau[2]) * np.absolute(self.tau_max[0] / tau[0]) * np.absolute(tau[2])
                tau[1] = np.sign(tau[1]) * np.absolute(self.tau_max[0] / tau[0]) * np.absolute(tau[1])
                tau[0] = np.sign(tau[0]) * self.tau_max[0]
            if np.absolute(tau[1]) > self.tau_max[1]:
                tau[2] = np.sign(tau[2]) * np.absolute(self.tau_max[1] / tau[1]) * np.absolute(tau[2])
                tau[0] = np.sign(tau[0]) * np.absolute(self.tau_max[1] / tau[1]) * np.absolute(tau[0])
                tau[1] = np.sign(tau[1]) * self.tau_max[1]
            if np.absolute(tau[2]) > self.tau_max[2]:
                tau[1] = np.sign(tau[1]) * np.absolute(self.tau_max[2] / tau[2]) * np.absolute(tau[1])
                tau[0] = np.sign(tau[0]) * np.absolute(self.tau_max[2] / tau[2]) * np.absolute(tau[0])
                tau[2] = np.sign(tau[2]) * self.tau_max[2]
        return tau
    
    def step(self, eta, nu, tau):
        pass

    def calculate_coriolis_matrix(self, nu):
        # u = nu[0]
        # v = nu[1]
        # r = nu[2]

        # C_RB = np.array([[0.0, 0.0, -self.m * (self.xg * r + v)], [0.0, 0.0, self.m * u],
        #                   [self.m*(self.xg*r+v), -self.m*u, 0.0]])
        # C_A = np.array([[0.0, 0.0, -self.M_A[1,1] * v + (-self.M_A[1,2])*r],[0.0,0.0,-self.M_A[0,0]*u],
        #                  [self.M_A[1,1]*v-(-self.M_A[1,2])*r, self.M_A[0,0]*u, 0.0]])
        # C = C_RB + C_A

        #return C
        pass

    def R(self,psi):
        R = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])
        R_T = np.transpose(R)
        return R, R_T
    
    def S(self,r):
        S = np.array([[0, -r, 0],
                    [r, 0, 0],
                    [0, 0, 0]])
        return S
    
    def ssa(self,angle):
        wrpd_angle = (angle + np.pi) % (2.0*np.pi) - np.pi
        return wrpd_angle

# Waypoints
pos = np.array([[10, 0],
        [10, 10],
        [0, 20],
        [30, 30],
        [40,0],
        [0,-10],
        [0, 0],
        [10, 0]])

pos2 = np.array([[10, 0], [20, 10], [30, 20], [50, 20],[50, 80], [-20, 80]])

lambda_val = 0.6  # Curvature constant
r = 2  # Differentiability order
generator = HybridPathGenerator(pos, r, lambda_val, 1)
path = generator.Path
N = path['NumSubpaths']
T = 200
dt = 0.05
time = np.arange(0, T, dt)

# Initial vectors
u_0 = 0 # Initial speed
eta_d = np.zeros((3, len(time)))
eta = np.zeros((3, len(time)))
eta_d[:,0] = np.array([pos[0,0], pos[0,1], np.pi/2])
eta[:,0] = np.array([pos[0,0], pos[0,1], np.pi/2])
nu = np.zeros((3, len(time)))
tau = np.zeros((3, len(time)))
AB = AdaptiveBackstep()
s = 0
sig = HybridPathSignals(path, s)

# Simulation loop
for i in range(1,len(time)):
    time_to_max_speed = 10
    if time[i] < time_to_max_speed:
        u_desired = 1/10 * time[i]
    else:
        u_desired = 1 # Desired speed
    v_ref, v_ref_s = sig.calc_vs(u_desired)
    s += v_ref * dt
    sig = HybridPathSignals(path, s)
    pd = sig.pd # Reference position
    psi_d = sig.psi # Reference heading
    eta_d[:,i] = np.array([pd[0], pd[1], psi_d])
    psi0 = eta_d[2,i-1]
    psi1 = eta_d[2,i]
    psi_vec = np.array([psi0, psi1])
    psi_vec = np.unwrap(psi_vec, period=2*np.pi)
    eta_d[2,i] = psi_vec[1]

    # Variables needed for tau
    #w = 0
    v_ref_t = 0 # Constant speed
    eta_d_s = np.array([sig.pd_der[0][0], sig.pd_der[0][1], sig.psi_der])
    eta_d_ss = np.array([sig.pd_der[1][0], sig.pd_der[1][1], sig.psi_dder])
    R, R_trsp = AB.R(eta[2,i-1])
    eta_error = eta[:,i-1] - eta_d[:,i]
    my = 0.5
    w = my * (eta_d_s @ R @ (R_trsp @ eta_error)) / (np.linalg.norm(eta_d_s)**2)

    tau[:,i] = AB.control_law(eta[:,i-1], nu[:,i-1], w, v_ref, v_ref_t, v_ref_s, eta_d[:,i], eta_d_s, eta_d_ss)
    
    # Step in nu and eta
    nu_dot = np.linalg.inv(AB.M) @ tau[:, i] - np.linalg.inv(AB.M) @ AB.D @ nu[:,i-1]
    nu[:,i] = nu[:,i-1] + nu_dot * dt
    eta_dot = R @ nu[:,i]
    eta[:,i] = eta[:,i-1] + eta_dot * dt

    psi0 = eta[2,i-1]
    psi1 = eta[2,i]
    psi_vec = np.array([psi0, psi1])
    psi_vec = np.unwrap(psi_vec, period=2*np.pi)
    eta[2,i] = psi_vec[1]

print(sig.pd)
print(sig.pd_der[0])
print(sig.pd_der[1])
print(v_ref)

print('Eta final: ',eta[:,-1])
print('Eta_d final: ', eta_d[:,-1])
print('Eta error final: ', eta_error)

plot = True
if plot:
    # Plotting
    plt.figure()
    plt.plot(eta_d[1,:], eta_d[0,:], label='Reference path', zorder = 0)
    plt.plot(eta[1,:], eta[0,:], label='Actual path', zorder = 1)
    for i in range(0, len(eta_d[2]), 100):
        plt.quiver(eta[1,i], eta[0,i], np.sin(eta[2,i]), np.cos(eta[2,i]), zorder = 2)
    plt.title('Actual path vs reference path')
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(time, eta[0,:], label='Actual x')
    plt.plot(time, eta_d[0,:], label='Reference x')
    plt.plot(time, eta[1,:], label='Actual y')
    plt.plot(time, eta_d[1,:], label='Reference y')
    plt.title('Actual position vs reference position')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(time, eta[2,:], label='Actual heading')
    plt.plot(time, eta_d[2,:], label='Reference heading')
    plt.title('Actual heading vs reference heading')
    plt.xlabel('Time [s]')
    plt.ylabel('Heading [rad]')
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(time, nu[0,:], label='Surge velocity')
    plt.plot(time, nu[1,:], label='Sway velocity')
    plt.title('velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(time, tau[0,:], label='Surge force')
    plt.plot(time, tau[1,:], label='Sway force')
    plt.title('Force')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.legend()

    plt.show()
