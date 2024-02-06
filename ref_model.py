import numpy as np
import matplotlib.pyplot as plt
import control


def R(psi):
    R = np.array([[np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]])
    return R

# From Fossen 2021 s. 337
# def x_d := [eta_d, eta_dot_d, eta_dot_dot_d]^T
# State space model: x_dot_d = A_d*x_d + B_d*r, where r is the reference setpoint
def asv_ref_model(eta_0, eta_ref, Ts, t_total):
    """
    Inputs: current position in NED, reference position in NED frame, time step and total time
    Output: Desired position in NED and velocity in body
    """
    N = int(t_total/Ts) # Sim length
    # Define time vector
    t = np.arange(0, t_total+2*Ts, Ts)

    # Initialize vectors
    x_d = np.zeros((9, N+2))
    x_d[:3,0] = eta_0
    eta_d = np.zeros((3, N+2))
    eta_d[:,0] = eta_0
    nu_d = np.zeros((3, N+2))


    # Define tuning parameters and matrices
    omega_b = 0.1 # Bandwidth for small vessel
    zeta = 0.7 # Damping ratio
    omega_n = omega_b/np.sqrt(1-2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2)) # Natural frequency
    Omega = np.diag([omega_n, omega_n, 0.3]) # Natural frequency matrix
    Delta = np.diag([zeta, zeta, zeta]) # Damping matrix

    # Defining A and B
    A_d = np.zeros((9,9))
    A_d[:3,3:6] = np.eye(3)
    A_d[3:6,6:] = np.eye(3)
    A_d[6:, :3] = -Omega**3
    A_d[6:, 3:6] = -(2*Delta+np.eye(3))@Omega**2
    A_d[6:, 6:] = -(2*Delta+np.eye(3))@Omega

    B_d = np.zeros((9,3))
    B_d[6:, :] = Omega**3

    sys = control.ss(A_d, B_d, np.eye(9), np.zeros((9,3)))
    sysd = control.c2d(sys, Ts, method='zoh')

    # Simulation
    for k in range(N+1):
        x_d[:,k+1] = sysd.A @ x_d[:,k] + sysd.B @ eta_ref
        eta_d[:,k+1] = x_d[:3,k]
        psi = eta_d[2,k]
        nu_d[:,k+1] = np.transpose(R(psi)) @ x_d[3:6,k]

    return eta_d, nu_d




def asv_vel_model(nu_0, nu_ref, Ts, t_total):
    N = int(t_total/Ts) # Sim length
    # Define time vector
    t = np.arange(0, t_total+2*Ts, Ts)

    # Initialize vectors
    x_d = np.zeros((6, N+2))
    x_d[:3,0] = nu_0
    nu_d = np.zeros((3, N+2))
    nu_d[:,0] = nu_0

    # Define tuning parameters and matrices
    omega_b = 0.1 # Bandwidth for small vessel
    zeta = 0.7 # Damping ratio
    omega_n = omega_b/np.sqrt(1-2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2)) # Natural frequency
    Omega = np.diag([0.5, omega_n, omega_n]) # Natural frequency matrix
    Delta = np.diag([0.9, zeta, zeta]) # Damping matrix

    # Defining A and B
    A_d = np.zeros((6,6))
    A_d[:3,3:] = np.eye(3)
    A_d[3:, :3] = -Omega**2
    A_d[3:, 3:] = -2*Delta@Omega

    B_d = np.zeros((6,3))
    B_d[3:, :] = Omega**2

    sys = control.ss(A_d, B_d, np.eye(6), np.zeros((6,3)))
    sysd = control.c2d(sys, Ts, method='zoh')

    # Simulation
    for k in range(N+1):
        x_d[:,k+1] = sysd.A @ x_d[:,k] + sysd.B @ nu_ref
        nu_d[:,k+1] = x_d[:3,k]

    return nu_d


# Example data
time = np.arange(0, 100+2*0.01, 0.01)
#eta, nu = asv_ref_model(np.array([0, 0, 0]), np.array([10, 10, np.pi/4]), 0.01, 100)
nu = asv_vel_model(np.array([0, 0, 0]), np.array([1.3, 0, 0]), 0.01, 100)

# Test nu
plt.figure('Velocity ref')
plt.plot(time, nu[0,:], label='u')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.grid()
plt.show()




#def test(pos: np.ndarray,  Ts: float, t_total: float):
#    pass


plot = False
if plot:
    # Plotting

    plt.figure('Heading')
    plt.plot(time,eta[2,:], label='psi')
    plt.plot(time, np.pi/4*np.ones(len(time)), label='psi_ref')
    plt.xlabel('Time [s]')
    plt.ylabel('Heading [rad]')
    plt.legend()
    plt.grid()

    scale_factor = 1.5
    selected_points = eta[:,::500]

    plt.figure('Position xy')
    plt.scatter(eta[1,:], eta[0,:], label='xy')
    plt.quiver(selected_points[1,:], selected_points[0,:], scale_factor*np.sin(selected_points[2,:]), 
            scale_factor*np.cos(selected_points[2,:]), label='heading')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.legend()
    plt.grid()

    plt.figure('Position x')
    plt.plot(time, eta[0,:], label='x')
    plt.plot(time, 10*np.ones(len(time)), label='x_ref')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid()

    plt.figure('Position y')
    plt.plot(time, eta[1,:], label='y')
    plt.plot(time, 10*np.ones(len(time)), label='y_ref')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid()

    plt.figure('Velocity')
    plt.plot(time, nu[0,:], label='u')
    plt.plot(time, nu[1,:], label='v')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid()

    U = np.sqrt(nu[0,:]**2+nu[1,:]**2)
    plt.figure('Speed')
    plt.plot(time, U, label='U')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.grid()
    plt.legend()

    plt.show()
    plt.close('all')


