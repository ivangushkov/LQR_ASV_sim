import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from asv_model import ASV

# Define system matrices A and B for a 3-dimensional state

m = 50
M = np.diag([m, m, m])
D = np.diag([10, 10, 5])
M_inv = np.linalg.inv(M)


heading_ref = 50*np.pi/180

asv = ASV(M, D)
A, B = asv.linearize_model(heading_ref)
C = np.zeros((3,6))
C[:3, :3] = np.eye(3)
## LQR stuff
# Define the cost matrices Q and R
Q = np.diag([10, 10, 5, 0.1, 1, 5])  # State cost matrix (2x2 identity matrix)
R = np.diag([1, 1, 1])  # Control input cost (1x1 identity matrix)

# Calculate the LQR gain matrix using the continuous-time algebraic Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Calculate the LQR gain matrix K
K_LQR = np.dot(np.dot(np.linalg.inv(R), B.T), P)
K_r = np.linalg.inv(C@np.linalg.inv(B@K_LQR - A)@B)

# Display the calculated LQR gain matrix
print("LQR Gain Matrix K:")
print(K_LQR)
print("Reference feedforward matrix K_r:")
print(K_r)

## Path parameters parameters
p0 = np.array([0, 0])
p1 = np.array([20, 20])

Pi_p = np.arctan2(p1[1]-p0[1], p1[0]-p0[0])
R_Pi_p = np.array(
    [[np.cos(Pi_p), -np.sin(Pi_p)],
    [np.sin(Pi_p), np.cos(Pi_p)]]
)

look_ahead = 5 # Look ahead distance


x_init = np.array([10, -20, 40*np.pi/180, 0, 0, 0])
x_ref = np.array([0, 0, heading_ref])

# Simulation parameters
T = 69.0        # Total simulation time
dt = 0.01       # Time step
num_steps = int(T / dt)  # Number of time steps

# Initialize arrays to store data
time = np.linspace(0, T, num_steps+1)
x_history = np.zeros((num_steps+1, A.shape[0]))
u_history = np.zeros((num_steps+1, B.shape[1]))
x_ref_history = np.zeros((num_steps+1, np.shape(x_ref)[0]))
cross_track_error_history = np.zeros(num_steps+1)
# Controller function
def controller_LQR(x, x_ref, K_LQR, K_r):
    u = -K_LQR @ x + K_r @ x_ref
    return u


# Simulation loop
x = x_init # Initial state
for i in range(num_steps+1):
    
    # generate reference at the look-ahead distance
    p_asv = np.array([x[0], x[1]])
    errors = R_Pi_p.T @ (p_asv - p0)
    along_track_error = errors[0]
    p_los_world = R_Pi_p @ np.array([along_track_error + look_ahead, 0]) + p0

    x_ref[:2] = p_los_world # Update the position reference at each time step
    x_ref_history[i, :] = x_ref
    
    # Calculate controller through LQR. Remeber that in practice you will also have thruster allocation!
    u = controller_LQR(x, x_ref, K_LQR, K_r)  # Calculate control input 'u' at each time step
    
    # Simulate system with Runge-Kutta4 integration (don't worry about it, it's like Forward Euler but better ^^)
    x = asv.RK4_integration_step(x, u, dt)

    x_history[i] = x  # Store state history
    u_history[i] = u  # Store control input history
    cross_track_error_history[i] = errors[1]

print(f"The cross track error by the end of the simulation: {cross_track_error_history[-1]}m")

# Plot shit
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
for j in range(3):
    plt.plot(time, x_history[:, j], label=f'State (x_{j+1})')
    plt.plot(time, x_ref_history[:, j], linestyle='--', label=f'Reference (x_ref_{j+1})')
plt.xlabel('Time')
plt.ylabel('State Value')
plt.legend()

plt.subplot(3, 1, 2)
for j in range(B.shape[1]):
    plt.plot(time, u_history[:, j], label=f'Control Input (u_{j+1})')
plt.xlabel('Time')
plt.ylabel('Control Input Value')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(x_history[:,0], x_history[:,1], label=f'Position')
plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', label='Path')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
plt.figure(1)
plt.plot(time, cross_track_error_history, label="cross track error")
plt.axis("equal")
plt.plot(time, np.zeros_like(time))
plt.legend()
plt.show()
