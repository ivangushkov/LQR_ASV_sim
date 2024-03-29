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

x_init = np.array([-10, -10, 40*np.pi/180, 0, 0, 0])
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

# Controller function
def controller_LQR(x, x_ref, K_LQR, K_r):
    u = -K_LQR @ x + K_r @ x_ref
    return u


# Simulation loop
x = x_init # Initial state
for i in range(num_steps+1):
    #x_ref[i] = reference_function_const(i * dt)  # Update the reference at each time step
    x_ref_history[i, :] = x_ref  # Update the reference at each time step
    
    u = controller_LQR(x, x_ref, K_LQR, K_r)  # Calculate control input 'u' at each time step
    x = asv.RK4_integration_step(x, u, dt)

    x_history[i] = x  # Store state history
    u_history[i] = u  # Store control input history

# Plot results
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
plt.xlabel('Time')
plt.ylabel('Control Input Value')
plt.legend()

plt.tight_layout()
plt.show()
