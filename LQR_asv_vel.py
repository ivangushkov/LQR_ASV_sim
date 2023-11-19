import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# Define system matrices A and B for a 3-dimensional state

m = 50
M = np.diag([m, m, m])
D = np.diag([10, 10, 5])
M_inv = np.linalg.inv(M)

A = -M_inv @ D  # System matrix 'A'
B = M_inv  # Input matrix 'B'
B_inv = M

## LQR stuff
# Define the cost matrices Q and R
Q = np.diag([10, 10, 5])  # State cost matrix (2x2 identity matrix)
R = np.diag([1, 1, 1])  # Control input cost (1x1 identity matrix)

# Calculate the LQR gain matrix using the continuous-time algebraic Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Calculate the LQR gain matrix K
K_LQR = np.dot(np.dot(np.linalg.inv(R), B.T), P)
K_r = np.linalg.inv(np.linalg.inv(B@K_LQR - A)@B)

# Display the calculated LQR gain matrix
print("LQR Gain Matrix K:")
print(K_LQR)
print("Reference feedforward matrix K_r:")
print(K_r)

# Simulation parameters
T = 69.0        # Total simulation time
dt = 0.01       # Time step
num_steps = int(T / dt)  # Number of time steps

# Initialize arrays to store data
time = np.linspace(0, T, num_steps+1)
x_history = np.zeros((num_steps+1, A.shape[0]))
x_dot_history = np.zeros((num_steps+1, A.shape[0]))
u_history = np.zeros((num_steps+1, B.shape[1]))
x_ref = np.zeros((num_steps+1, A.shape[0]))
x_ref_dot = np.zeros((num_steps+1, A.shape[0]))

# Controller function (example: proportional control)
def controller(x, x_ref):
    k_p = 100.0  # Proportional gain
    error = x_ref - x
    return k_p * error

def controller_LQR(x, x_ref, K_LQR, K_r, B_inv, x_ref_dot):
    u = -K_LQR @ x + K_r @ x_ref + B_inv @ x_ref_dot
    return u

# Define a time-varying reference function for the 3-dimensional state x

def reference_function_const(t):
    return np.array([2, 0, 0])

def reference_function(t):
    return np.array([np.sin(2 * t), np.cos(t), np.exp(-t)])

def reference_dot_function(t):
    return np.array([2*np.cos(2 * t), -np.sin(t), -np.exp(-t)])

# Simulation loop
x = np.array([1.0, 0.5, -1.0])  # Initial state
for i in range(num_steps+1):
    #x_ref[i] = reference_function_const(i * dt)  # Update the reference at each time step
    x_ref[i] = reference_function(i * dt)  # Update the reference at each time step
    x_ref_dot[i] = reference_dot_function(i * dt)
    u = controller_LQR(x, x_ref[i], K_LQR, K_r, B_inv, x_ref_dot[i])  # Calculate control input 'u' at each time step
    x_dot = np.dot(A, x) + np.dot(B, u)  # Calculate x_dot using the system dynamics
    x += dt * x_dot  # Update state using Forward Euler integration
    x_history[i] = x  # Store state history
    x_dot_history[i] = x_dot  # Store state history
    u_history[i] = u  # Store control input history

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for j in range(A.shape[0]):
    plt.plot(time, x_history[:, j], label=f'State (x_{j+1})')
    plt.plot(time, x_ref[:, j], linestyle='--', label=f'Reference (x_ref_{j+1})')
plt.xlabel('Time')
plt.ylabel('State Value')
plt.legend()

plt.subplot(2, 1, 2)
for j in range(B.shape[1]):
    plt.plot(time, u_history[:, j], label=f'Control Input (u_{j+1})')
plt.xlabel('Time')
plt.ylabel('Control Input Value')
plt.legend()

plt.figure(figsize=(12, 4))
for j in range(A.shape[0]):
    plt.plot(time, x_dot_history[:, j], label=f'State Derivative (x_dot_{j+1})')
    plt.plot(time, x_ref_dot[:, j], label=f'State Derivative reference (x_dot_ref_{j+1})')
plt.xlabel('Time')
plt.ylabel('State Derivative Value')
plt.legend()

plt.tight_layout()
plt.show()