'''
Author: LOTEAT
Date: 2024-09-11 14:18:52
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# control point
control_points = np.array([
    # start point
    [0, 0],
    [1, 2],
    [2, -1],
    # end point
    [3, 0]
])

def bezier_curve(t, points):
    n = len(points) - 1
    curve = np.zeros((2,))
    for i in range(n + 1):
        binomial_coeff = np.math.comb(n, i)
        curve += binomial_coeff * (t**i) * ((1 - t)**(n - i)) * points[i]
    return curve

# update function
def update(num, line, control_points, control_lines, trajectory_lines):
    t = np.linspace(0, num / 100, 100)
    bezier_points = np.array([bezier_curve(ti, control_points) for ti in t])
    line.set_data(bezier_points[:, 0], bezier_points[:, 1])
    
    for i, (line, point) in enumerate(zip(control_lines, control_points)):
        x_data = [control_points[i, 0], bezier_points[-1, 0]]
        y_data = [control_points[i, 1], bezier_points[-1, 1]]
        line.set_data(x_data, y_data)
    
    if num > 0:
        trajectory_lines.set_data(bezier_points[:, 0], bezier_points[:, 1])
    
    return line, *control_lines, trajectory_lines


fig, ax = plt.subplots()
ax.set_xlim(-1, 4)
ax.set_ylim(-2, 3)

line, = ax.plot([], [], lw=2)

control_lines = []
for i in range(len(control_points) - 1):
    line, = ax.plot([], [], 'k--', lw=2)
    control_lines.append(line)
    
for i, (x, y) in enumerate(control_points):
    ax.text(x, y, f'P{i+1}', fontsize=12, ha='right')
    
ax.plot(control_points[:, 0], control_points[:, 1], 'ro', markersize=8)

trajectory_lines, = ax.plot([], [], 'b', lw=3, alpha=0.7)

ani = animation.FuncAnimation(
    fig, update, frames=np.arange(0, 101), fargs=(line, control_points, control_lines, trajectory_lines),
    blit=True, interval=50
)
ani.save('bezier_curve_%dorder.gif' % (len(control_points) - 1), writer='pillow')
plt.show()
