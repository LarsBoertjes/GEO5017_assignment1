import matplotlib.pyplot as plt
import pandas as pd
# will get warning if you don't have pyarrow on your system / venv

df = pd.read_csv('data/positions.csv')

x = df['X']
y = df['Y']
z = df['Z']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Trajectory Plot')
plt.show()