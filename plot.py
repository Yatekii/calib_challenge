from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('labeled/0.txt', delimiter=' ')
df.columns = ['pitch', 'yawca']
ax = df.plot()
ax.set_xlabel("frame")
ax.set_ylabel("rad")
plt.show()