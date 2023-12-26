import json
import matplotlib.pyplot as plt
import numpy as np

with open('./offset31.json', 'r') as file:
    data = json.load(file)
with open('./new_offset31.json', 'r') as file:
    data2 = json.load(file)
losses = data.get("losses", [])
losses = losses[1:]

new_losses = data2.get("losses", [])
new_losses = new_losses[1:]

steps = np.arange(5, len(losses) * 5 + 5, 5)

plt.plot(steps, losses, marker='o', linestyle='-')
plt.plot(steps, new_losses, marker='s', linestyle='-', color = 'red')

plt.title('lose curve')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.ylim(0, 1)
plt.savefig('combine_loss_curve.png')
plt.show()