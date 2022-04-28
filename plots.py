

import matplotlib.pyplot as plt


baseline = [0.6981, 0.6831, 0.6174, 0.6164, 0.6386]
ourmodel = [0.9873, 0.6739, 0.6335, 0.6110, 0.6345]

epochs = [1, 2, 3, 4, 5]

plt.plot(epochs, baseline, label='Baseline')
plt.plot(epochs, ourmodel, label='Proposed model')
plt.title('Valid Loss')
plt.ylabel('Loss')
plt.legend()
plt.show()