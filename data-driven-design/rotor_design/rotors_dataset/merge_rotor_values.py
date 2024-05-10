import numpy as np

rotor_values = np.concatenate([np.load('rotor_values_part%d.npy' % i)[:, :, np.newaxis] for i in range(10)], axis=2)

np.save('rotor_values.npy', rotor_values)