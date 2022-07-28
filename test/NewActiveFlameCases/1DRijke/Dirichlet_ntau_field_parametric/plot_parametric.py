import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [12.50, 5.50]



text_array = np.loadtxt('omega.txt')

c = text_array[:, 0]
dir_real = text_array[:, 1]
dir_imag = text_array[:, 2]


fig, ax = plt.subplots(nrows=1, ncols=2)


ax[0].plot(c, dir_real, label=r"$\omega_r$")
ax[0].set_xlabel("x_f")
ax[0].set_ylabel("Real part of Frequency")
ax[0].legend()

ax[1].plot(c, dir_imag, label=r"$\omega_i$")
ax[1].set_xlabel("x_f")
ax[1].set_ylabel("Imaginary part of Frequency")
ax[1].legend()

plt.savefig("x_f"+".pdf",bbox_inches='tight')
plt.show()
