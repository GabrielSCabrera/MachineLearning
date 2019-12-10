from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import matplotlib

class PDE_solver:

    def __init__(self):
        pass

    def set_initial_conditions(self, u0):
        """
            u0  –   1-D numpy array of length > 2
        """
        u0 = np.array(u0).squeeze()
        if u0.ndim != 1 or u0.shape[0] < 3:
            msg = ("Parameter 'u0' in 'set_initial_conditions' must be a "
                   "1-D array of length > 2")
            raise ValueError(msg)

        self.u0 = u0
        # Number of position points
        self.N = self.u0.shape[0] if u0.ndim == 1 else 1

    def solve(self, t, x):
        """
            T   –   simulation runtime
            dt  –   simulation time-step
        """
        @njit
        def get_u_t(u2, u1):
            return u2 - u1

        @njit
        def get_u_x(u, du):
            diff = u[1:] - u[:-1]
            du[1:-1] = diff[1:] - diff[:-1]
            return du

        @njit
        def integrator(u, t, du, dx_2, dt):
            perc = 0
            new_perc = 0
            for i in range(u.shape[0]-1):
                u[i+1] = u[i] + 0.5*get_u_x(u[i], du)*dx_2
                u[i+1] = u[i+1] + get_u_t(u[i+1], u[i])*dt
                u[i+1,0] = 0
                u[i+1,-1] = 0
                new_perc = int(100*i/(u.shape[0]-1))
                if perc < new_perc:
                    perc = new_perc
                    print(perc)
            print("100")
            return u

        dt = t[1]-t[0]
        dx = x[1]-x[0]

        # Initializing array 'u'
        u = np.zeros((t.shape[0], x.shape[0]))

        # Setting initial conditions
        u[0] = np.array(self.u0)

        # Allocating memory to acceleration functions
        du = np.zeros(self.N)

        u = integrator(u, t, du, dx**2, dt)

        self.u = u
        self.x = x
        self.t = t
        self.dt = dt
        self.dx = dx

    def plot(self, max_points = 1E4):

        u, x, t = self.u, self.x, self.t

        x_step = np.min((np.max((int(x.shape[0]//max_points), 1)), x.shape[0]))
        t_step = np.min((np.max((int(t.shape[0]//max_points), 1)), t.shape[0]))

        x = x[::x_step]
        t = t[::t_step]
        u = u[::t_step, ::x_step]

        x, t = np.meshgrid(x, t)

        fig, ax = plt.subplots()

        im = ax.pcolormesh(t, x, u, cmap = plt.get_cmap('magma'))
        cbar = fig.colorbar(im, ax = ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        cbar.ax.set_ylabel("Function $u(x,t)$")
        fig.tight_layout()

        plt.show()

if __name__ == "__main__":
    L, dx = 1, 0.01
    dt_min = 0.5*dx**2
    T, dt = 5E4, 0.001

    if dt < dt_min:
        print(f"dt is too small, should be greater than {dx**2:g}")
    # Array of time values from 0 to T
    t = np.arange(0, T + dt, dt)
    # Array of position values from 0 to L
    x = np.arange(0, L + dx, dx)
    # Array of initial conditions
    u0 = np.abs(np.random.normal(1E4, 1E3, x.shape[0]))
    u0 = u0 + np.min(u0)

    I = PDE_solver()

    I.set_initial_conditions(u0)

    I.solve(t, x)
    I.plot()
