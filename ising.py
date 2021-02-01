import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


UP = 1
DOWN = -1


class IsingSim:
    def __init__(self, L, energy):
        self.L = L
        self.N = L * L
        self.dem_energy_dist = np.zeros(self.N, dtype=int)
        self.target_energy = energy
        self.sys_energy = 0
        self.dem_energy = 0
        self.mcs = 0
        self.sys_energy_acc = 0
        self.dem_energy_acc = 0
        self.magnetization = 0
        self.m_acc = 0
        self.m2_acc = 0
        self.accepted_moves = 0
        self.temperature = 0

        self.lattice = np.full((L, L), UP, dtype=np.int8)

        nx = [1, -1, 0, 0]
        ny = [0, 0, 1, -1]
        # Lookup table of neighbor direction vectors
        self.dirs = np.array(list(zip(nx, ny)))
        self.irand = 0
        self.nrand = 1024 * 10
        self.rand_pts = np.random.randint(self.L, size=(self.nrand, 2))

        self.init()

    def init(self):
        tries = 0
        energy = -self.N
        mag = self.N
        max_tries = 10 * self.N
        while energy < self.target_energy and tries < max_tries:
            pt = self.get_random_pt()
            de = self.get_delta(pt)
            if de > 0:
                energy += de
                spin = -self.lattice[pt]
                self.lattice[pt] = spin
                mag += 2 * spin
            tries += 1
        self.sys_energy = energy

    def step(self):
        for i in range(self.N):
            pt = self.get_random_pt()
            de = self.get_delta(pt)
            if de <= self.dem_energy:
                spin = -self.lattice[pt]
                self.lattice[pt] = spin
                self.accepted_moves += 1
                self.sys_energy += de
                self.dem_energy -= de
                self.magnetization += 2 * spin
            self.sys_energy_acc += self.sys_energy
            self.dem_energy_acc += self.dem_energy
            self.m_acc += self.magnetization
            self.m2_acc += self.magnetization * self.magnetization
        self.mcs += 1
        self.temperature = 4.0 / np.log(
            1 + 4 / (self.dem_energy_acc / self.mcs * self.N)
        )

    def get_delta(self, pt):
        # (-1, 0) and (0, -1) allow periodic wrapping from the left and top
        # edges to the right and bottom edges. (1, 0) and (0, 1) don't.
        nn = pt + self.dirs
        # Enforce periodic condition for (1, 0) and (0, 1)
        nn[nn == self.L] = 0
        # Use indexing tricks to get neighbors and sum them
        de = 2 * self.lattice[pt] * self.lattice[nn.T[0], nn.T[1]].sum()
        return de

    def get_random_pt(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.rand_pts = np.random.randint(self.L, size=(self.nrand, 2))
        pt = self.rand_pts[self.irand]
        self.irand += 1
        return tuple(pt)

    def get_state(self):
        return (
            self.lattice,
            self.sys_energy,
            self.dem_energy,
            self.magnetization,
            self.temperature,
        )


class SimAnimation:
    def __init__(self, sim, interval):
        self.sim = sim
        self.fig, self.axs = plt.subplots(2, 2, constrained_layout=True)
        self.im = None
        self.ani = None
        self.interval = interval
        self.paused = False
        self.sys_eng = []
        self.dem_eng = []
        self.mag = []
        self.temp = []

    def init(self):
        lattice, seng, deng, mag, temp = self.sim.get_state()
        self.sys_eng.append(seng)
        self.dem_eng.append(deng)
        self.mag.append(mag)
        self.temp.append(temp)
        self.im = self.axs[0, 0].imshow(
            lattice,
            interpolation="none",
            animated=True,
            cmap="gray",
        )
        plt.axis("off")
        return (self.im,)

    def update(self, *args):
        self.sim.step()
        lattice, seng, deng, mag, temp = self.sim.get_state()
        self.im.set_data(lattice)
        return (self.im,)

    def on_click(self, event):
        """Toggle play/pause with space bar"""
        if event.key != " ":
            return
        if self.paused:
            self.ani.event_source.start()
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.paused = True

    def run(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_click)
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            interval=self.interval,
            blit=True,
        )
        plt.show()


if __name__ == "__main__":
    sim = IsingSim(50, 100)
    SimAnimation(sim, 1).run()
