import numpy as np
import matplotlib.pyplot as plt

def create_obstacle(Obs):
    theta = np.arange(0, 2 * np.pi * 5, 0.1)
    Obsx = Obs[1] + Obs[3] * np.sin(theta / 5)
    Obsy = Obs[2] + Obs[3] * np.cos(theta / 5)
    return Obsx, Obsy
        
def plot_obstacle(Obs):
    Obsx, Obsy = create_obstacle(Obs)
    plt.plot(Obsx, Obsy)

def main():

    Z    = np.loadtxt("results.txt", skiprows=1)
    Zref = np.loadtxt("reference.txt", skiprows=1)

    Obs  = np.loadtxt("obstacles.txt", skiprows=1, ndmin=2)

    xref  = Zref[:, 0]
    yref  = Zref[:, 1]
    thref = Zref[:, 2]
    vref  = Zref[:, 3]
    wref  = Zref[:, 4]

    x  = Z[:, 0]
    y  = Z[:, 1]
    th = Z[:, 2]
    v  = Z[:, 3]
    w  = Z[:, 4]

    plt.figure(1)
    plt.plot(x, y, label='Real')
    plt.plot(xref, yref, label='Reference')
    for i in range(Obs.shape[0]):
        plot_obstacle(Obs[i])
    # plt.plot(Obs1[:, 0], Obs1[:, 1])
    plt.xlabel("x")
    plt.xlabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(th)
    plt.plot(thref)
    plt.xlabel("t")
    plt.xlabel("theta")
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(v)
    plt.plot(vref)
    plt.xlabel("t")
    plt.xlabel("v")
    plt.grid()
    plt.show()

    plt.figure(4)
    plt.plot(w)
    plt.plot(wref)
    plt.xlabel("t")
    plt.xlabel("w")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()