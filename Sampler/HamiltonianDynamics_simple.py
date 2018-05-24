import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def kinetic_energy(velocity):
    """
    calculate kinetic energy
    :param velocity: torch.tensor [num_samples, D_velocity]
    :return: a vector with length of num_samples
    """

    return 0.5 * torch.sum((velocity.data)**2, dim=1) # 1D FloatTensor of size batch_size

# compute energy: U + K
def hamiltonian(pos, vel, energy_fn):
    """
    Hamiltionian energy = potential + kinetic
    :param pos: torch.tensor [num_samples, D_position]
    :param vel: torch.tensor [num_samples, D_velocity]
    :param energy_fn: a function
    :return: a vector with length of num_samples
    """
    U = energy_fn(pos)  # 1D Variable of size batch_size
    U = U.data
    K = kinetic_energy(vel)
    return U + K


def metropolis_hastings_accept(energy_prev, energy_next):
    """
    Performs a Metropolis-Hastings accept-reject move.
    :param energy_prev: torch.tensor [num_samples],
        the energy in time-step t
    :param energy_next: torch.tensor [num_samples],
        the energy in time-step t+1
    :return: ByteTensor torch.tensor [num_samples]
        true->accept
    """
    energy_diff = energy_prev - energy_next # 1D FloatTensor of size batch_size
    # energy_diff = min(1, energy_diff)
    energy_diff = torch.min(torch.ones(1), energy_diff)
    # random_sample = torch.rand(1)[0]
    random_sample = torch.rand(len(energy_diff))
    return (torch.exp(energy_diff) - random_sample) >= 0


# obtain a single sample after n_steps leapfrog
def simulate_dynamic(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    """
    return final (position, velocity) after n_steps leapfrog
    :param initial_pos: batch_size, dim
    :param initial_vel: batch_size, dim
    :param stepsize:
    :param n_steps:
    :param energy_fn:
    :return: final_position, final_velocity
    """

    def leapfrog(pos, vel, step):
        """

        :param pos: torch.tensor [D_position],
            position at time t
        :param vel: torch.tensor [D_velocity],
            velocity at time (t-stepsize/2)
        :param step: scalar
        :return: new_pos, new_vel
            position at time (t+stepsize)
            velocity at time (t+stepsize/2)
        """
        # grad of potential nergy
        # energy_fn.backward(torch.ones(initial_pos.size()))
        if not pos.requires_grad:
            pos.requires_grad = True
            pos.volatile = False
        potential_of_give_pos = energy_fn(pos)
        potential_of_give_pos.backward()
        dE_dpos = pos.grad

        new_vel = vel - step * dE_dpos  # at time (t + stepsize/2)
        new_pos = pos + step * new_vel  # at time (t + stepsize)
        return new_pos, new_vel

    # velocity at time t+stepsize/2
    # initial_pos = Variable(initial_pos, requires_grad=True)
    # initial_vel = Variable(initial_vel, requires_grad=False)

    initial_potential = energy_fn(initial_pos)
    initial_potential.backward()
    dE_dpos = initial_pos.grad
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos

    # position at time t+stepsize
    pos_full_step = initial_pos + stepsize * vel_half_step

    # perform leapfrog
    temp_pos = pos_full_step
    temp_vel = vel_half_step
    for lf_step in range(n_steps):
        temp_pos, temp_vel = leapfrog(temp_pos, temp_vel, stepsize)

    final_pos = temp_pos
    final_vel = temp_vel

    # final_pos = Variable(final_pos, requires_grad=True)
    final_pos.requires_grad = True
    final_pos.volatile = False
    potential = energy_fn(final_pos)
    potential.backward()
    final_vel = final_vel - 0.5 * stepsize * final_pos.grad

    return final_pos, final_vel


def hmc_move(positions, energy_fn, stepsize, n_steps):
    """
    Perform one iteration of sampling.
    1. Start by sampling a random velocity from a Gaussian.
    2. Perform n_steps leapfrog
    3. decide whether to accept or reject
    :param positions: start sampling from positions
    :param energy_fn: potential energy function
    :param stepsize: leapfrog stepsize
    :param n_steps: leapfrog steps
    :return: accept(bool), final_pos(torch.tensor)
    """
    initial_vel = torch.randn(positions.size()) # with zero mean and unit variance
    positions = Variable(positions, requires_grad=True)
    initial_vel = Variable(initial_vel, requires_grad=True)

    final_pos, final_vel = simulate_dynamic(initial_pos=positions,
                                            initial_vel=initial_vel,
                                            stepsize=stepsize,
                                            n_steps=n_steps,
                                            energy_fn=energy_fn)

    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn)
    )

    return accept, final_pos.data

def hmc_sampling(init_pos, energy_fn, n_samples, stepsize=0.01, n_steps=20, gap=20):
    # result_samples = torch.zeros(n_samples+gap, 1, 2)
    # last_pos = init_pos
    # result_samples.append(init_pos)
    # result_samples[0, :, :] = init_pos
    last_pos = init_pos

    for i in range(1, n_samples+gap):
        # last_pos = result_samples[i-1, :, :]

        accept, new_pos = hmc_move(last_pos, energy_fn, stepsize, n_steps)

        # if accept:
            # result_samples.append([new_pos[0][0], new_pos[0][1]])
            # result_samples[i, :, :] = new_pos
        # else:
            # result_samples.append([last_pos[0][0], last_pos[0][1]])
            # result_samples[i, :, :] = last_pos

        last_pos = accept.float() * new_pos + (1 - accept.float()) * last_pos

    # return result_samples[gap:, :, :]
    return last_pos

def NormalEnergy(x):
    # x = Variable(x, requires_grad=True)
    # x: batch_size, dim
    # u = torch.zeros(1, 2)
    u = torch.FloatTensor([2, 2])
    Sigma = torch.FloatTensor([[1.0, 0.8], [0.8, 1.0]])
    Sigma = torch.inverse(Sigma)
    # Sigma = Sigma.t()

    if isinstance(x, Variable):
        u = Variable(u, requires_grad=True)
        Sigma = Variable(Sigma, requires_grad=True)

    diff = x - u

    temp = 0.5 * torch.mul(torch.matmul(diff, Sigma), diff)
    temp = torch.sum(temp, dim=1)

    return temp # batch_size 1D FloatTensor

def grad_test():
    x = torch.ones(1, 2)
    x = Variable(x, requires_grad=True)
    y = NormalEnergy(x)
    y.backward()
    print(x.grad)

def vis_test():
    n_samples = 25
    stepsize = 0.1
    n_steps = 20
    dim = 2
    batch_size = 2

    initial_pos = torch.randn(batch_size, dim)
    samples = hmc_sampling(initial_pos, NormalEnergy, n_samples, stepsize, n_steps)
    # samples = samples.view(samples.size(0), -1)
    print(torch.mean(samples, 0))
    # samples = np.array(samples)
    samples = samples.numpy()

    # print("mean")
    # print(np.mean(samples, axis=0))
    # print("covariance")
    # print(np.cov(samples.T))
    # # print(torch.std(samples, 0))
    #
    # fig = plt.figure(0)
    # plt.title('Dynamics Sampling')
    # plt.xlabel('x')
    # plt.ylabel('y')
    #
    # x = samples[:, 0]
    # y = samples[:, 1]
    # plt.scatter(x, y, c='red', marker='+')
    #
    #
    # mu = np.array([2, 2])
    # Sigma = np.array([[1, 0.8], [0.8, 1]])
    #
    # x, y = np.random.multivariate_normal(mu, Sigma, 1000).T
    # print("true covariance:")
    # s = [x, y]
    # print(np.cov(s))
    # plt.scatter(x, y, c='green', marker='*')
    # plt.show()




if __name__ == '__main__':
    # grad_test()
    vis_test()








