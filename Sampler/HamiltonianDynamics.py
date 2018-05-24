import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def kinetic_energy(velocity):
    """
    calculate kinetic nergy 0.5 * vel^2
    :param velocity: Variable [batch, 1, height, width]
    :return: a 1D torch.tensor with length of batch_size
    """
    temp = (velocity.data)**2
    temp = torch.sum(temp, dim=1)
    temp = torch.sum(temp, dim=1)
    temp = torch.sum(temp, dim=1)
    return 0.5 * temp.data

# compute energy: U+K
def hamiltonian(pos, vel, energy_fn):
    """
    energy = potention + kinetic
    :param pos: Variable [batch, 1, height, width]
    :param vel: Variable [batch, 1, height, width]
    :param energy_fn: energy function
    :return: a 1D tensor with length of batch_size
    """
    U = energy_fn(pos)  # 1D Variable with length of batch_size
    U = U.data
    K = kinetic_energy(vel) # 1D tensor with length of batch_size
    return U + K

def metropolis_hastings_accept(energy_prev, energy_next):
    """
    Performs a Metropolis-Hastings accept-reject move.
    :param energy_prev: 1D torch.tensor of size batch_size
    :param energy_next: 1D torch.tensor of size batch_size
    :return: 1D torch.ByteTensor of size batch_size. if true --> accept
    """
    batch_size = len(energy_prev)
    energy_diff = energy_prev - energy_next
    ones = torch.ones(batch_size)
    energy_diff = torch.min(energy_diff, ones)
    rnd = torch.rand(batch_size)

    return (torch.exp(energy_diff) - rnd) > 0

# obtain a single sample for evenry batch after leapfrog(n_steps)
def simulate_dynamic(initial_pos, initial_vel, step_size, n_steps, energy_fn):
    """
    obtain the final (position, velocity) after n_steps leapfrog
    :param initial_pos: Variable [batch, 1, height, width]
    :param initial_vel: Variable [batch, 1, height, width[
    :param step_size:
    :param n_steps: leapfrog iteration
    :param energy_fn:
    :return: final_position, final_velocity.
            [batch, 1, height, width], [batch, 1, height, width]
    """
    def leapfrog(pos, vel, step):
        """
        one(half?) step leap frog
        :param pos: Variable [batch, 1, height, width]
                    position at time t
        :param vel: Variable [batch, 1, height, width]
                    velocity at time (t-stepsize/2)
        :param step: scalar
        :return: new_pos, new_vel
                position at time (t+stepsize)
                velocity at time (t+stepsize/2)
        """
        if not pos.requires_grad:
            pos.requires_grad = True
            pos.volatile = False
        potential_of_give_pos = energy_fn(pos)
        potential_of_give_pos.backward()
        dE_dpos = pos.grad

        new_vel = vel - step * dE_dpos  # at time (t + stepsize/2)
        new_pos = pos + step * new_vel  # at time (t + stepsize)
        return new_pos, new_vel

    initial_potential = energy_fn(initial_pos)
    initial_potential.backward()
    dE_dpos = initial_pos.grad
    # velocity at time t + stepsize / 2
    vel_half_step = initial_vel - 0.5 * step_size * dE_dpos

    # position at time t + stepsize
    pos_full_step = initial_pos + step_size * vel_half_step

    # perform leapfrog iteration
    temp_pos = pos_full_step
    temp_vel = vel_half_step
    for lf_step in range(n_steps):
        temp_pos, temp_vel = leapfrog(temp_pos, temp_vel, step_size)

    final_pos = temp_pos
    final_vel = temp_vel

    final_pos.requires_grad = True
    final_pos.volatile = False
    potential = energy_fn(final_pos)
    potential.backward()
    final_vel = final_vel - 0.5 * step_size * final_pos.grad

    return final_pos, final_vel


def hmc_move(positions, energy_fn, step_size, n_steps):
    """
    Perform one iteration of sampling
    1. Start by sampling a random velocity from a Gaussian.
    2. Perform n_steps leapfrog
    3. decide whether to accept or reject
    :param positions: torch.tensor [batch, 1, height, width]
    :param energy_fn: torch.tensor [batch, 1, height, width]
    :param step_size: leapfrog step size
    :param n_steps:   leapfrog iteration times
    :return: accept(torch.ByteTensor of size batch_size)
             final_pos(torch.tensor of size [batch, 1, height, width]
    """
    # random initial velocity from normal distribution
    initial_vel = torch.randn(positions.size()) # batch, 1, height, width

    positions = Variable(positions, requires_grad=True)
    initial_vel = Variable(initial_vel, requires_grad=True)

    final_pos, final_vel = simulate_dynamic(initial_pos=positions,
                                            initial_vel=initial_vel,
                                            step_size=step_size,
                                            n_steps=n_steps,
                                            energy_fn=energy_fn
                                            )

    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(positions, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn)
    )

    return accept, final_pos.data

def hmc_sampling(init_pos, energy_fn, n_samples, step_size=0.01, n_steps=20, gap=20):
    """
    sample n_samples sample from given energy-based distribution
    :param init_pos: torch.tensor [batch, 1, height, width]
    :param energy_fn:
    :param n_samples:
    :param step_size:
    :param n_steps:
    :param gap:
    :return: a list of tensor, each element [1, 1, height, width]
    """
    result_samples = []
    result_samples.append(init_pos)

    for i in range(1, n_samples+gap):
        last_pos = result_samples[-1]
        accept, new_pos = hmc_move(last_pos, energy_fn, step_size, n_steps)

        # accept: 1D ByteTensor of size batch_size
        temp = accept.float() * new_pos + (1 - accept.float()) * last_pos
        result_samples.append(temp)

def test():
    n_samples = 1000
    stepsize = 0.1
    n_steps = 20

    height = 32
    width = 32
    batch_size = 8

    # torch.tensor [batch, height, width]
    initial_pos = torch.randn(batch_size, height, width)
