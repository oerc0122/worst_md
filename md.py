"""
The worst MD code you have ever seen

Inefficient, slow, bad
Implements Lennard-Jones NVE dynamics
No PBCs
Box just determines initial configuration
"""

from random import random
from math import sqrt


def init(box: list[float] = (10, 10, 10), natoms: int = 50) -> tuple[list[list[float]],
                                                                     list[list[float]],
                                                                     list[list[float]],
                                                                     list[float]]:
    """ Build box with atoms """
    positions = [[random()*box[i] for i in range(3)] for _ in range(natoms)]
    velocity = [[0.0 for _ in range(3)] for _ in range(natoms)]
    acceleration = [[0.0 for _ in range(3)] for _ in range(natoms)]
    mass = [2.0 for _ in range(natoms)]

    return positions, velocity, acceleration, mass


def pot(pos_diff: float) -> float:
    """ Lennard-Jones potential, eps/sig = 1 """
    return (pos_diff**-6) - (pos_diff**-12)


def d_pot(pos_diff: float) -> float:
    """ Derivative of LJ for forces """
    return -6*(pos_diff**-7) + 12*(pos_diff**-13)


def compute(position, velocity, mass):
    """
    Compute energies and forces
    """

    potential_e = kinetic_e = 0

    force = [[0 for _ in range(3)] for _ in range(len(position))]

    for i, r_i in enumerate(position):

        for j, r_j in enumerate(position):

            if i != j:
                r_ij = [ri - rj for ri, rj in zip(r_i, r_j)]

                pos_diff = sqrt(sum(p**2 for p in r_ij))
                potential_e += 0.5 * pot(pos_diff)
                force[i] = [f - rij*d_pot(pos_diff) / pos_diff
                            for f, rij in zip(force[i], r_ij)]

        kinetic_e += sum(0.5*mass[i]*v_i*v_i for v_i in velocity[i])

    return potential_e, kinetic_e, force


def vel_verlet(position, velocity, acceleration, force, mass, timestep):
    """
    Move atoms
    """

    for i, _ in enumerate(position):
        for dim in range(3):
            position[i][dim] += ((velocity[i][dim] * timestep) +
                                 (0.5 * timestep * timestep * acceleration[i][dim]))
            velocity[i][dim] += ((0.5 * timestep * force[i][dim] / mass[i]) +
                                 (0.5 * timestep * acceleration[i][dim]))
            acceleration[i][dim] = force[i][dim] / mass[i]

    return position, velocity, acceleration


def main(natoms: int, nsteps: int, timestep: float) -> None:
    """
    Run a calculation
    """

    pos, vel, acc, mass = init(natoms=natoms)

    potential_e, kinetic_e, force = compute(pos, vel, mass)

    total_e = potential_e + kinetic_e

    with open('output.dat', 'w', encoding='utf-8') as out_file:

        for i in range(nsteps):
            if not i % 10:
                print(f"Step {i}/{nsteps}")
            potential_e, kinetic_e, force = compute(pos, vel, mass)
            # PE, KE, Loss
            print(potential_e, kinetic_e, (potential_e+kinetic_e-total_e)/total_e, file=out_file)

            pos, vel, acc = vel_verlet(pos, vel, acc, force, mass, timestep)


if __name__ == "__main__":
    main(50, 1000, 0.5)
