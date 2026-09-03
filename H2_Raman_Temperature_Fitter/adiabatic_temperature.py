import cantera as ct

phis = [1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]

gas = ct.Solution("gri30.yaml")

print("Phi\tAdiabatic Flame Temperature (K)")

for phi in phis:

    # Initial reactant conditions
    gas.TP = 300.0, ct.one_atm

    # Set H2-air mixture at specified equivalence ratio
    gas.set_equivalence_ratio(
        phi,
        fuel="H2:1",
        oxidizer="O2:1, N2:3.76"
    )

    # Adiabatic, constant-pressure equilibrium
    gas.equilibrate("HP")

    print(f"{phi:.1f}\t{gas.T:.1f}")