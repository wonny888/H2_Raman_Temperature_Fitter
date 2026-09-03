phi = 1.6

# A: hydrogen entering
a = phi

# B: oxygen entering
b = 0.5

# C: nitrogen entering with the air
c = 1.88

# D: hydrogen that actually burns
# 0.5 O2 can burn at most 1.0 H2
d = min(a, 2.0 * b)

# Products
H2 = a - d          # unburned hydrogen
H2O = d             # burned H2 becomes H2O
O2 = b - 0.5 * d    # unused oxygen
N2 = c              # nitrogen mostly passes through

# Convert amounts into mole fractions
total = H2 + H2O + O2 + N2

H2conc = H2 / total
H2Oconc = H2O / total
O2conc = O2 / total
N2conc = N2 / total

print("H2 =", H2conc)
print("N2 =", N2conc)
print("H2O =", H2Oconc)
print("O2 =", O2conc)
print("Total =", H2conc + N2conc + H2Oconc + O2conc)