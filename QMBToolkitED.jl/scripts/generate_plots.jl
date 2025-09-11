using Pkg
Pkg.develop(path="..")

using QMBToolkitED
using Plots
using LaTeXStrings

# 1. Define the Model
t = 1.0  # Hopping strength
U = 2.0  # On-site interaction
beta = 20.0 # Inverse temperature
n_matsubara = 100 # Number of Matsubara frequencies
n_tau = 100 # Number of imaginary time points

# 2. Construct the Hamiltonian
hopping_str = """
0 0 $(-U/2) 0.0
1 1 $(-U/2) 0.0
2 2 $(-U/2) 0.0
3 3 $(-U/2) 0.0
0 2 $(-t) 0.0
2 0 $(-t) 0.0
1 3 $(-t) 0.0
3 1 $(-t) 0.0
"""

coulomb_str = """
4
0 0 1 1 0 $(U) 0.0
1 1 0 0 1 $(U) 0.0
2 2 3 3 2 $(U) 0.0
3 3 2 2 3 $(U) 0.0
"""

H, annh = build_hamiltonian(IOBuffer(hopping_str), IOBuffer(coulomb_str));

# 3. Solve the System
evals, evecs = solve_hamiltonian(H);

# 4. Calculate Green's Functions
iwn, g11_matsubara = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 1);
_,   g12_matsubara = calculate_green_matsubara(evals, evecs, annh, beta, n_matsubara, 1, 3);
tau_grid, g11_tau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 1);
_,        g12_tau = calculate_green_tau(evals, evecs, annh, beta, n_tau, 1, 3);

# 5. Plot Matsubara Green's Function
p1 = plot(imag.(iwn), real.(g11_matsubara), title=L"G_{11}(i\omega_n)", label="Re", xlabel=L"\omega_n")
plot!(p1, imag.(iwn), imag.(g11_matsubara), label="Im")
p2 = plot(imag.(iwn), real.(g12_matsubara), title=L"G_{12}(i\omega_n)", label="Re", xlabel=L"\omega_n")
plot!(p2, imag.(iwn), imag.(g12_matsubara), label="Im")
matsubara_plot = plot(p1, p2, layout=(1, 2), size=(800, 300))
savefig(matsubara_plot, "docs/src/assets/matsubara_plot.png")

# 6. Plot Imaginary-Time Green's Function
p3 = plot(tau_grid, real.(g11_tau), title=L"G_{11}(\tau)", label="", xlabel=L"\tau")
p4 = plot(tau_grid, real.(g12_tau), title=L"G_{12}(\tau)", label="", xlabel=L"\tau")
tau_plot = plot(p3, p4, layout=(1, 2), size=(800, 300))
savefig(tau_plot, "docs/src/assets/tau_plot.png")

println("Plots saved to docs/src/assets/")
