#!/usr/bin/env julia
"""
Convert text format (Uijkl.txt and hopping.txt) back to ED HDF5 input file.
This is the reverse of convert_ed_h5_to_txt.jl.
"""

using ArgParse
using HDF5

"""
Read Coulomb tensor from Uijkl.txt.

Format:
- First line: number of non-zero elements
- Following lines: index i j k l Re(U) Im(U)
  (all indices 0-based)

Returns U_out in the output convention:
  H = 0.5 Σ U[i,j,k,l] c†_i c†_k c_l c_j
"""
function read_coulomb_tensor_txt(filename::String, n_imp_flavors::Int)
    U_out = zeros(ComplexF64, n_imp_flavors, n_imp_flavors, n_imp_flavors, n_imp_flavors)

    open(filename, "r") do io
        n_nonzero = parse(Int, strip(readline(io)))
        for _ in 1:n_nonzero
            parts = split(strip(readline(io)))
            # Format: idx\ti j k l re im  (tab-separated index, then space-separated)
            # After split on whitespace: [idx, i, j, k, l, re, im]
            i = parse(Int, parts[2]) + 1  # 0-based -> 1-based
            j = parse(Int, parts[3]) + 1
            k = parse(Int, parts[4]) + 1
            l = parse(Int, parts[5]) + 1
            re = parse(Float64, parts[6])
            im = parse(Float64, parts[7])
            U_out[i, j, k, l] = complex(re, im)
        end
    end

    println("Read $(count(!iszero, U_out)) non-zero U elements from $filename")
    return U_out
end

"""
Reverse the convention swap to recover U_h5.

Output convention: H = 0.5 Σ U_out[i,j,k,l] c†_i c†_k c_l c_j
H5 convention:     H = 0.5 Σ U_h5[i,j,k,l] c†_i c†_j c_l c_k

Forward was: U_out[i,j,k,l] = U_h5[i,k,j,l]
Reverse:     U_h5[i,j,k,l] = U_out[i,k,j,l]
"""
function reverse_tensor_convention(U_out::Array{ComplexF64,4})
    n = size(U_out, 1)
    U_h5 = zeros(ComplexF64, n, n, n, n)
    for i in 1:n, j in 1:n, k in 1:n, l in 1:n
        U_h5[i, j, k, l] = U_out[i, k, j, l]
    end
    return U_h5
end

"""
Convert flat flavor Coulomb tensor U_h5 back to 6D spin-orbital tensor
and write to H5 file as interaction/values.

Flavor indexing (1-based):
  flavor = n_spin * (orb - 1) + spin

The Coulomb interaction is spin-independent: int_tensor[s1,s2,o1,o2,o3,o4] = V[o1,o2,o3,o4].
However, U_h5 has Pauli-excluded zeros where f1=f2 or f3=f4 (same flavor on two
creation or two annihilation operators). This affects same-spin elements where o1=o2
or o3=o4 (since f1=f2 when s1=s2 and o1=o2).

Fix: extract V[o1,o2,o3,o4] from the different-spin channel (s1≠s2), which is never
Pauli-excluded (f1≠f2 guaranteed when spins differ). Then fill all spin combinations.
"""
function write_interaction_h5(fh5, U_h5::Array{ComplexF64,4}, n_orb::Int, n_spin::Int)
    # Extract spin-independent V[o1,o2,o3,o4] from different-spin channel
    # s1=1, s2=2 guarantees f1≠f2 (different spins → different flavors)
    V = zeros(Float64, n_orb, n_orb, n_orb, n_orb)
    s1, s2 = 1, 2
    for o1 in 1:n_orb, o2 in 1:n_orb, o3 in 1:n_orb, o4 in 1:n_orb
        f1 = n_spin * (o1 - 1) + s1
        f2 = n_spin * (o2 - 1) + s2
        f3 = n_spin * (o3 - 1) + s1
        f4 = n_spin * (o4 - 1) + s2
        V[o1, o2, o3, o4] = real(U_h5[f1, f2, f3, f4])
    end

    # Fill 6D tensor for ALL spin combinations (interaction is spin-independent)
    int_tensor = zeros(Float64, n_spin, n_spin, n_orb, n_orb, n_orb, n_orb)
    for s1 in 1:n_spin, s2 in 1:n_spin
        for o1 in 1:n_orb, o2 in 1:n_orb, o3 in 1:n_orb, o4 in 1:n_orb
            int_tensor[s1, s2, o1, o2, o3, o4] = V[o1, o2, o3, o4]
        end
    end

    # Julia writes column-major; Python/h5py reads row-major (reversed dims).
    # Forward read: Julia raw shape (o4,o3,o2,o1,s2,s1), then permutedims(6,5,4,3,2,1) -> (s1,s2,o1,o2,o3,o4).
    # Reverse write: permutedims(6,5,4,3,2,1) on (s1,s2,o1,o2,o3,o4) -> (o4,o3,o2,o1,s2,s1), write that.
    int_raw = permutedims(int_tensor, (6, 5, 4, 3, 2, 1))
    fh5["interaction/values"] = int_raw

    println("Written interaction/values with shape (Python): ($n_spin, $n_spin, $n_orb, $n_orb, $n_orb, $n_orb)")
end

"""
Read hopping matrix from hopping.txt.

Format:
- n_total^2 lines
- Each line: i j Re(H_ij) Im(H_ij)  (indices 0-based, j runs first)
"""
function read_hopping_matrix_txt(filename::String)
    lines = readlines(filename)
    # Filter out empty lines
    lines = filter(!isempty, lines)
    n_total = isqrt(length(lines))
    @assert n_total^2 == length(lines) "hopping.txt should have n^2 lines, got $(length(lines))"

    H = zeros(ComplexF64, n_total, n_total)
    for line in lines
        parts = split(strip(line))
        i = parse(Int, parts[1]) + 1  # 0-based -> 1-based
        j = parse(Int, parts[2]) + 1
        re = parse(Float64, parts[3])
        im = parse(Float64, parts[4])
        H[i, j] = complex(re, im)
    end

    println("Read $(n_total)x$(n_total) hopping matrix from $filename")
    return H, n_total
end

"""
Decompose flat hopping matrix back into H0_x, Epsk, Vk_x and write to H5.

Flavor indexing (0-based):
- Impurity: flavor = n_spin * orb + spin
- Bath:     flavor = n_imp_flavors + n_spin * bath + spin

Hopping matrix structure:
  [ H_imp   V     ]
  [ V†      Epsk  ]
"""
function write_hopping_h5(fh5, H::Array{ComplexF64,2}, n_orb::Int, n_spin::Int)
    n_total = size(H, 1)
    n_imp_flavors = n_spin * n_orb
    n_bath_flavors = n_total - n_imp_flavors
    n_bath = n_bath_flavors ÷ n_spin
    @assert n_bath * n_spin == n_bath_flavors "Bath flavors ($n_bath_flavors) not divisible by n_spin ($n_spin)"

    # Extract and write H0_x (impurity on-site/hopping)
    # H0_{orb_from}[orb_to, spin] = H[n_spin*orb_from + spin + 1, n_spin*orb_to + spin + 1]
    for orb_from in 0:(n_orb-1)
        h0 = zeros(Float64, n_orb, n_spin)  # (orbital_to, spin)
        for orb_to in 0:(n_orb-1)
            for s in 0:(n_spin-1)
                f_from = n_spin * orb_from + s + 1
                f_to = n_spin * orb_to + s + 1
                h0[orb_to+1, s+1] = real(H[f_from, f_to])
            end
        end
        # Julia write: shape (orbital_to, spin) -> permutedims(2,1) -> (spin, orbital_to)
        # So Python sees (orbital_to, spin)
        h0_raw = permutedims(h0, (2, 1))
        fh5["H0_$(orb_from)/values"] = h0_raw

        println("Written H0_$(orb_from)/values with shape (Python): ($n_orb, $n_spin)")
    end

    # For n_bath=0, pad with 1 bath per orbital (all zeros) so edlib sees non-empty arrays.
    # A bath with zero coupling (Vk=0) is physically equivalent to no bath.
    n_bath_write = max(n_bath, n_orb)

    # Extract and write Bath/Epsk (bath energies, diagonal)
    # Epsk[bath, spin] = H[n_imp + n_spin*bath + spin + 1, n_imp + n_spin*bath + spin + 1]
    epsk = zeros(Float64, n_bath_write, n_spin)
    for b in 0:(n_bath-1)
        for s in 0:(n_spin-1)
            f_bath = n_imp_flavors + n_spin * b + s + 1
            epsk[b+1, s+1] = real(H[f_bath, f_bath])
        end
    end
    epsk_raw = permutedims(epsk, (2, 1))  # (spin, bath) for Julia write -> Python sees (bath, spin)
    fh5["Bath/Epsk/values"] = epsk_raw

    println("Written Bath/Epsk/values with shape (Python): ($n_bath_write, $n_spin)")

    # Extract and write Bath/Vk_x (hybridization from bath to impurity orbital x)
    # Vk_{orb}[bath, spin] = H[n_spin*orb + spin + 1, n_imp + n_spin*bath + spin + 1]
    for orb in 0:(n_orb-1)
        vk = zeros(Float64, n_bath_write, n_spin)
        for b in 0:(n_bath-1)
            for s in 0:(n_spin-1)
                f_imp = n_spin * orb + s + 1
                f_bath = n_imp_flavors + n_spin * b + s + 1
                vk[b+1, s+1] = real(H[f_imp, f_bath])
            end
        end
        vk_raw = permutedims(vk, (2, 1))  # (spin, bath) -> Python sees (bath, spin)
        fh5["Bath/Vk_$(orb)/values"] = vk_raw

        println("Written Bath/Vk_$(orb)/values with shape (Python): ($n_bath_write, $n_spin)")
    end
end

"""
Main function to convert text files to ED H5 file.
"""
function convert_txt_to_ed_h5(uijkl_file::String, hopping_file::String,
                               output_file::String, seet_reference_file::String;
                               n_orb::Int=2, n_spin::Int=2, mu::Float64=0.0)
    @assert isfile(uijkl_file) "Uijkl file not found: $uijkl_file"
    @assert isfile(hopping_file) "Hopping file not found: $hopping_file"

    # Ensure output directory exists
    mkpath(dirname(output_file))

    println("Converting to $output_file")
    println("  uijkl_file = $uijkl_file")
    println("  hopping_file = $hopping_file")
    println("  n_orb = $n_orb, n_spin = $n_spin, mu = $mu")

    n_imp_flavors = n_spin * n_orb

    # Read and convert Coulomb tensor
    println("\nProcessing Uijkl.txt...")
    U_out = read_coulomb_tensor_txt(uijkl_file, n_imp_flavors)
    U_h5 = reverse_tensor_convention(U_out)

    # Read hopping matrix
    println("\nProcessing hopping.txt...")
    H, n_total = read_hopping_matrix_txt(hopping_file)

    # Write H5 file
    println("\nWriting $output_file...")
    h5open(output_file, "w") do fh5
        write_interaction_h5(fh5, U_h5, n_orb, n_spin)
        write_hopping_h5(fh5, H, n_orb, n_spin)
        fh5["mu"] = mu
        println("Written mu = $mu")

        # Write GreensFunction_orbitals/values: orbital pairs for GF computation
        # Format: each row is (orb_i, orb_j) pair. Include all off-diagonal pairs.
        # SEET uses [[0,1],[1,0]] for 2-orbital case.
        gf_orbs = Int32[]
        for i in 0:(n_orb-1), j in 0:(n_orb-1)
            if i != j
                push!(gf_orbs, Int32(i), Int32(j))
            end
        end
        gf_orbs_matrix = permutedims(reshape(gf_orbs, 2, :), (2, 1))
        fh5["GreensFunction_orbitals/values"] = gf_orbs_matrix
        println("Written GreensFunction_orbitals/values")

        # Copy freq, Delta, G0_imp, G_imp from SEET reference file
        h5open(seet_reference_file, "r") do ref
            freq_data = read(ref, "freq")
            fh5["freq"] = freq_data
            println("Written freq ($(length(freq_data)) points) from reference file")

            # Copy Delta group (data, data_in, static). The new embedding.exe
            # only writes data_in/static (no data), while the old ed_input_generator
            # wrote all three. Skip whichever keys are not present in the reference.
            for dset_name in ["Delta/data", "Delta/data_in", "Delta/static"]
                if haskey(ref, dset_name)
                    fh5[dset_name] = read(ref, dset_name)
                    println("Written $dset_name from reference file")
                else
                    println("Skipped $dset_name (not in reference file)")
                end
            end

            # Copy G0_imp/data
            fh5["G0_imp/data"] = read(ref, "G0_imp/data")
            println("Written G0_imp/data from reference file")

            # Copy G_imp/data
            fh5["G_imp/data"] = read(ref, "G_imp/data")
            println("Written G_imp/data from reference file")
        end
    end

    n_bath = (n_total - n_imp_flavors) ÷ n_spin
    println("\nConversion complete!")
    println("  Output: $output_file")
    println("  Total flavors: $n_total (impurity: $n_imp_flavors, bath sites: $n_bath)")
end

function parse_commandline()
    s = ArgParseSettings(description="Convert text format (Uijkl.txt and hopping.txt) back to ED HDF5 input file.")

    @add_arg_table! s begin
        "--uijkl_file"
            help = "path to the Uijkl.txt interaction file"
            required = true
        "--hopping_file"
            help = "path to the hopping.txt file"
            required = true
        "--output_h5_file"
            help = "path to the output HDF5 file"
            required = true
        "--seet_reference_file"
            help = "path to SEET reference H5 file (for freq grid, Delta, G0_imp, G_imp)"
            required = true
        "--n_orb"
            help = "number of orbitals"
            arg_type = Int
            default = 2
        "--n_spin"
            help = "number of spin components"
            arg_type = Int
            default = 2
        "--mu"
            help = "chemical potential"
            arg_type = Float64
            default = 0.0
    end

    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    convert_txt_to_ed_h5(
        args["uijkl_file"],
        args["hopping_file"],
        args["output_h5_file"],
        args["seet_reference_file"];
        n_orb=args["n_orb"],
        n_spin=args["n_spin"],
        mu=args["mu"],
    )
end
