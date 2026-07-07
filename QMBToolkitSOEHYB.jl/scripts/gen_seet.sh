#!/bin/bash

n_spin_impurity=2
n_orbital_impurity=2

beta=100
n_tau=1000
n_matsubara=1000
dlr_eps_counterterms=1e-3
dlr_eps_hybridization=1e-12
hyb_type="from_seet"
hyb_file="./seet_ed/ed.1.input.h5"
init_dlr_euv=2.0
soe_subroutine_counterterms="ESPRIT"
n_soe_counterterms=0
# soe_subroutine_hyb="DLR"
soe_subroutine_hyb="ESPRIT"
n_soe_hybridization=7
do_subtraction_trick=true

# Extract the exponent from dlr_eps_hybridization (e.g., 1e-12 -> 12)
# dlr_eps_hyb_exp=$(echo $dlr_eps_hybridization | sed 's/1e-//')

# output_dir="../examples/spinless_semicircular/beta${beta}/ESPRIT${n_soe_counterterms}_DLR-${dlr_eps_hyb_exp}"
n_soe_total=$((n_soe_hybridization + n_soe_counterterms))
output_dir="./"
hopping_file="${output_dir}/hopping.txt"
counterterm_exe="$HOME/src/QMBToolkit/QMBToolkitSOEHYB.jl/scripts/gen_soe_counterterms_hyb.jl"

mkdir -p $output_dir

# julia gen_interaction_hopping.jl \
#     --n_spin_impurity $n_spin_impurity \
#     --n_orbital_impurity $n_orbital_impurity \
#     --t $t \
#     --U $U \
#     --J $J \
#     --Uprime $Uprime \
#     --mu $mu \
#     --output_dir $output_dir

# get SOE for counterterms and hybridization function

julia $counterterm_exe \
    --beta $beta \
    --n_spin_impurity $n_spin_impurity \
    --n_orbital_impurity $n_orbital_impurity \
    --n_tau $n_tau \
    --n_matsubara $n_matsubara \
    --dlr_eps_counterterms $dlr_eps_counterterms \
    --dlr_eps_hybridization $dlr_eps_hybridization \
    --hyb_type $hyb_type \
    --hyb_file $hyb_file \
    --init_dlr_euv $init_dlr_euv \
    --hopping_file $hopping_file \
    --soe_subroutine_counterterms $soe_subroutine_counterterms \
    --n_soe_counterterms $n_soe_counterterms \
    --soe_subroutine_hyb $soe_subroutine_hyb \
    --n_soe_hybridization $n_soe_hybridization \
    --output_dir $output_dir \
    --do_subtraction_trick $do_subtraction_trick


