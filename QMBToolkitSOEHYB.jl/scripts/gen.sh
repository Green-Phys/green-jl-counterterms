#!/bin/bash


# build interaction tensor and hopping matrix
n_spin_impurity=1
n_orbital_impurity=1
t=0.0
U=2.0 
J=0.2
Uprime=$(echo "$U-2.0*$J" | bc -l)
mu=$(echo "(3.0 * $U - 5.0 * $J - 3.0) / 2.0" | bc -l) 
output_dir="../examples/spinless_semi_circular/beta10/ESPRI2_DLR-3"

mkdir -p $output_dir

julia gen_interaction_hopping.jl \
    --n_spin_impurity $n_spin_impurity \
    --n_orbital_impurity $n_orbital_impurity \
    --t $t \
    --U $U \
    --J $J \
    --Uprime $Uprime \
    --mu $mu \
    --output_dir $output_dir

# get SOE for counterterms and hybrdization function
beta=10
n_tau=1000
n_matsubara=1000
dlr_eps_counterterms=1e-3
dlr_eps_hybridization=1e-3
hyb_type="SemiCircular"
hopping_file="${output_dir}/hopping.txt"
soe_subroutine_counterterms="ESPRIT"
n_soe_counterterms=2
soe_subroutine_hyb="DLR"
n_soe_hybridization=7

julia gen_soe_counterterms_hyb.jl \
    --beta $beta \
    --n_spin_impurity $n_spin_impurity \
    --n_orbital_impurity $n_orbital_impurity \
    --n_tau $n_tau \
    --n_matsubara $n_matsubara \
    --dlr_eps_counterterms $dlr_eps_counterterms \
    --dlr_eps_hybridization $dlr_eps_hybridization \
    --hyb_type $hyb_type \
    --hopping_file $hopping_file \
    --soe_subroutine_counterterms $soe_subroutine_counterterms \
    --n_soe_counterterms $n_soe_counterterms \
    --soe_subroutine_hyb $soe_subroutine_hyb \
    --n_soe_hybridization $n_soe_hybridization \
    --output_dir $output_dir


