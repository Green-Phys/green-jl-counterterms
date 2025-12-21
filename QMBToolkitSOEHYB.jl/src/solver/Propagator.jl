"""
Propagator types for time-dependent operators.

Provides wrapper types around Array{T,3} for proper abstraction:
- DensePropagator: Trivial wrapper around Array{T,3}
- BlockDensePropagator: Block-sparse time-dependent array with quantum number conservation

These types encapsulate propagator data and provide a consistent interface for
DLR operations (convolution, reflection, tau2dlr, dlr2tau).
"""

using QMBToolkitED.BlockDense

#=============================================================================
# DensePropagator - trivial wrapper around Array{T,3}
=============================================================================#

"""
    DensePropagator{T<:Number}

Time-dependent dense propagator. A wrapper around Array{T,3} where:
- First dimension: tau grid points
- Second dimension: row index (Fock space dimension)
- Third dimension: column index (Fock space dimension)

This is a trivial wrapper that provides consistent interface with BlockDensePropagator.
"""
struct DensePropagator{T<:Number}
    data::Array{T,3}
end

# Constructors
DensePropagator(data::Array{T,3}) where T = DensePropagator{T}(data)

function Base.zeros(::Type{DensePropagator{T}}, n_tau::Int, dim1::Int, dim2::Int) where T
    return DensePropagator{T}(zeros(T, n_tau, dim1, dim2))
end

# Size and access
Base.size(p::DensePropagator) = size(p.data)
Base.size(p::DensePropagator, d::Int) = size(p.data, d)
Base.getindex(p::DensePropagator, args...) = getindex(p.data, args...)
Base.setindex!(p::DensePropagator, val, args...) = setindex!(p.data, val, args...)
Base.view(p::DensePropagator, args...) = view(p.data, args...)

# Arithmetic
function Base.:+(A::DensePropagator{T}, B::DensePropagator{T}) where T
    return DensePropagator{T}(A.data .+ B.data)
end

function Base.:-(A::DensePropagator{T}, B::DensePropagator{T}) where T
    return DensePropagator{T}(A.data .- B.data)
end

function Base.:*(α::Number, A::DensePropagator{T}) where T
    return DensePropagator{T}(α .* A.data)
end

Base.:*(A::DensePropagator{T}, α::Number) where T = α * A
Base.:/(A::DensePropagator{T}, α::Number) where T = (1/α) * A
Base.:-(A::DensePropagator{T}) where T = (-1) * A

function Base.copyto!(dest::DensePropagator{T}, src::DensePropagator{T}) where T
    copyto!(dest.data, src.data)
    return dest
end

function Base.fill!(A::DensePropagator{T}, value) where T
    fill!(A.data, value)
    return A
end

#=============================================================================
# BlockDensePropagator - block-sparse time-dependent array
=============================================================================#

"""
    BlockDensePropagator{T<:Number}

Time-dependent block-dense propagator for DLR operations.

Each block is stored as a 3D array: (time_points, row_dim, col_dim)
where time_points is either tau grid or DLR grid.

Fields:
- `basis::BlockDenseBasis`: Basis defining the block structure
- `flux::OperatorFlux`: Quantum number change (flux) of the operator
- `blocks::Dict{Tuple{Int,Int}, Array{T,3}}`: Blocks indexed by (target_sector, source_sector)
"""
struct BlockDensePropagator{T<:Number}
    basis::BlockDenseBasis
    flux::OperatorFlux
    blocks::Dict{Tuple{Int,Int}, Array{T,3}}
end

# Constructor
function BlockDensePropagator(basis::BlockDenseBasis, flux::OperatorFlux, blocks::Dict{Tuple{Int,Int}, Array{T,3}}) where T
    return BlockDensePropagator{T}(basis, flux, blocks)
end

"""
    Base.zeros(::Type{BlockDensePropagator{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T

Create a zero BlockDensePropagator with specified basis, flux, and time dimension.

Allocates zero blocks for all sector pairs allowed by the flux.
"""
function Base.zeros(::Type{BlockDensePropagator{T}}, basis::BlockDenseBasis, flux::OperatorFlux, n_tau::Int) where T
    blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    # For zero flux, allocate diagonal blocks only
    if flux.delta_qn == zero_qn(typeof(basis.sectors[1].qn))
        # Diagonal blocks only
        for (i, sector) in enumerate(basis.sectors)
            dim = sector.dim
            blocks[(i, i)] = zeros(T, n_tau, dim, dim)
        end
    else
        # For non-zero flux, need to determine allowed blocks based on quantum number arithmetic
        # Currently just create empty dict - will be populated as needed
    end

    return BlockDensePropagator{T}(basis, flux, blocks)
end

#=============================================================================
# BlockDensePropagator Arithmetic Operations
=============================================================================#

"""
    Base.:+(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T

Add two BlockDensePropagator with the same flux.
"""
function Base.:+(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T
    @assert A.basis === B.basis "Arrays must have same basis"
    @assert A.flux.delta_qn == B.flux.delta_qn "Arrays must have same flux to be added"

    # Start with copy of A's blocks
    result_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, A_block) in A.blocks
        result_blocks[key] = copy(A_block)
    end

    # Add B's blocks
    for (key, B_block) in B.blocks
        if haskey(result_blocks, key)
            result_blocks[key] .+= B_block
        else
            result_blocks[key] = copy(B_block)
        end
    end

    return BlockDensePropagator{T}(A.basis, A.flux, result_blocks)
end

"""
    Base.:-(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T

Subtract two BlockDensePropagator with the same flux.
"""
function Base.:-(A::BlockDensePropagator{T}, B::BlockDensePropagator{T}) where T
    return A + (-1) * B
end

"""
    Base.:*(α::Number, A::BlockDensePropagator{T}) where T

Multiply BlockDensePropagator by scalar.
"""
function Base.:*(α::Number, A::BlockDensePropagator{T}) where T
    if iszero(α)
        # Return zero propagator (no blocks)
        return BlockDensePropagator{T}(A.basis, A.flux, Dict{Tuple{Int,Int}, Array{T,3}}())
    end

    # Scale all blocks
    scaled_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()
    for (key, block) in A.blocks
        scaled_blocks[key] = α * block
    end

    return BlockDensePropagator{T}(A.basis, A.flux, scaled_blocks)
end

Base.:*(A::BlockDensePropagator{T}, α::Number) where T = α * A
Base.:/(A::BlockDensePropagator{T}, α::Number) where T = (1/α) * A
Base.:-(A::BlockDensePropagator{T}) where T = (-1) * A

"""
    Base.copyto!(dest::BlockDensePropagator{T}, src::BlockDensePropagator{T}) where T

Copy contents of src into dest.
"""
function Base.copyto!(dest::BlockDensePropagator{T}, src::BlockDensePropagator{T}) where T
    @assert dest.basis === src.basis "Arrays must have same basis"
    @assert dest.flux.delta_qn == src.flux.delta_qn "Arrays must have same flux"

    for (key, src_block) in src.blocks
        @assert haskey(dest.blocks, key) "Destination missing block $key"
        copyto!(dest.blocks[key], src_block)
    end

    return dest
end

"""
    Base.fill!(A::BlockDensePropagator{T}, value) where T

Fill all elements of all blocks with a value.
"""
function Base.fill!(A::BlockDensePropagator{T}, value) where T
    for (key, block) in A.blocks
        fill!(block, value)
    end
    return A
end

"""
    add_scaled!(dest::BlockDensePropagator{T}, α::Number, src::BlockDensePropagator{T}) where T

In-place operation: dest .+= α .* src
"""
function add_scaled!(dest::BlockDensePropagator{T}, α::Number, src::BlockDensePropagator{T}) where T
    @assert dest.basis === src.basis "Arrays must have same basis"
    @assert dest.flux.delta_qn == src.flux.delta_qn "Arrays must have same flux"

    if iszero(α)
        return dest
    end

    for (key, src_block) in src.blocks
        if haskey(dest.blocks, key)
            dest.blocks[key] .+= α .* src_block
        else
            dest.blocks[key] = α * src_block
        end
    end

    return dest
end

#=============================================================================
# BlockDensePropagator DLR Operations
=============================================================================#

# QMBToolkitDLR re-exports DLRGrid, dlr2tau, tau2dlr from Lehmann
using QMBToolkitDLR

"""
    tau2dlr_blockdense(dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform block-dense propagator from tau space to DLR representation.
Each block is transformed independently using standard tau2dlr.
"""
function tau2dlr_blockdense(dlr_grid::DLRGrid, prop::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T
    blocks_dlr = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_tau) in prop.blocks
        # Transform this block: (n_tau, n_row, n_col) → (r, n_row, n_col)
        block_dlr = tau2dlr(dlr_grid, block_tau, tau_grid)
        blocks_dlr[(target, source)] = block_dlr
    end

    return BlockDensePropagator{T}(prop.basis, prop.flux, blocks_dlr)
end

"""
    dlr2tau_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform block-dense propagator from DLR representation to tau space.
Each block is transformed independently using standard dlr2tau.
"""
function dlr2tau_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}, tau_grid::Vector{Float64}) where T
    blocks_tau = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_dlr) in prop_dlr.blocks
        # Transform this block: (r, n_row, n_col) → (n_tau, n_row, n_col)
        block_tau = dlr2tau(dlr_grid, block_dlr, tau_grid)
        blocks_tau[(target, source)] = block_tau
    end

    return BlockDensePropagator{T}(prop_dlr.basis, prop_dlr.flux, blocks_tau)
end

"""
    do_reflection_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}) where T

Perform reflection operation on block-dense propagator: F(τ) = H(β - τ).
Each block is reflected independently using standard DLR reflection.
"""
function do_reflection_blockdense(dlr_grid::DLRGrid, prop_dlr::BlockDensePropagator{T}) where T
    tau_grid_reflected = dlr_grid.beta .- dlr_grid.tau
    blocks_tau = Dict{Tuple{Int,Int}, Array{T,3}}()

    for ((target, source), block_dlr) in prop_dlr.blocks
        # Reflect this block
        block_tau = dlr2tau(dlr_grid, block_dlr, tau_grid_reflected)
        blocks_tau[(target, source)] = block_tau
    end

    return BlockDensePropagator{T}(prop_dlr.basis, prop_dlr.flux, blocks_tau)
end

"""
    do_reflection_tau2tau_blockdense(dlr_grid::DLRGrid, prop_tau::BlockDensePropagator{T}) where T

Perform reflection operation on tau-space BlockDense data: F(τ) = H(β - τ).
First transforms to DLR, then applies reflection.
"""
function do_reflection_tau2tau_blockdense(dlr_grid::DLRGrid, prop_tau::BlockDensePropagator{T}) where T
    prop_dlr = tau2dlr_blockdense(dlr_grid, prop_tau, dlr_grid.tau)
    return do_reflection_blockdense(dlr_grid, prop_dlr)
end

#=============================================================================
# BlockDensePropagator Convolution
=============================================================================#

"""
    BlockDenseConvolutionWorkspace{T}

Workspace for block-dense convolution operations.
"""
struct BlockDenseConvolutionWorkspace{T<:Number}
    basis::BlockDenseBasis
    flux_result::OperatorFlux
    block_workspaces::Dict{Tuple{Int,Int}, ConvolutionWorkspace{T}}
end

"""
    create_blockdense_convolution_workspace(basis::BlockDenseBasis, flux_F::OperatorFlux, flux_G::OperatorFlux, r::Int, n_tau::Int, ::Type{T}) where T

Create workspace for block-dense convolution.
"""
function create_blockdense_convolution_workspace(basis::BlockDenseBasis, flux_F::OperatorFlux, flux_G::OperatorFlux, r::Int, n_tau::Int, ::Type{T}) where T<:Number
    flux_result = OperatorFlux(combine_qn(flux_F.delta_qn, flux_G.delta_qn))
    block_workspaces = Dict{Tuple{Int,Int}, ConvolutionWorkspace{T}}()

    # Determine which output blocks exist and find maximum intermediate dimension
    for (target_idx, target_sector) in enumerate(basis.sectors)
        for (source_idx, source_sector) in enumerate(basis.sectors)
            # Check if this block can exist: target_qn = source_qn + flux_result
            if target_sector.qn == combine_qn(source_sector.qn, flux_result.delta_qn)
                # This block exists in the output
                target_dim = target_sector.dim
                source_dim = source_sector.dim

                # Find maximum intermediate dimension for this block
                max_intermediate_dim = 0
                for (intermediate_idx, intermediate_sector) in enumerate(basis.sectors)
                    if target_sector.qn == combine_qn(intermediate_sector.qn, flux_F.delta_qn) &&
                       intermediate_sector.qn == combine_qn(source_sector.qn, flux_G.delta_qn)
                        max_intermediate_dim = max(max_intermediate_dim, intermediate_sector.dim)
                    end
                end

                if max_intermediate_dim > 0
                    block_workspaces[(target_idx, source_idx)] = ConvolutionWorkspace{T}(r, target_dim, max_intermediate_dim, source_dim, n_tau)
                else
                    block_workspaces[(target_idx, source_idx)] = ConvolutionWorkspace{T}(r, target_dim, 1, source_dim, n_tau)
                end
            end
        end
    end

    return BlockDenseConvolutionWorkspace{T}(basis, flux_result, block_workspaces)
end

"""
    do_convolution_dlr2tau_blockdense!(workspace::BlockDenseConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDensePropagator{T}, G_dlr::BlockDensePropagator{T}) where T

Compute convolution H = F * G for block-dense propagators in DLR representation.
"""
function do_convolution_dlr2tau_blockdense!(workspace::BlockDenseConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::BlockDensePropagator{T}, G_dlr::BlockDensePropagator{T}) where T<:Number
    H_tau_blocks = Dict{Tuple{Int,Int}, Array{T,3}}()

    # Compute each output block
    for ((target_idx, source_idx), block_workspace) in workspace.block_workspaces
        target_sector = workspace.basis.sectors[target_idx]
        source_sector = workspace.basis.sectors[source_idx]

        H_block_tau = nothing

        for (intermediate_idx, intermediate_sector) in enumerate(workspace.basis.sectors)
            F_key = (target_idx, intermediate_idx)
            G_key = (intermediate_idx, source_idx)

            if haskey(F_dlr.blocks, F_key) && haskey(G_dlr.blocks, G_key)
                F_block = F_dlr.blocks[F_key]
                G_block = G_dlr.blocks[G_key]

                # Use dense convolution for this block pair
                block_contrib = do_convolution_dlr2tau!(block_workspace, dlr_grid, kernel_cache, F_block, G_block)

                if H_block_tau === nothing
                    H_block_tau = block_contrib
                else
                    H_block_tau .+= block_contrib
                end
            end
        end

        if H_block_tau !== nothing
            H_tau_blocks[(target_idx, source_idx)] = H_block_tau
        end
    end

    return BlockDensePropagator{T}(workspace.basis, workspace.flux_result, H_tau_blocks)
end
