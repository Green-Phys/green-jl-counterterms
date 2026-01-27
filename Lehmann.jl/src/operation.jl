# function _tensor2matrix(tensor::AbstractArray{T,N}, axis) where {T,N}
#     # internal function to move the axis dim to the first index, then reshape the tensor into a matrix
#     dim = N
#     n1 = size(tensor)[axis]
#     partialsize = deleteat!(collect(size(tensor)), axis) # the size of the tensor except the axis-th dimension
#     n2 = reduce(*, partialsize)

#     if axis == 1 #no need to permutate the axis
#         return reshape(tensor, (n1, n2)), partialsize
#     elseif axis == 2 && dim == 2 #for matrix, simply transpose, no copy is created
#         return transpose(tensor), partialsize
#     else
#         permu = [i for i = 1:dim]
#         permu[1], permu[axis] = axis, 1
#         partialsize = collect(size(tensor)[permu][2:end])
#         ntensor = permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created 
#         # ntensor = nocopy ? PermutedDimsArray(tensor, permu) : permutedims(tensor, permu) # permutate the axis-th and the 1st dim
#         ntensor = reshape(ntensor, (n1, n2)) # no copy is created
#         return ntensor, partialsize
#     end
# end

# function _matrix2tensor(mat, partialsize, axis)
#     # internal function to reshape matrix to a tensor, then swap the first index with the axis-th dimension
#     @assert size(mat)[2] == reduce(*, partialsize) # total number of elements of mat and the tensor must match
#     tsize = vcat(size(mat)[1], partialsize)
#     tensor = reshape(mat, Tuple(tsize))
#     dim = length(partialsize) + 1

#     if axis == 1
#         return tensor
#     elseif axis == 2 && dim == 2
#         return transpose(tensor) #transpose do not create copy
#     else
#         permu = [i for i = 1:dim]
#         permu[1], permu[axis] = axis, 1
#         return permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created
#         # ntensor = nocopy ? PermutedDimsArray(tensor, permu) : permutedims(tensor, permu) # permutate the axis-th and the 1st dim
#         # return ntensor
#     end
# end

#replace one of the tuple elements. See https://discourse.julialang.org/t/computing-tuple-replacements/69581/10
@generated function _reset_tuple(t::NTuple{N,Int}, x, i) where {N}
    Expr(:tuple, (:(ifelse($j == i, x, t[$j])) for j = 1:N)...)
end

# @generated function _remove_tuple(t::NTuple{N,Int}, x, i) where {N}
#     Expr(:tuple, (:(ifelse($j == i, x, t[$j])) for j in 1:N)...)
# end

function _matrix_tensor_dot(
    mat::AbstractMatrix{TC},
    tensor::AbstractArray{T,N},
    axis::Int,
) where {T,TC,N}
    #calculate \sum_j mat[i, j]*tensor[..., j, ...]  where j is the axis-th dimension of tensor
    @assert 0 < axis <= N
    _n, _m = size(mat)
    _size = collect(size(tensor))
    @assert (_m == _size[axis]) "matrix size $(size(mat)) and tensor size ($(_size)) do not match at axis = $axis"
    _target_size = _reset_tuple(size(tensor), _n, axis)

    if axis == 1
        _r = reduce(*, _size[(axis+1):end])
        _tensor = reshape(tensor, (_m, _r))
        res = mat * _tensor

        # _target_size = (_n, _size[2:end]...)::NTuple{N,Int}
        return reshape(res, _target_size)
    elseif axis == N
        # _target_size = (_size[1:end-1]..., _n)::NTuple{N,Int}
        _l = reduce(*, _size[1:(axis-1)])
        _tensor = reshape(tensor, (_l, _m))
        res = _tensor * transpose(mat)
        return reshape(res, _target_size)
    else
        _l = reduce(*, _size[1:(axis-1)])
        _r = reduce(*, _size[(axis+1):end])
        _tensor = reshape(tensor, (_l, _m, _r))
        res = zeros(promote_type(T, TC), _l, _n, _r)
        @inbounds for j = 1:_r
            @inbounds for q = 1:_n
                @inbounds for k = 1:_m
                    @inbounds for i = 1:_l
                        res[i, q, j] += _tensor[i, k, j] * mat[q, k]
                    end
                end
            end
        end
        return reshape(res, _target_size)
    end
end



function _tensor2matrix(tensor::AbstractVector{T}, ::Val{axis}) where {T,axis}
    return reshape(tensor, length(tensor), 1), nothing
end

function _tensor2matrix(tensor::AbstractArray{T,N}, ::Val{axis}) where {T,N,axis}
    # internal function to move the axis dim to the first index, then reshape the tensor into a matrix
    _size = size(tensor)
    n1 = _size[axis]
    partialsize = (_size[1:(axis-1)]..., _size[(axis+1):end]...) # the size of the tensor except the axis-th dimension
    n2 = reduce(*, partialsize)

    if axis == 1 #no need to permutate the axis
        return reshape(tensor, (n1, n2)), partialsize
    elseif axis == N
        _tensor = reshape(tensor, (n2, n1))
        return transpose(_tensor), partialsize  #transpose do not create copy
    else
        permu = (axis, 2:(axis-1)..., 1, (axis+1):N...)
        _tensor = permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created 
        # ntensor = nocopy ? PermutedDimsArray(tensor, permu) : permutedims(tensor, permu) # permutate the axis-th and the 1st dim
        ntensor = reshape(_tensor, (n1, n2)) # no copy is created
        return ntensor, partialsize
    end
end

function _matrix2tensor(
    mat::AbstractMatrix{T},
    partialsize::Nothing,
    ::Val{axis},
) where {T,axis}
    return reshape(mat, length(mat))
end

function _matrix2tensor(
    mat::AbstractMatrix{T},
    partialsize::NTuple{dim,Int},
    ::Val{axis},
) where {T,dim,axis}
    # internal function to reshape matrix to a tensor, then swap the first index with the axis-th dimension
    n1, n2 = size(mat)
    @assert n2 == reduce(*, partialsize) # total number of elements of mat and the tensor must match
    # tsize = vcat(n1, partialsize)
    N = dim + 1

    if axis == 1
        tsize = (n1, partialsize...)
        tensor = reshape(mat, tsize)
        return tensor
    elseif axis == N # mat must be transposed to (n2, n1)
        _tensor = transpose(mat) #transpose do not create copy
        return reshape(_tensor, (partialsize..., n1))
    else
        # permu = [i for i = 1:dim]
        # permu[1], permu[axis] = axis, 1
        permu = (axis, 2:(axis-1)..., 1, (axis+1):N...)
        tsize = (n1, partialsize...)
        tensor = reshape(mat, tsize)
        return permutedims(tensor, permu) # permutate the axis-th and the 1st dim, a copy of the tensor is created
        # ntensor = nocopy ? PermutedDimsArray(tensor, permu) : permutedims(tensor, permu) # permutate the axis-th and the 1st dim
        # return ntensor
    end
end


function _weightedLeastSqureFit(dlrGrid, Gτ, error, kernel, sumrule)
    Nτ, Nω = size(kernel)
    @assert size(Gτ)[1] == Nτ
    if isnothing(sumrule) == false #require sumrule
        @assert dlrGrid.symmetry == :none && dlrGrid.isFermi "only unsymmetrized ferminoic sum rule has been implemented!"
        # println(size(Gτ))
        M = Int(floor(length(dlrGrid.ω) / 2))
        # M = length(dlrGrid.ω)

        kernel_m0 = kernel[:, M]
        # kernel = kernel[:, 1:Nω-1] #a copy of kernel submatrix will be created
        kernel = hcat(kernel[:, 1:(M-1)], kernel[:, (M+1):end])

        for i = 1:Nτ
            Gτ[i, :] .-= kernel_m0[i] * sumrule
        end

        for i = 1:(Nω-1)
            kernel[:, i] .-= kernel_m0
        end
        # kernel = view(kernel, :, 1:Nω-1)
    end

    if isnothing(error)
        B = kernel
        C = Gτ
    else
        @assert size(error) == size(Gτ)
        w = 1.0 ./ (error .+ 1e-16)

        for i = 1:Nτ
            wview = view(w, i, :)
            w[i, :] /= sum(wview) / length(wview)
        end
        B = w .* kernel
        C = w .* Gτ
    end
    # ker, ipiv, info = LAPACK.getrf!(B) # LU factorization
    # coeff = LAPACK.getrs!('N', ker, ipiv, C) # LU linear solvor for green=kernel*coeff
    coeff = B \ C#solve C = B * coeff

    if isnothing(sumrule) == false
        #make sure Gτ doesn't get modified after the linear fitting
        for i = 1:Nτ
            Gτ[i, :] .+= kernel_m0[i] * sumrule
        end
        #add back the coeff that are fixed by the sum rule
        coeffmore = sumrule' .- sum(coeff, dims = 1)
        cnew = zeros(eltype(coeff), size(coeff)[1] + 1, size(coeff)[2])
        cnew[1:(M-1), :] .= coeff[1:(M-1), :]
        cnew[(M+1):end, :] .= coeff[M:end, :]
        # println(size(coeffmore), ", ", size(cnew))
        # println(coeffmore)
        cnew[M, :] = coeffmore #broadcast cnew[M, :] .= coeffmore doesn't work for Julia 1.6
        return cnew
    else
        return coeff
    end
end

"""
    function tau2dlr(dlrGrid::DLRGrid, green, τGrid = dlrGrid.τ; error = nothing, axis = 1, sumrule = nothing, verbose = true)

imaginary-time domain to DLR representation

#Members:
- `dlrGrid`  : DLRGrid struct.
- `green`    : green's function in imaginary-time domain.
- `τGrid`    : the imaginary-time grid that Green's function is defined on. 
- `error`    : error the Green's function. 
- `axis`     : the imaginary-time axis in the data `green`.
- `sumrule`  : enforce the sum rule 
- `verbose`  : true to print warning information
"""
function tau2dlr(
    dlrGrid::DLRGrid{T,S},
    green::AbstractArray{TC,N},
    τGrid = dlrGrid.τ;
    error = nothing,
    axis = 1,
    sumrule = nothing,
    verbose = true,
) where {T,S,TC,N}
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    @assert size(green)[axis] == length(τGrid)
    ωGrid = dlrGrid.ω

    if length(τGrid) == length(dlrGrid.τ) && isapprox(τGrid, dlrGrid.τ; rtol = 10 * eps(T))
        if length(dlrGrid.kernel_τ) == 1
            dlrGrid.kernel_τ = Spectral.kernelT(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                τGrid,
                ωGrid,
                dlrGrid.β,
                true,
            )
        end
        kernel = dlrGrid.kernel_τ
    else
        kernel =
            Spectral.kernelT(T, Val(dlrGrid.isFermi), Val(S), τGrid, ωGrid, dlrGrid.β, true)
    end
    g, partialsize = _tensor2matrix(green, Val(axis))

    if isnothing(sumrule) == false
        # if dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha
        #     sumrule = sumrule ./ 2.0
        # end
        if isnothing(partialsize) == false
            sumrule = reshape(sumrule, size(g)[2])
        end
    end

    if isnothing(error) == false
        @assert size(error) == size(green)
        error, partialsize = _tensor2matrix(error, Val(axis))
    end

    coeff = _weightedLeastSqureFit(dlrGrid, g, error, kernel, sumrule)

    if verbose && all(x -> abs(x) < 1e16, coeff) == false
        @warn(
            "Some of the DLR coefficients are larger than 1e16. The quality of DLR fitting could be bad."
        )
    end

    if isnothing(sumrule) == false
        #check how exact is the sum rule
        coeffsum = sum(coeff, dims = 1) .- sumrule
        if verbose &&
           all(
            x -> abs(x) < 1000 * dlrGrid.rtol * max(maximum(abs.(green)), 1.0),
            coeffsum,
        ) == false
            @warn(
                "Sumrule error $(maximum(abs.(coeffsum))) is larger than the DLRGrid error threshold."
            )
        end
    end

    return _matrix2tensor(coeff, partialsize, Val(axis))
end

"""
    function dlr2tau(dlrGrid::DLRGrid, dlrcoeff, τGrid = dlrGrid.τ; axis = 1, verbose = true)

DLR representation to imaginary-time representation

#Members:
- `dlrGrid`  : DLRGrid
- `dlrcoeff` : DLR coefficients
- `τGrid`    : expected fine imaginary-time grids 
- `axis`     : imaginary-time axis in the data `dlrcoeff`
- `verbose`  : true to print warning information
"""
function dlr2tau(
    dlrGrid::DLRGrid{T,S},
    dlrcoeff::AbstractArray{TC,N},
    τGrid = dlrGrid.τ;
    axis = 1,
    verbose = true,
) where {T,S,TC,N}
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    @assert size(dlrcoeff)[axis] == length(dlrGrid)

    β = dlrGrid.β
    ωGrid = dlrGrid.ω

    if length(τGrid) == length(dlrGrid.τ) && isapprox(τGrid, dlrGrid.τ; rtol = 10 * eps(T))
        if length(dlrGrid.kernel_τ) == 1
            dlrGrid.kernel_τ = Spectral.kernelT(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                τGrid,
                ωGrid,
                dlrGrid.β,
                true,
            )
        end
        kernel = dlrGrid.kernel_τ
    else
        kernel =
            Spectral.kernelT(T, Val(dlrGrid.isFermi), Val(S), τGrid, ωGrid, dlrGrid.β, true)
    end

    # G = kernel * coeff # tensor dot product: \sum_i kernel[..., i]*coeff[i, ...]
    return _matrix_tensor_dot(kernel, dlrcoeff, axis)

end

"""
    function matfreq2dlr(dlrGrid::DLRGrid, green, nGrid = dlrGrid.n; error = nothing, axis = 1, sumrule = nothing, verbose = true)

Matsubara-frequency representation to DLR representation

#Members:
- `dlrGrid`  : DLRGrid struct.
- `green`    : green's function in Matsubara-frequency domain
- `nGrid`    : the n grid that Green's function is defined on. 
- `error`    : error the Green's function. 
- `axis`     : the Matsubara-frequency axis in the data `green`
- `sumrule`  : enforce the sum rule 
- `verbose`  : true to print warning information
"""
function matfreq2dlr(
    dlrGrid::DLRGrid{T,S},
    green::AbstractArray{TC,N},
    nGrid = dlrGrid.n;
    error = nothing,
    axis = 1,
    sumrule = nothing,
    verbose = true,
) where {T,S,TC,N}
    @assert length(size(green)) >= axis "dimension of the Green's function should be larger than axis!"
    @assert size(green)[axis] == length(nGrid)
    @assert eltype(nGrid) <: Integer
    ωGrid = dlrGrid.ω

    if (S == :ph && dlrGrid.isFermi == false) || (S == :pha && dlrGrid.isFermi == true)
        if length(nGrid) == length(dlrGrid.n) &&
           isapprox(nGrid, dlrGrid.n; rtol = 10 * eps(T))
            if length(dlrGrid.kernel_n) == 1
                dlrGrid.kernel_n = Spectral.kernelΩ(
                    T,
                    Val(dlrGrid.isFermi),
                    Val(S),
                    nGrid,
                    ωGrid,
                    dlrGrid.β,
                    true,
                )
            end
            kernel = dlrGrid.kernel_n
        else
            kernel = Spectral.kernelΩ(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                nGrid,
                ωGrid,
                dlrGrid.β,
                true,
            )
        end
    else
        if length(nGrid) == length(dlrGrid.n) &&
           isapprox(nGrid, dlrGrid.n; rtol = 10 * eps(T))
            if length(dlrGrid.kernel_n) == 1
                dlrGrid.kernel_nc = Spectral.kernelΩ(
                    T,
                    Val(dlrGrid.isFermi),
                    Val(S),
                    nGrid,
                    ωGrid,
                    dlrGrid.β,
                    true,
                )
            end
            kernel = dlrGrid.kernel_nc
        else
            kernel = Spectral.kernelΩ(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                nGrid,
                ωGrid,
                dlrGrid.β,
                true,
            )
        end
    end
    # if typ != eltype(green)
    #     green = convert.(typ, green)
    # end

    # if typ != eltype(kernel)
    #     dlrGrid.kernel_n = convert.(typ, dlrGrid.kernel_n)
    # end

    g, partialsize = _tensor2matrix(green, Val(axis))


    if isnothing(sumrule) == false
        # if dlrGrid.symmetry == :ph || dlrGrid.symmetry == :pha
        #     sumrule = sumrule ./ 2.0
        # end
        if isnothing(partialsize) == false
            sumrule = reshape(sumrule, size(g)[2])
        end
    end

    if isnothing(error) == false
        @assert size(error) == size(green)
        error, partialsize = _tensor2matrix(error, Val(axis))
    end
    coeff = _weightedLeastSqureFit(dlrGrid, g, error, kernel, sumrule)



    if verbose && all(x -> abs(x) < 1e16, coeff) == false
        @warn(
            "Some of the DLR coefficients are larger than 1e16. The quality of DLR fitting could be bad."
        )
    end

    if isnothing(sumrule) == false
        #check how exact is the sum rule
        coeffsum = sum(coeff, dims = 1) .- sumrule
        if verbose &&
           all(
            x -> abs(x) < 1000 * dlrGrid.rtol * max(maximum(abs.(green)), 1.0),
            coeffsum,
        ) == false
            @warn(
                "Sumrule error $(maximum(abs.(coeffsum))) is larger than the DLRGrid error threshold."
            )
        end
    end
    return _matrix2tensor(coeff, partialsize, Val(axis))
end

"""
    function dlr2matfreq(dlrGrid::DLRGrid, dlrcoeff, nGrid = dlrGrid.n; axis = 1, verbose = true)

DLR representation to Matsubara-frequency representation

#Members:
- `dlrGrid`  : DLRGrid
- `dlrcoeff` : DLR coefficients
- `nGrid`    : expected fine Matsubara-freqeuncy grids (integer)
- `axis`     : Matsubara-frequency axis in the data `dlrcoeff`
- `verbose`  : true to print warning information
"""
function dlr2matfreq(
    dlrGrid::DLRGrid{T,S},
    dlrcoeff::AbstractArray{TC,N},
    nGrid = dlrGrid.n;
    axis = 1,
    verbose = true,
) where {T,S,TC,N}
    @assert length(size(dlrcoeff)) >= axis "dimension of the dlr coefficients should be larger than axis!"
    @assert size(dlrcoeff)[axis] == length(dlrGrid)
    @assert eltype(nGrid) <: Integer
    ωGrid = dlrGrid.ω

    if (S == :ph && dlrGrid.isFermi == false) || (S == :pha && dlrGrid.isFermi == true)
        if length(nGrid) == length(dlrGrid.n) &&
           isapprox(nGrid, dlrGrid.n; rtol = 10 * eps(T))
            if length(dlrGrid.kernel_n) == 1
                dlrGrid.kernel_n = Spectral.kernelΩ(
                    T,
                    Val(dlrGrid.isFermi),
                    Val(S),
                    nGrid,
                    ωGrid,
                    dlrGrid.β,
                    true,
                )
            end
            kernel = dlrGrid.kernel_n
        else
            kernel = Spectral.kernelΩ(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                nGrid,
                ωGrid,
                dlrGrid.β,
                true,
            )
        end
    else
        if length(nGrid) == length(dlrGrid.n) &&
           isapprox(nGrid, dlrGrid.n; rtol = 10 * eps(T))
            if length(dlrGrid.kernel_n) == 1
                dlrGrid.kernel_nc = Spectral.kernelΩ(
                    T,
                    Val(dlrGrid.isFermi),
                    Val(S),
                    nGrid,
                    ωGrid,
                    dlrGrid.β,
                    true,
                )
            end
            kernel = dlrGrid.kernel_nc
        else
            kernel = Spectral.kernelΩ(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                nGrid,
                ωGrid,
                dlrGrid.β,
                true,
            )
        end
    end

    return _matrix_tensor_dot(kernel, dlrcoeff, axis)
end

"""
    function tau2matfreq(dlrGrid, green, nNewGrid = dlrGrid.n, τGrid = dlrGrid.τ; error = nothing, axis = 1, sumrule = nothing, verbose = true)

Fourier transform from imaginary-time to Matsubara-frequency using the DLR representation

#Members:
- `dlrGrid`  : DLRGrid
- `green`    : green's function in imaginary-time domain
- `nNewGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `τGrid`    : the imaginary-time grid that Green's function is defined on. 
- `error`    : error the Green's function. 
- `axis`     : the imaginary-time axis in the data `green`
- `sumrule`  : enforce the sum rule 
- `verbose`  : true to print warning information
"""
function tau2matfreq(
    dlrGrid::DLRGrid{T,S},
    green::AbstractArray{TC,N},
    nNewGrid::AbstractVector{Int} = dlrGrid.n,
    τGrid = dlrGrid.τ;
    error = nothing,
    axis = 1,
    sumrule = nothing,
    verbose = true,
) where {T,S,TC,N}
    coeff = tau2dlr(
        dlrGrid,
        green,
        τGrid;
        error = error,
        axis = axis,
        sumrule = sumrule,
        verbose = verbose,
    )
    return dlr2matfreq(dlrGrid, coeff, nNewGrid, axis = axis, verbose = verbose)
end

"""
    function matfreq2tau(dlrGrid, green, τNewGrid = dlrGrid.τ, nGrid = dlrGrid.n; error = nothing, axis = 1, sumrule = nothing, verbose = true)

Fourier transform from Matsubara-frequency to imaginary-time using the DLR representation

#Members:
- `dlrGrid`  : DLRGrid
- `green`    : green's function in Matsubara-freqeuncy repsentation
- `τNewGrid` : expected fine imaginary-time grids
- `nGrid`    : the n grid that Green's function is defined on. 
- `error`    : error the Green's function. 
- `axis`     : Matsubara-frequency axis in the data `green`
- `sumrule`  : enforce the sum rule 
- `verbose`  : true to print warning information
"""
function matfreq2tau(
    dlrGrid,
    green,
    τNewGrid = dlrGrid.τ,
    nGrid = dlrGrid.n;
    error = nothing,
    axis = 1,
    sumrule = nothing,
    verbose = true,
)
    coeff = matfreq2dlr(
        dlrGrid,
        green,
        nGrid;
        error = error,
        axis = axis,
        sumrule = sumrule,
        verbose = verbose,
    )
    return dlr2tau(dlrGrid, coeff, τNewGrid, axis = axis, verbose = verbose)
end

"""
    function tau2tau(dlrGrid, green, τNewGrid, τGrid = dlrGrid.τ; error = nothing, axis = 1, sumrule = nothing, verbose = true)

Interpolation from the old imaginary-time grid to a new grid using the DLR representation

#Members:
- `dlrGrid`  : DLRGrid
- `green`    : green's function in imaginary-time domain
- `τNewGrid` : expected fine imaginary-time grids
- `τGrid`    : the imaginary-time grid that Green's function is defined on. 
- `error`    : error the Green's function. 
- `axis`     : the imaginary-time axis in the data `green`
- `sumrule`  : enforce the sum rule 
- `verbose`  : true to print warning information
"""
function tau2tau(
    dlrGrid,
    green,
    τNewGrid,
    τGrid = dlrGrid.τ;
    error = nothing,
    axis = 1,
    sumrule = nothing,
    verbose = true,
)
    coeff = tau2dlr(
        dlrGrid,
        green,
        τGrid;
        error = error,
        axis = axis,
        sumrule = sumrule,
        verbose = verbose,
    )
    return dlr2tau(dlrGrid, coeff, τNewGrid, axis = axis, verbose = verbose)
end

"""
    function matfreq2matfreq(dlrGrid, green, nNewGrid, nGrid = dlrGrid.n; error = nothing, axis = 1, sumrule = nothing, verbose = true)

Fourier transform from Matsubara-frequency to imaginary-time using the DLR representation

#Members:
- `dlrGrid`  : DLRGrid
- `green`    : green's function in Matsubara-freqeuncy repsentation
- `nNewGrid` : expected fine Matsubara-freqeuncy grids (integer)
- `nGrid`    : the n grid that Green's function is defined on. 
- `error`    : error the Green's function. 
- `axis`     : Matsubara-frequency axis in the data `green`
- `sumrule`  : enforce the sum rule 
- `verbose`  : true to print warning information
"""
function matfreq2matfreq(
    dlrGrid,
    green,
    nNewGrid,
    nGrid = dlrGrid.n;
    error = nothing,
    axis = 1,
    sumrule = nothing,
    verbose = true,
)
    coeff = matfreq2dlr(
        dlrGrid,
        green,
        nGrid;
        error = error,
        axis = axis,
        sumrule = sumrule,
        verbose = verbose,
    )
    return dlr2matfreq(dlrGrid, coeff, nNewGrid, axis = axis, verbose = verbose)
end

# function convolution(dlrGrid, green1, green2; axis = 1)

# end

#=============================================================================
# In-place DLR operations with workspace buffers
#
# These functions are optimized for repeated calls with axis=3 and τGrid=dlr_grid.τ
# (the common case in QMBToolkitSOEHYB). They avoid allocations by:
# 1. Using pre-allocated workspace buffers
# 2. Caching the LU factorization of kernel_τ
# 3. Using fast contiguous transpose copies (transpose views are 20x slower for BLAS)
=============================================================================#

"""
    struct DLRWorkspace{T}

Pre-allocated workspace for in-place DLR operations with axis=3.
Holds contiguous buffers for efficient BLAS operations.

# Fields
- `buffer1`: Contiguous buffer of shape (n_dlr, n_other) for linear solve
- `buffer2`: Contiguous buffer of shape (n_dlr, n_other) for matrix multiply
- `n_dlr`: DLR rank (number of DLR basis functions)
- `n_other`: Product of non-DLR dimensions (e.g., dim1 * dim2)
"""
mutable struct DLRWorkspace{T<:Number}
    buffer1::Matrix{T}
    buffer2::Matrix{T}
    n_dlr::Int
    n_other::Int
end

"""
    DLRWorkspace(T::Type, n_dlr::Int, n_other::Int)

Create a DLRWorkspace with buffers sized for (dim1, dim2, n_dlr) arrays.

# Arguments
- `T`: Element type
- `n_dlr`: DLR rank (size of τ grid)
- `n_other`: Product of other dimensions (dim1 * dim2)
"""
function DLRWorkspace(::Type{T}, n_dlr::Int, n_other::Int) where {T<:Number}
    buffer1 = Matrix{T}(undef, n_dlr, n_other)
    buffer2 = Matrix{T}(undef, n_dlr, n_other)
    return DLRWorkspace{T}(buffer1, buffer2, n_dlr, n_other)
end

"""
    DLRWorkspace(dlrGrid::DLRGrid, dim1::Int, dim2::Int, ::Type{T}=Float64)

Create a DLRWorkspace sized for arrays of shape (dim1, dim2, n_dlr).
"""
function DLRWorkspace(
    dlrGrid::DLRGrid,
    dim1::Int,
    dim2::Int,
    ::Type{T} = Float64,
) where {T<:Number}
    n_dlr = length(dlrGrid)
    n_other = dim1 * dim2
    return DLRWorkspace(T, n_dlr, n_other)
end

# Fast transpose copy: src[i,j] -> dest[j,i]
# Uses proper loop ordering for cache efficiency
@inline function _fast_transpose_copy!(
    dest::AbstractMatrix{T},
    src::AbstractMatrix{T},
) where {T}
    m, n = size(src)
    @boundscheck @assert size(dest) == (n, m)
    @inbounds for j = 1:n
        @simd for i = 1:m
            dest[j, i] = src[i, j]
        end
    end
    return dest
end

# Copy 3D array (dim1, dim2, n_tau) to contiguous matrix (n_tau, dim1*dim2)
# Optimized for cache efficiency: processes in blocks for better locality
@inline function _tensor3_to_matrix!(
    dest::AbstractMatrix{T},
    src::AbstractArray{T,3},
) where {T}
    dim1, dim2, n_tau = size(src)
    @boundscheck @assert size(dest) == (n_tau, dim1 * dim2)

    # For small arrays or when n_tau is small, use simple loop
    # For larger arrays, the reshape+permutedims approach may be faster
    if dim1 * dim2 <= 64 || n_tau <= 32
        idx = 1
        @inbounds for j = 1:dim2
            for i = 1:dim1
                @simd for k = 1:n_tau
                    dest[k, idx] = src[i, j, k]
                end
                idx += 1
            end
        end
    else
        # For larger arrays, use a cache-blocked approach
        # Process columns of dest in blocks to improve cache reuse
        BLOCK_SIZE = 8
        @inbounds for j_block = 1:BLOCK_SIZE:dim2
            j_end = min(j_block + BLOCK_SIZE - 1, dim2)
            for i_block = 1:BLOCK_SIZE:dim1
                i_end = min(i_block + BLOCK_SIZE - 1, dim1)
                # Process this block: traverse tau (innermost read) for each (i,j) pair
                for k = 1:n_tau
                    for j = j_block:j_end
                        @simd for i = i_block:i_end
                            idx = (j - 1) * dim1 + i
                            dest[k, idx] = src[i, j, k]
                        end
                    end
                end
            end
        end
    end
    return dest
end

# Copy contiguous matrix (n_tau, dim1*dim2) back to 3D array (dim1, dim2, n_tau)
# Optimized for cache efficiency
@inline function _matrix_to_tensor3!(
    dest::AbstractArray{T,3},
    src::AbstractMatrix{T},
) where {T}
    dim1, dim2, n_tau = size(dest)
    @boundscheck @assert size(src) == (n_tau, dim1 * dim2)

    # For small arrays, use simple loop
    if dim1 * dim2 <= 64 || n_tau <= 32
        idx = 1
        @inbounds for j = 1:dim2
            for i = 1:dim1
                @simd for k = 1:n_tau
                    dest[i, j, k] = src[k, idx]
                end
                idx += 1
            end
        end
    else
        # For larger arrays, use a cache-blocked approach
        BLOCK_SIZE = 8
        @inbounds for j_block = 1:BLOCK_SIZE:dim2
            j_end = min(j_block + BLOCK_SIZE - 1, dim2)
            for i_block = 1:BLOCK_SIZE:dim1
                i_end = min(i_block + BLOCK_SIZE - 1, dim1)
                for k = 1:n_tau
                    for j = j_block:j_end
                        @simd for i = i_block:i_end
                            idx = (j - 1) * dim1 + i
                            dest[i, j, k] = src[k, idx]
                        end
                    end
                end
            end
        end
    end
    return dest
end

"""
    ensure_kernel_τ_lu!(dlrGrid::DLRGrid{T,S}) where {T,S}

Ensure the LU factorization of kernel_τ is computed and cached.
This is called automatically by tau2dlr! but can be called explicitly to
pre-compute the factorization.
"""
function ensure_kernel_τ_lu!(dlrGrid::DLRGrid{T,S}) where {T,S}
    # First ensure kernel_τ is computed
    if length(dlrGrid.kernel_τ) == 1
        dlrGrid.kernel_τ = Spectral.kernelT(
            T,
            Val(dlrGrid.isFermi),
            Val(S),
            dlrGrid.τ,
            dlrGrid.ω,
            dlrGrid.β,
            true,
        )
    end

    # Then compute LU factorization if not already done
    if isnothing(dlrGrid.kernel_τ_lu)
        dlrGrid.kernel_τ_lu = lu(dlrGrid.kernel_τ)
    end

    return dlrGrid.kernel_τ_lu
end

"""
    tau2dlr!(dest::AbstractArray{T,3}, dlrGrid::DLRGrid, green::AbstractArray{T,3},
             workspace::DLRWorkspace{T}) where T

In-place transform from τ-space to DLR coefficients for 3D arrays.
Assumes τGrid = dlrGrid.τ (the common case in QMBToolkitSOEHYB).

# Arguments
- `dest`: Output array of shape (dim1, dim2, n_dlr) for DLR coefficients
- `dlrGrid`: DLR grid (must have same τ grid)
- `green`: Input array of shape (dim1, dim2, n_dlr) on τ grid
- `workspace`: Pre-allocated DLRWorkspace

# Notes
- This function uses the cached LU factorization for efficiency
- The workspace buffers must be sized appropriately (n_dlr, dim1*dim2)
"""
function tau2dlr!(
    dest::AbstractArray{TC,3},
    dlrGrid::DLRGrid{T,S},
    green::AbstractArray{TC,3},
    workspace::DLRWorkspace{TC},
) where {T,S,TC<:Number}
    dim1, dim2, n_tau = size(green)
    n_dlr = length(dlrGrid)

    @boundscheck begin
        @assert size(dest) == (dim1, dim2, n_dlr) "dest size $(size(dest)) != expected ($dim1, $dim2, $n_dlr)"
        @assert n_tau == n_dlr "Input τ dimension $n_tau != DLR size $n_dlr (must use dlrGrid.τ)"
        @assert workspace.n_dlr >= n_dlr "Workspace n_dlr $(workspace.n_dlr) < required $n_dlr"
        @assert workspace.n_other >= dim1 * dim2 "Workspace n_other $(workspace.n_other) < required $(dim1*dim2)"
    end

    # Ensure LU factorization is available
    lu_fact = ensure_kernel_τ_lu!(dlrGrid)

    # Copy input to contiguous buffer: green(dim1, dim2, n_tau) -> buffer1(n_tau, dim1*dim2)
    buffer = view(workspace.buffer1, 1:n_dlr, 1:(dim1*dim2))
    _tensor3_to_matrix!(buffer, green)

    # Solve in-place: kernel * coeff = input, so coeff = kernel \ input
    # buffer now contains the solution (DLR coefficients)
    ldiv!(lu_fact, buffer)

    # Copy result back: buffer(n_dlr, dim1*dim2) -> dest(dim1, dim2, n_dlr)
    _matrix_to_tensor3!(dest, buffer)

    return dest
end

"""
    dlr2tau!(dest::AbstractArray{T,3}, dlrGrid::DLRGrid, dlrcoeff::AbstractArray{T,3},
             workspace::DLRWorkspace{T}) where T

In-place transform from DLR coefficients to τ-space for 3D arrays.
Assumes τGrid = dlrGrid.τ (the common case in QMBToolkitSOEHYB).

# Arguments
- `dest`: Output array of shape (dim1, dim2, n_dlr) for τ values
- `dlrGrid`: DLR grid (must use same τ grid)
- `dlrcoeff`: Input array of shape (dim1, dim2, n_dlr) with DLR coefficients
- `workspace`: Pre-allocated DLRWorkspace

# Notes
- This function uses matrix multiplication with contiguous buffers for efficiency
- The workspace buffers must be sized appropriately (n_dlr, dim1*dim2)
"""
function dlr2tau!(
    dest::AbstractArray{TC,3},
    dlrGrid::DLRGrid{T,S},
    dlrcoeff::AbstractArray{TC,3},
    workspace::DLRWorkspace{TC},
) where {T,S,TC<:Number}
    # Delegate to the version with explicit τGrid = dlrGrid.τ
    return dlr2tau!(dest, dlrGrid, dlrcoeff, dlrGrid.τ, workspace)
end

"""
    dlr2tau!(dest::AbstractArray{T,3}, dlrGrid::DLRGrid, dlrcoeff::AbstractArray{T,3},
             τGrid::AbstractVector, workspace::DLRWorkspace{T}) where T

In-place transform from DLR coefficients to τ-space for 3D arrays with custom τGrid.
This version is useful for reflection operations where τGrid = β - dlrGrid.τ.

# Arguments
- `dest`: Output array of shape (dim1, dim2, n_tau) for τ values
- `dlrGrid`: DLR grid
- `dlrcoeff`: Input array of shape (dim1, dim2, n_dlr) with DLR coefficients
- `τGrid`: Custom τ grid to evaluate on
- `workspace`: Pre-allocated DLRWorkspace

# Notes
- This function uses matrix multiplication with contiguous buffers for efficiency
- When τGrid != dlrGrid.τ, the kernel is computed on-the-fly (small allocation)
- The workspace buffers avoid the larger tensor allocation/permutation overhead
"""
function dlr2tau!(
    dest::AbstractArray{TC,3},
    dlrGrid::DLRGrid{T,S},
    dlrcoeff::AbstractArray{TC,3},
    τGrid::AbstractVector,
    workspace::DLRWorkspace{TC},
) where {T,S,TC<:Number}
    dim1, dim2, n_dlr_in = size(dlrcoeff)
    n_dlr = length(dlrGrid)
    n_tau = length(τGrid)

    @boundscheck begin
        @assert size(dest) == (dim1, dim2, n_tau) "dest size $(size(dest)) != expected ($dim1, $dim2, $n_tau)"
        @assert n_dlr_in == n_dlr "Input DLR dimension $n_dlr_in != DLR size $n_dlr"
        @assert workspace.n_dlr >= max(n_dlr, n_tau) "Workspace n_dlr $(workspace.n_dlr) < required $(max(n_dlr, n_tau))"
        @assert workspace.n_other >= dim1 * dim2 "Workspace n_other $(workspace.n_other) < required $(dim1*dim2)"
    end

    # Get or compute the kernel
    if length(τGrid) == length(dlrGrid.τ) && isapprox(τGrid, dlrGrid.τ; rtol = 10 * eps(T))
        # Use cached kernel_τ
        if length(dlrGrid.kernel_τ) == 1
            dlrGrid.kernel_τ = Spectral.kernelT(
                T,
                Val(dlrGrid.isFermi),
                Val(S),
                dlrGrid.τ,
                dlrGrid.ω,
                dlrGrid.β,
                true,
            )
        end
        kernel = dlrGrid.kernel_τ
    else
        # Compute kernel for custom τGrid (small allocation: n_tau × n_dlr matrix)
        kernel = Spectral.kernelT(
            T,
            Val(dlrGrid.isFermi),
            Val(S),
            τGrid,
            dlrGrid.ω,
            dlrGrid.β,
            true,
        )
    end

    # Copy input to contiguous buffer: dlrcoeff(dim1, dim2, n_dlr) -> buffer1(n_dlr, dim1*dim2)
    buffer_in = view(workspace.buffer1, 1:n_dlr, 1:(dim1*dim2))
    buffer_out = view(workspace.buffer2, 1:n_tau, 1:(dim1*dim2))
    _tensor3_to_matrix!(buffer_in, dlrcoeff)

    # Matrix multiply: dest = kernel * coeff
    # kernel is (n_tau, n_dlr), buffer_in is (n_dlr, n_other), result is (n_tau, n_other)
    mul!(buffer_out, kernel, buffer_in)

    # Copy result back: buffer_out(n_tau, dim1*dim2) -> dest(dim1, dim2, n_tau)
    _matrix_to_tensor3!(dest, buffer_out)

    return dest
end
