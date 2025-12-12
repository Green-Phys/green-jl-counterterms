"""
Stub implementations for BlockDense operations.

These will be implemented in Phase 4.
For now, they provide clear error messages when BlockDense format is attempted.
"""

function tau2dlr_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 4). Use :dense format.")
end

function dlr2tau_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 4). Use :dense format.")
end

function do_convolution_dlr2tau_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 4). Use :dense format.")
end

function do_reflection_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 4). Use :dense format.")
end