"""
A concise, lightweight, high-performance Julia implementation of the Thomas block-tridiagonal matrix algorithm. You can copy the code to use in your scripts or modules.
"""

mutable struct BlockThomas
    n::Int
    m::Int
    lu_buf::Vector{LU{Float64,Matrix{Float64},Vector{Int64}}}
    D_buf::Matrix{Float64}
    b_buf::Vector{Float64}
    function BlockThomas(n, m)
        lu_buf = Vector{LU{Float64,Matrix{Float64},Vector{Int64}}}(undef, m)
        D_buf = mat(n, n)
        b_buf = vec(n)
        return new(n, m, lu_buf, D_buf, b_buf)
    end
end

const BlockThomasBuf = Vector{BlockThomas}(undef, nthreads())

function get_block_thomas_buffer(n::Int, m::Int)
    tid = threadid()
    return if !isassigned(BlockThomasBuf, tid)
        BlockThomasBuf[tid] = BlockThomas(n, m)
    elseif BlockThomasBuf[tid].n != n || BlockThomasBuf[tid].m != m
        BlockThomasBuf[tid] = BlockThomas(n, m)
    else
        BlockThomasBuf[tid]
    end
end

"""
Note that this function modifies `x`, `L`, `D` and `b` in place
"""
@views function block_thomas_tridiagonal!(x::AbstractVector, L::AbstractVector{T},
    D::AbstractVector{T}, U::AbstractVector{T}, b::AbstractVector) where {T}
    n = size(D[1], 1) # each block has size n*n
    m = length(D) # num of blocks
    (; lu_buf, D_buf, b_buf) = get_block_thomas_buffer(n, m)

    # Forward elimination
    for i = 1:m-1
        lu_buf[i] = lu!(D[i])
        rdiv!(L[i], lu_buf[i])
        D[i+1] .-= mul!(D_buf, L[i], U[i])
        b[n*i+1:n*(i+1)] .-= mul!(b_buf, L[i], b[n*(i-1)+1:n*i])
    end

    # Backward substitution
    ldiv!(x[n*(m-1)+1:n*m], lu!(D[m]), b[n*(m-1)+1:n*m])
    for i = m-1:-1:1
        ldiv!(x[n*(i-1)+1:n*i], lu_buf[i], b[n*(i-1)+1:n*i] - mul!(b_buf, U[i], x[n*i+1:n*(i+1)]))
    end

    return x
end
