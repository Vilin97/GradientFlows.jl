struct Blob{S,F,A,L} <: Solver
    score_values::S
    ε::F
    allocated_memory::A
    verbose::Int
    logger::L
end

Blob(;verbose=0, logger=Logger()) = Blob(nothing, nothing, nothing, verbose, logger)

struct BlobAllocMemCPU{T}
    diff_norm2s::Matrix{T}
    mol_sum::Vector{T}
    mols::Matrix{T}
end

"Initialize solver."
function initialize(solver::Blob, u0, score_values::Matrix{T}, problem_name; kwargs...) where {T}
    n = size(score_values, 2)
    diff_norm2s = zeros(T, n, n)
    mol_sum = zeros(T, n)
    mols = zeros(T, n, n)
    allocated_memory = BlobAllocMemCPU(diff_norm2s, mol_sum, mols)
    ε = blob_bandwidth(u0)
    logger = Logger(solver.logger.log_level, score_values)
    Blob(copy(score_values), T(ε), allocated_memory, solver.verbose, logger)
end

"Fill in solver.score_values."
function update!(solver::Blob{S,F,A}, integrator) where {S,F,A<:BlobAllocMemCPU}
    @unpack score_values, ε, allocated_memory, verbose, logger = solver
    @unpack diff_norm2s, mol_sum, mols = allocated_memory
    u = integrator.u
    d, n = size(u)

    diff_norm2s .= 0
    mol_sum .= 0
    score_values .= 0
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        diff_norm2s[p, q] += (u[k, p] - u[k, q])^2
    end
    @tturbo for p in 1:n, q in 1:n
        mols[p, q] = exp(-diff_norm2s[p, q] / ε) / sqrt((π * ε)^d)
        mol_sum[p] += mols[p, q]
    end
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        fac = -2 / ε * mols[p, q]
        diff_k = u[k, p] - u[k, q]
        score_values[k, p] += fac * diff_k / mol_sum[p]
        score_values[k, p] += fac * diff_k / mol_sum[q]
    end
    if verbose > 0 && integrator.iter % 10 == 0
        test_loss = pretty(l2_error_normalized(score_values, true_score(integrator.p, integrator.t, integrator.u)), 7)
        println("Time $(integrator.t) test loss = $test_loss")
    end
    log!(logger, solver)
    nothing
end

function Base.show(io::IO, solver::Blob)
    Base.print(io, "Blob")
end
name(solver::Blob) = "blob"

"ε = C * n^(-2 / (d + 6)) is optimal for gradient matching."
function blob_bandwidth(u) 
    d, n = size(u)
    Σ = diag(cov(u'))
    4 * prod(Σ)^(1/d) * n^(-2 / (d + 6))
end