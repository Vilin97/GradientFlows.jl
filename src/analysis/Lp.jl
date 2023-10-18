"Compute Lp distance between two functions."
function Lp_distance(f1::F1, f2::F2; p, d, xlim, verbose=false, rtol=0.05, maxevals=10^5, kwargs...) where {F1<:Function,F2<:Function}
    result, integration_error = hcubature(x -> abs.(f1(x) - f2(x))^p, fill(-xlim, d), fill(xlim, d); rtol=rtol, maxevals=maxevals, kwargs...)
    result = max(zero(result), result)^(1 / p)
    verbose > 0 && (println("relative integration error ~ $(integration_error/abs(result))"))
    return result
end

"Compute Lp error of a particle solution."
function Lp_error(u::AbstractMatrix, true_pdf::Function; xlim=nothing, kwargs...)
    d = size(u, 1)
    empirical_pdf(x) = kde(x, u; kwargs...)
    xlim_ = isnothing(xlim) ? maximum(abs.(u)) : eltype(u)(xlim)
    return Lp_distance(empirical_pdf, true_pdf; d=d, xlim=xlim_, kwargs...)
end

function Lp_error(u::AbstractMatrix, dist::D; kwargs...) where {D<:Distribution}
    true_pdf(x) = pdf(dist, x)
    return Lp_error(u, true_pdf; kwargs...)
end