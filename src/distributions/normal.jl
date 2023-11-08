function marginal_pdf(dist::MvNormal, x::Number)
    σ = cov(dist)[1, 1]
    return (2π * σ)^(-1 / 2) * exp(-(x - mean(dist)[1])^2 / (2σ))
end

"E |X|^k"
function abs_moment(dist::MvNormal, k::Int)
    Σ = cov(dist)
    if k == 2
        return tr(Σ)
    elseif k == 4
        return tr(Σ)^2 + 2 * sum(abs2, Σ)
    else
        error("abs_moment(dist, k) is not implemented for k = $k.")
    end
end