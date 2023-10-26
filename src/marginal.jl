function marginal_pdf(dist::MvNormal, x::Number)
    σ = cov(dist)[1,1]
    return (2π * σ)^(-1 / 2) * exp(-(x-mean(dist)[1])^2 / (2σ))
end