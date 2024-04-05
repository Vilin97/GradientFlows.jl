gradlogpdf(d::MixtureModel, x) = gradient(x_ -> logpdf(d, x_), x)[1]
marginal_pdf(d::MixtureModel, x) = sum([marginal_pdf(component, x) * p for (component, p) in zip(d.components, d.prior.p)])

mean(d::MixtureModel) = sum([mean(component) * p for (component, p) in zip(d.components, d.prior.p)])

function cov(d::MixtureModel)
    μ = mean(d)
    Σ = zeros(size(μ, 1), size(μ, 1))
    for (component, p) in zip(d.components, d.prior.p)
        Σ += p * (cov(component) + (mean(component) - μ) * (mean(component) - μ)')
    end
    return Σ
end