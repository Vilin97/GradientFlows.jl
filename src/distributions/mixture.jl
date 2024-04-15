gradlogpdf(d::MixtureModel, x) = gradient(x_ -> logpdf(d, x_), x)[1]
marginal_pdf(d::MixtureModel, x) = sum([marginal_pdf(component, x) * p for (component, p) in zip(d.components, d.prior.p)])
