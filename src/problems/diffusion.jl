"Make a diffusion problem with the given dimension, number of particles, and solver."
function diffusion_problem(d, n, solver_; t0::F=1.0, t_end::F=2.0, dt::F=0.01, rng=DEFAULT_RNG, kwargs...) where {F}
    f!(du, u, prob, t) = (du .= -prob.params.D .* prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = MvNormal(params.D .* 2t .* I(d))
    params = (D=F(1 / 4),) # diffusion coefficient
    ρ0 = ρ(t0, params)
    u0 = rand(rng, ρ0, n)
    name = "diffusion"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    diffusion_coefficient(u, params) = params.D
    covariance(t, params) = cov(ρ(t, params))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, diffusion_coefficient, covariance)
end