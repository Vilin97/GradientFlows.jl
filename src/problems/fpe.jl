"Make a fokker planck problem with potential V(x) = |x|^2/2 with the given dimension, number of particles, and solver."
function fpe_problem(d, n, solver_; t0::F=0.1, t_end::F=1.1, dt::F=0.01, rng=DEFAULT_RNG, kwargs...) where {F}
    f!(du, u, prob, t) = (du .= -u .- prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = MvNormal((1 - exp(-2t)) * I(d))
    params = nothing
    ρ0 = ρ(t0, params)
    u0 = rand(rng, ρ0, n)
    name = "fpe"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    diffusion_coefficient(u, params) = 1
    covariance(t, params) = cov(ρ(t, params))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, diffusion_coefficient, covariance)
end