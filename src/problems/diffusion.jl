"Make a diffusion problem with the given dimension, number of particles, and solver."
function diffusion_problem(d, n, solver_; t0::F=1.0, t_end::F=5.0, rng=DEFAULT_RNG) where {F}
    if solver_ isa SBTM && F==Float64
        return diffusion_problem(d, n, solver_; t0=Float32(t0), t_end=Float32(t_end), rng=rng)
    end
    f!(du, u, prob, t) = (du .= -prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = MvNormal(2t * I(d))
    dt = F(0.01)
    params = nothing
    ρ0 = ρ(t0, params)
    u0 = F.(rand(rng, ρ0, n))
    solver = initialize(solver_, u0, score(ρ0, u0))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver)
end