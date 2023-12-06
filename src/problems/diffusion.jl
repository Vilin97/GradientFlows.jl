"Make a diffusion problem with the given dimension, number of particles, and solver."
function diffusion_problem(d, n, solver_; t0::F=1.0, t_end::F=2.0, dt::F=0.01, rng=DEFAULT_RNG, kwargs...) where {F}
    if solver_ isa SBTM && F == Float64
        return diffusion_problem(d, n, solver_; t0=Float32(t0), t_end=Float32(t_end), dt=Float32(dt), rng=rng, kwargs...)
    end
    f!(du, u, prob, t) = (du .= -prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = MvNormal(2t * I(d))
    params = nothing
    ρ0 = ρ(t0, params)
    u0 = rand(rng, ρ0, n)
    name = "diffusion"
    solver = initialize(solver_, u0, score(ρ0, u0), name; kwargs...)
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name)
end