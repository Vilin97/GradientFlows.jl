struct GradFlowProblem{FU,D,M,DF,F,P,S}
    f!::FU # f!(du, u, prob :: GradFlowProblem, t) 
    ρ0::D # initial distribution
    u0::M # sample of initial distribution
    ρ::DF  # ρ(t, params) target distribution, if known
    tspan::Tuple{F,F} # (t0, t_end)
    dt::F # time step
    params::P # physical parameters of the problem
    solver::S # the solver to use
end

function Base.show(io::IO, prob::GradFlowProblem)
    println(io, "$(typeof(prob)) with \n  ρ₀ = $(prob.ρ0) \n  tspan = $(prob.tspan) \n  dt = $(prob.dt) \n  params = $(prob.params) \n  solver = $(prob.solver)")
end

true_dist(prob::GradFlowProblem, t) = prob.ρ(t, prob.params)

function diffusion_problem(d, n, solver_; t0::F=1.0, t_end::F=5.0, rng=DEFAULT_RNG) where {F}
    f!(du, u, prob, t) = (du .= -prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = MvNormal(2t * I(d))
    dt = F(0.01)
    params = nothing
    ρ0 = ρ(t0, params)
    u0 = F.(rand(rng, ρ0, n))
    solver = initialize(solver_, score(ρ0, u0))
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver)
end

function set_solver(problem::GradFlowProblem, solver_)
    @unpack f!, ρ0, u0, ρ, tspan, dt, params, solver = problem
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, initialize(solver_, score(ρ0, u0)))
end
