struct GradFlowProblem{FU,D,M,DF,F,P,S}
    f!::FU # f!(du, u, prob :: GradFlowProblem, t) 
    ρ0::D # initial distribution
    u0::M # sample of initial distribution
    ρ::DF  # ρ(t, params) target distribution, if known
    tspan::Tuple{F,F} # (t0, t_end)
    dt::F # time step
    params::P # physical parameters of the problem
    solver::S # the solver to use
    name::String
end

function Base.show(io::IO, prob::GradFlowProblem)
    print(io, "$(prob.name) with \n tspan = $(prob.tspan) \n  dt = $(prob.dt) \n  params = $(prob.params) \n  solver = $(prob.solver)")
end

true_dist(prob::GradFlowProblem, t) = prob.ρ(t, prob.params)

function set_solver(problem::GradFlowProblem, solver_)
    @unpack f!, ρ0, u0, ρ, tspan, dt, params, solver = problem
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, initialize(solver_, u0, score(ρ0, u0)), problem.name)
end

"Reset the problem to its initial state so it can be solved again."
function reset!(problem::GradFlowProblem)
    reset!(problem.solver, problem.u0, score(problem.ρ0, problem.u0))
    nothing
end

"Resample u0."
function resample!(problem::GradFlowProblem; rng=DEFAULT_RNG)
    n = size(problem.u0, 2)
    problem.u0 .= rand(rng, problem.ρ0, n)
    reset!(problem)
    nothing
end