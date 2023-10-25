function update!(integrator)
    prob = integrator.p
    solver = prob.solver
    u_modified!(integrator, false)
    @timeit DEFAULT_TIMER "update score" update!(solver, integrator) # dispatch on the solver type
end

function solve(prob::GradFlowProblem; kwargs...)
    @timeit DEFAULT_TIMER "reset problem" reset!(prob)
    ts = collect(prob.tspan[1]:prob.dt:prob.tspan[2])
    cb = PresetTimeCallback(ts, update!, save_positions=(false, false))
    timed_f! = (du, u, prob, t) -> (@timeit DEFAULT_TIMER "move particles" prob.f!(du, u, prob, t))
    ode_problem = ODEProblem(timed_f!, prob.u0, prob.tspan, prob)
    return solve(ode_problem, Euler(), dt=prob.dt, callback=cb; kwargs...)
end