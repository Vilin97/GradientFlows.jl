function update!(integrator)
    prob = integrator.p
    solver = prob.solver
    u_modified!(integrator, false)
    update!(solver, integrator) # dispatch on the solver type
end

function solve(prob::GradFlowProblem; kwargs...)
    ts = collect(prob.tspan[1]:prob.dt:prob.tspan[2])
    cb = PresetTimeCallback(ts, update!, save_positions=(false, false))
    ode_problem = ODEProblem(prob.f!, prob.u0, prob.tspan, prob)
    return solve(ode_problem, Euler(), dt=prob.dt, callback=cb; kwargs...)
end