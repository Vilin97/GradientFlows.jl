"Transfer the problem to the GPU."
function CUDA.cu(problem::GradFlowProblem{D,F,M}) where {D,F,M}
    @unpack f!, ρ0, u0, ρ, tspan, dt, params, solver = problem
    cu_u = cu(u0)
    cu_params = cu.(params)
    cu_ρ0 = cu(ρ0)
    cu_solver = initialize(solver, score(cu_ρ0, cu_u))
    cu_ρ(t, params) = cu(ρ(t, params))
    return GradFlowProblem(f!, cu_ρ0, cu_u, cu_ρ, tspan, dt, cu_params, cu_solver)
end

"Transfer the distribution to the GPU."
function CUDA.cu(ρ::D) where {D <: MvNormal}
    MvNormal(cu(ρ.μ), cu(ρ.Σ))
end