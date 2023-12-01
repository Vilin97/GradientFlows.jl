using GradientFlows, LinearAlgebra

"ε = C * n^(-2 / (d + 6)) is optimal for gradient matching."
function blob_bandwidth(u) 
    d, n = size(u)
    Σ = diag(cov(u'))
    4 * prod(Σ)^(1/d) * n^(-2 / (d + 6))
end

# blob_epsilon(d, n) = 2 * n^(-2 / (d + 4))

for (d, n) in [(10, 10000)]
    u = randn(d, n) * 2
    integrator = (u = u,)
    probs = [diffusion_problem(d, n, Blob(ε)) for ε in [blob_bandwidth(u), blob_bandwidth(u) * 2, blob_bandwidth(u) * 4, blob_bandwidth(u) * 8]]
    for prob in probs
        set_u0!(prob, u)
        score = copy(prob.solver.score_values)
        GradientFlows.update!(prob.solver, integrator)
        error = round(sum(abs2, score .- prob.solver.score_values) / n, digits=2)
        rel_error = round(n * error / sum(abs2, score), digits=2)
        ε = round(prob.solver.ε, digits=2)
        println("d=$(rpad(d,2)) n=$n ε=$(rpad(ε,5)) error=$error, rel_error=$rel_error")
    end
end

