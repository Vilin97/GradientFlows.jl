using GradientFlows, Flux
using Flux: gpu
using CUDA

CUDA.allowscalar(false)
d = 2
n = 10
u0 = CUDA.rand(d, n)
score_values = CUDA.rand(d, n)
solver = GradientFlows.initialize(SBTM(gpu(mlp(2,depth=1)); logger=Logger(2), init_max_iterations=100), u0, score_values)

s = solver.s
s(u0)
GradientFlows.l2_error_normalized(solver.s, u0, score_values)
train_s!(solver, u0, score_values)

struct DummyIntegrator
    u
    iter
    t
end
integrator = DummyIntegrator(u0, 0, 0.0)
GradientFlows.update!(solver, integrator)

d = 5
n = 1000
u = CUDA.rand(d, n)
s = CUDA.rand(d, n)

f1(u,s) = sum( sum(abs2, u[:,q]) * s[:,q] for q in 1:n)
f2(u,s) = @views sum( sum(abs2, u[:,q]) * s[:,q] for q in 1:n)

f1(u,s)
f2(u,s)
CUDA.@time f1(u,s)
CUDA.@time f2(u,s)

CUDA.@time reduce(+, s, dims=2)
CUDA.@time reduce((i,j) -> s[:,i] + s[:,j], 1:n, init=CUDA.zeros(d,1))



f1(u, s, n=size(u,2)) = sum( sum(abs2, u[:,q]) * s[:,q] - sum(u[:,q] .* s[:,q]) * s[:,q] for q in 1:n)
using Einsum
f2(u, s) = (@einsum first[a] := abs2(u[c,q]) * s[a,q]) - (@einsum second[a] := u[b,q] * s[b,q] * s[a,q])
using Tullio, CUDA, KernelAbstractions
function f2bis(u, s)
  @tullio out[a] := abs2(u[c,q]) * s[a,q]
  @tullio out[a] += - u[b,q] * s[b,q] * s[a,q]
end
using LinearAlgebra
f3(u, s) = s * (vec(sum(abs2, u; dims=1)) - diag(u' * s))
f4(u, s) = s * (vec(sum(abs2, u; dims=1))) - u * diag(u' * s)

d = 5
n = 100
u = CUDA.rand(d, n)
s = CUDA.rand(d, n)

f1(u, s)
f2(u, s) # Error: scalar indexing is disallowed
f2bis(u, s) # Error: scalar indexing is disallowed
f3(u, s)
f4(u, s)
CUDA.@time f1(u, s) # 10 seconds
CUDA.@time f3(u, s) # 0.006 seconds
CUDA.@time f4(u, s) # 0.006 seconds


function landau_cu_f!(du, u, prob, t)
    s = prob.solver.score_values
    n = size(u, 2)
    for p in 1:n
        up = u .- u[:,p]
        sp = s .- s[:,p]
        du[:,p] .= sp * (vec(sum(abs2, up; dims=1))) - up * diag(up' * sp)
        # du[:,p] .= s * (vec(sum(abs2, u; dims=1))) - u * diag(u' * s)
    end
    du .*= prob.params.B / n
end

struct DummyProb
    solver
    params
end
d = 5
n = 1000
cuu = CUDA.rand(d, n)
u = Array(cuu)
s = CUDA.rand(d, n)
cudu = CUDA.zeros(d, n)
du = Array(cudu)

cuprob = DummyProb((score_values = s, ), (B = 1f0, ))
prob = DummyProb((score_values = Array(s), ), (B = 1f0, ))
CUDA.@time landau_cu_f!(cudu, cuu, cuprob, 0f0)
CUDA.@time GradientFlows.landau_5d_f!(du, u, prob, 0f0)
du ≈ Array(cudu)