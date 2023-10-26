emp_mean(u::AbstractMatrix) = vec(mean(u, dims=2))
emp_cov(u::AbstractMatrix) = cov(u, dims=2)