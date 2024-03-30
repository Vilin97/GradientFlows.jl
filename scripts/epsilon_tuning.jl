# Plan
# 1. Go over d = 2,5,10
# 2. Go over n = 100 .* 2.^(0:10)
# 3. Go over two distributions -- normal and polynormal (BKW). Keep covariance = 1.
# 4. Go over epsilon = n^-2/(4+2d) and epsilon = n^-2/(4+d) 
# 5. Make plots, here the x-axis is number of samples n, and the y-axis is the L2 error between the true density and the KDE.

using GradientFlows, Plots, Polynomials, LinearAlgebra, Distributions
include("plot.jl")
# ENV["GKSwstype"] = "nul" # no GUI
# default(display_type=:inline)

function Lp_error_plot(distribution, ds, ns, epsilons)
    Lp_errors = zeros(length(ds), length(ns), length(epsilons))
    for (i,d) in enumerate(ds)
        dist = distribution(d)
        @show d
        for (j,n) in enumerate(ns)
            average_errors = zeros(length(epsilons))
            @show [epsilon(n,d) for (_, epsilon) in epsilons]
            N = 100
            for _ in 1:N
                sample = rand(dist, n)
                average_errors .+= [Lp_error(sample, dist; h=epsilon(n,d)*I(d), p=2, xlim=3) for (_, epsilon) in epsilons]
            end
            Lp_errors[i,j,:] = average_errors / N
        end
    end

    plots = []
    for (i,d) in enumerate(ds)
        push!(plots, plot_metric_over_n(ns, [ep[1] for ep in epsilons], "L2 reconstruction error, d=$d", "|ρₜⁿ∗ϕ - ρₜ*|₂", Lp_errors[i,:,:]))
    end
    L2_plt = plot(plots..., size=PLOT_WINDOW_SIZE ./ 2, layout=(length(ds),1))
end
function pdf_plot(dist, d, ns)
    slice_plots = []
    for (i,n) in enumerate(ns)
        @show n
        u = rand(dist, n)
        p_slice = Plots.plot(xlabel="x", ylabel="Σᵢϕ([x,0...] - Xᵢ)/n", margin=PLOT_MARGIN, title="Slice of reconstruction pdf, n=$n");
        plot!(p_slice, range(-5,5,length=200), x -> pdf(dist, [x, zeros(d-1)...]), label="true", lw=PLOT_LINE_WIDTH);
        plot!(p_slice, range(-5,5,length=200), x -> kde([x, zeros(d-1)...], u; h=n^(-2/(4+d)) * I(d)), label="h=n^(-2/(4+d))=$(round(n^(-2/(4+d)), digits=3))", lw=PLOT_LINE_WIDTH)
        plot!(p_slice, range(-5,5,length=200), x -> kde([x, zeros(d-1)...], u; h=n^(-2/(4+2d)) * I(d)), label="h=n^(-2/(4+2d))=$(round(n^(-2/(4+2d)), digits=3))", lw=PLOT_LINE_WIDTH);
        push!(slice_plots, p_slice)
    end
    slice_plt = plot(slice_plots..., size=PLOT_WINDOW_SIZE)
end

distribution(d) = GradientFlows.MvNormal(I(d))
ds = [2,3]
ns = 100 .* 2 .^ (0:6)
epsilons = [("n^(-2/(4+2d))", (n,d) -> n^(-2/(4+2d))), ("n^(-2/(4+d))", (n,d) -> n^(-2/(4+d)))]
L2_plt = Lp_error_plot(distribution, ds, ns, epsilons)

ns = 10 .^ (3:6)
d=2
dist = distribution(d)
slice_plt = pdf_plot(dist, d, ns)

plt = plot(L2_plt, slice_plt, size=PLOT_WINDOW_SIZE, plot_title="Normal density reconstruction")
savefig(plt, "data/plots/hyperparameter_tuning/epsilon_tuning_normal")
