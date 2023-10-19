"|x-y|^2, assumes size(x)==size(y), does not autodiff"
function normsq(x, y)
    s = zero(eltype(x))
    @tturbo for i in 1:length(x)
        s += abs2(x[i] - y[i])
    end
    s
end