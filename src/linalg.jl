"|x-y|^2, assumes size(x)==size(y), does not autodiff"
function normsq(x, y)
    s = zero(eltype(x))
    @tturbo warn_check_args = false for i in 1:length(x)
        s += abs2(x[i] - y[i])
    end
    s
end

"|x-y|, assumes size(x)==size(y), does not autodiff"
norm(x, y) = sqrt(normsq(x, y))

"(x-y)ᵀA(x-y), does not autodiff"
function normsq(x, y, A)
    s = zero(eltype(x))
    @tturbo warn_check_args = false for i in 1:length(x), j in 1:length(y)
        s += (x[i] - y[i]) * A[i, j] * (x[j] - y[j])
    end
    s
end
