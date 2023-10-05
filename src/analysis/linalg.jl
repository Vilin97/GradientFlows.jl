############ fast linear algebra ############
"sum of squares of the elements of z"
function normsq(z)
    res = zero(eltype(z))
    for zi in z
        res += zi * zi
    end
    res
end

"sum of squares of the elements of z1 - z2"
function normsq(z1, z2)
    res = zero(eltype(z1))
    for i in eachindex(z1)
        res += (z1[i] - z2[i]) * (z1[i] - z2[i])
    end
    res
end

function fastdot(z, v)
    res = zero(eltype(z))
    for i in eachindex(z)
        res += z[i] * v[i]
    end
    res
end