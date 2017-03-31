struct FixedRank <: AbstractManifold
    m::Int
    n::Int
    k::Int
end

dim(s::FixedRank) = prod(s.d) - 1

typicaldist(s::FixedRank) = pi

inner(s::FixedRank, X, U, V) = dot(U,V)

norm(s::FixedRank, X, U) = norm(U)


function dist(s::FixedRank, U, V)
    inner_prod = max(min(inner(s,nothing, U, V), 1), -1)
    return acos(inner_prod)
end

proj(s::FixedRank, X, H) = H - inner(s,nothing, X, H) * X

function ehess2rhess(s::FixedRank, X, egrad, ehess, U)
    return proj(s,X,ehess) - inner(s,nothing, X, egrad) * U
end

function exp(s::FixedRank, X, U)
    norm_U = norm(s,nothing,U)
    if norm_U > 1e-3
        return X * cos(norm_U) + U * sin(norm_U) / norm_U
    else
        return retr(s,U)
    end
end

retr(s::FixedRank, X, U) = _normalize(s, X + U)

function log(s::FixedRank, X, Y)
    P = proj(s,X,Y-X)
    distance = dist(s,X,Y)
    if dist > 1e-6
        P *= distance / norm(s,nothing,P)
    end
    return P
end

rand(s::FixedRank) = _normalize(randn(s.d...))

function randvec(s::FixedRank,X)
    H = randn(s.d...)
    P = proj(s,X,H)
    return _normalize(P)
end

transp(s::FixedRank, X, Y, U) = proj(s, Y, U)

pairmean(s::FixedRank, X, Y) = _normalize(s, X + Y)

_normalize(s::FixedRank, X) = X / norm(s,nothing, X)
