struct FixedRank <: AbstractManifold
    m::Int
    n::Int
    k::Int
end

dim(s::FixedRank) = (s.m + s.n - s.k) * s.k

typicaldist(s::FixedRank) = dim(s)

inner(::FixedRank, ::Any, G, H) = sum(tensor_double_dot(a, b) for (a, b) in zip(G, H))

Base.norm(s::FixedRank, X, G) = sqrt(inner(s, X, G, G))

dist(s::FixedRank, U, V) = @assert false "not implemented"

_apply_ambient(::FixedRank, Z::Tuple, W) = dot(Z[1], dot(Z[2], dot(Z[3]', W)))

_apply_ambient(::FixedRank, Z, W) = dot(Z, W)

_apply_ambient_transpose(::FixedRank, Z::Tuple, W) = dot(Z[3], dot(Z[2], dot(Z[1]', W)))

_apply_ambient_transpose(::FixedRank, Z, W) = dot(Z', W)

function proj(s::FixedRank, X, Z)
    ZV = _apply_ambient(s, Z, X[3]')
    UtZV = dot(X[1]', ZV)
    ZtU = _apply_ambient_transpose(s, Z, X[1])

    Up = ZV - dot(X[1], UtZV)
    M = UtZV
    Vp = ZtU - dot(X[3]', UtZV')

    return _TangentVector((Up, M, Vp))
end

function egrad2grad(s::FixedRank, X, egrad)
    utdu = dot(x[1]', egrad[1])
    uutdu = dot(x[1], utdu)
    Up = (egrad[1] - uutdu) / x[2]

    vtdv = dot(x[3], egrad[3]')
    vvtdv = dot(x[3]', vtdv)
    Vp = (egrad[3]' - vvtdv) / x[2]

    i = eye(s.k)
    extended_x = reshape(x[2], (1, size(x[2])...))
    f = 1 / (extended_x^2 - extended_x^2 + i)

    M = (f * (utdu - utdu.T) * x[2] + extended_x * f * (vtdv - vtdv') + diag(egrad[2]))

    return _TangentVector((Up, M, Vp))
end

ehess2rhess(s::FixedRank, X, egrad, ehess, U) = @assert false "not implemented"

Base.exp(s::FixedRank, X, Y) = @assert false "not implemented"

function retr(s::FixedRank, X, U)
    Qu, Ru = qr(Z[1])
    Qv, Rv = qr(Z[3])

    T = vcat(hcat(diag(X[2]) + Z[2], Rv.T), hcat(Ru, zeros(s.k, s.k)))

    Ut, St, Vt = svd(T)
    Vt = Vt'

    U = dot(hcat(X[1], Qu), Ut[:, 1:s.k])
    V = dot(hcat((X[3]', Qv)), Vt[:, 1:s.k])
    # We np.spacing(2) was added to the term below.
    S = St[1:s.k]
    return (U, S, V')
end

Base.log(s::FixedRank, X, Y) = @assert false "not implemented"

function Base.rand(s::FixedRank)
    u = rand(Steifel(s.m, s.k))
    s = reverse(sort(rand(s.k)))
    vt = rand(Steifel(s.n, s.k))'
    return (u, s, vt)
end

function randvec(s::FixedRank,X)
    Up = randn(s.m, s.k)
    Vp = randn(s.n, s.k)
    M = randn(s.k, s.k)

    Z = _tangent(s, X, (Up, M, Vp))
    nrm = norm(s, X, Z)

    return _TangentVector((Z[1]/nrm, Z[2]/nrm, Z[3]/nrm))
end

function _tangent(::FixedRank, X, Z)
    Up = Z[1] - dot(X[1], dot(X[1]', Z[1]))
    Vp = Z[3] - dot(X[3]', dot(X[3], Z[3]))

    return _TangentVector((Up, Z[2], Vp))
end

function tangent2ambient(s::FixedRank, X, Z)
    U = hcat(dot(X[1], Z[2]) + Z[1], X[1])
    S = eye(2 * s.k)
    V = hcat([X[3]', Z[3]])
    return (U, S, V)
end

pairmean(s::FixedRank, X, Y) = @assert false "not implemented"

transp(s::FixedRank, X1, X2, G) = proj(s, X2, tangent2ambient(s, X1, G))

zerovec(s::FixedRank, X) = _TangentVector((zeros(s.m, s.k), zeros(s.k, s.k), zeros(s.n, s.k)))

struct _TangentVector
    t::Tuple
end

function to_ambient(v::_TangentVector, x)
    Z1 = dot(dot(x[3], v[2]), x[1])
    Z2 = dot(x[3], v[1])
    Z3 = dot(x[1], v[3]')
    return Z1 + Z2 + Z3
end

# TODO overwrite addition
