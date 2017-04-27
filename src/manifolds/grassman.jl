struct Grassmann <: AbstractManifold
    n::Int
    p::Int
    k::Int
    function Grassmann(n::Int, p::Int, k::Int)
        @assert n >= p | p >= 1 "Need n >= p >= 1"
        @assert k >= 1 "Need k >= 1"
        return new(n, p, k)
    end
end
Grassmann(m,n) = Grassmann(m,n,1)

dim(s::Grassmann) = s.k*(s.n * s.p - s.p^2)

typicaldist(s::Grassmann) = sqrt(s.p, s.k)

inner(::Grassmann, ::Any, U, V) = full_tensor_dot(U,V)

Base.norm(::Grassmann, ::Any, U) = norm(U)

function dist(::Grassmann, X, Y)
    u, s, v = svd(multiprod(multitransp(X), Y))
    s[s .> 1] = one(eltype(s))
    return norm(acos(s))
end

proj(::Grassmann, X, U) = U - multiprod(X, multiprod(multitransp(X), U))

egrad2rgrad(s::Grassmann, X, U) = proj(s, X, U)

function ehess2rhess(s::Grassmann, X, egrad, ehess, H)
    PXehess = proj(s, X, ehess)
    XtG = multiprod(multitransp(X), egrad)
    HXtG = multiprod(H, XtG)
    return PXehess - HXtG
end

function Base.exp(s::Grassmann, X, U)
    u, s, vt = svd(U)
    cos_s = reshape(cos(s), size(s)[1:(end-1)]...,1,size(s)[end])
    sin_s = reshape(sin(s), size(s)[1:(end-1)]...,1,size(s)[end])

    Y = (multiprod(multiprod(X, multitransp(vt) * cos_s), vt) +
         multiprod(u * sin_s, vt))
    for i in 1:s.k
        Y[i], unused = qr(Y[i])
    end
    return Y
end

function retr(::Grassmann, X, G)
    u, s, vt = svd(X + G)
    return multiprod(u, vt)
end

function Base.log(::Grassmann, X, Y)
    ytx = multiprod(multitransp(Y), X)
    At = multitransp(Y) - multiprod(ytx, multitransp(X))
    Bt = ytx\At
    u, s, vt = svd(multitransp(Bt))
    arctan_s = reshape(atan.(s), (size(s)[1:(end-1)]...,1,size(s)[end]))


    U = multiprod(u * arctan_s', vt)
    return U
end

function Base.rand(s::Grassmann)
    if s.k == 1
        X = randn(s.n, s.p)
        q, r = qr(X)
        return q
    end

    X = zeros(s.k, s.n, s.p)

    for i in 1:s.k
        X[i], r = qr(randn(s.n, s.p))
    end
    return X
end

function randvec(s::Grassmann,X)
    U = randn(size(X)...)
    U = proj(s, X, U)
    return  U / norm(U)
end

transp(s::Grassmann, ::Any, Y, U) = proj(s, Y, U)

pairmean(s::Grassmann, X, Y) = @assert false "Not implemented"
