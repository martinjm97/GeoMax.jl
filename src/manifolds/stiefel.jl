struct Stiefel <: AbstractManifold
    n::Int
    p::Int
    k::Int
    function Stiefel(n::Int, p::Int, k::Int)
        @assert n >= p & p >= 1 "Need n >= p >= 1."
        @assert k >= 1 "k must be greater than or equal to 1"
        return new(n, p, k)
    end
end
Stiefel(n, p) = Stiefel(n, p, 1)

dim(s::Stiefel) = s.k * (s.n * s.p - 0.5 * s.p * (s.p + 1))

typicaldist(s::Stiefel) = sqrt(s.p * s.k)

inner(s::Stiefel, X, G, H) = vecdot(G,H)

Base.norm(s::Stiefel, X, G) =  norm(G)

function dist(s::Stiefel, U, V)
    @assert false "not implemented"
end

function proj(s::Stiefel, X, U)
    return U - multiprod(X, multisym(multiprod(multitransp(X), U)))
end

egrad2rgrad = proj

function ehess2rhess(s::Stiefel, X, egrad, ehess, H)
    XtG = multiprod(multitransp(X), egrad)
    symXtG = multisym(XtG)
    HsymXtG = multiprod(H, symXtG)
    return proj(s, X, ehess - HsymXtG)
end

# check exp works
function Base.exp(s::Stiefel, X, U)
    if s.k == 1
        m1 = hcat(dot(U, X'), (-dot(U, U')))
        m2 = hcat(eye(s.p), dot(U, X'))
        W = expm(vcat(m1,m2))
        Z = vcat(expm((-dot(U,X')), zeros(s.p, s.p)))
        Y = dot(dot(vcat(X, U), W), Z)
    else
        Y = zeros(size(X))
        for i in 1:s.k
            m1 = hcat(dot(U[i], X[i]'), - dot(U[i], U[i]'))
            m2 = hcat(eye(s.p), dot(U[i], X[i]'))
            W = expm(vcat(m1, m2))
            Z = vcat(expm((-dot(U[i], X[i]')), zeros(s.p, s.p)))
            Y[i] = dot(dot(vcat(X[i], U[i]), W), Z)
        end
    end
    return Y
end

function retr(s::Stiefel, X, G)
    if s.k == 1
        q, r = qr(X + G)
        xNew = dot(q, diag(sign(sign(diag(r))+.5)))
    else
        xNew = X + G
        for i in eachindex(s.k)
            q, r = qr(xNew[i])
            xNew[i]  = dot(q, diag(sign(sign(diag(r)+.5))))
        end
    end
    return xNew
end

function Base.log(s::Stiefel, X, Y)
    @assert false "not implemented"
end

function Base.rand(s::Stiefel)
    if s.k == 1
        X = randn(s.n, s.p)
        q, r = qr(X)
        return q
    end

    X = zeros(s.k, s.n, s.p)
    for i in eachindex(s.k)
        X[i], r = qr(randn(s.n, s.p))
    end
    return X
end

function randvec(s::Stiefel, X)
    U = randn(size(X)...)
    U = proj(s,X,U)
    U = U / norm(U)
    return U
end

function transp(s::Stiefel, x1, x2, d)
    return proj(x2, d)
end

function pairmean(s::Stiefel, X, Y)
    @assert false "not implemented"
end
