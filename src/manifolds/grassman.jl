struct Grassmann <: AbstractManifold
    @assert n >= p | p >= 1 "Need n >= p >= 1."
    @assert k >= 1 "Need k >= 1"
    n::Int
    p::Int
    k::Int
end
Grassman(m,n) = Grassman(m,n,1)


dim(s::Grassman) = s.k*(s.n * s.p - s.p^2)

typicaldist(s::Grassman) = sqrt(s.p, s.k)

# TODO figure out tensordot
inner(s::Grassman, X, U, V) = dot(U,V)

norm(s::Grassman, X, U) = norm(U)


function dist(s::Grassman, X, Y)
    u, s, v = svd(multiprod(multitransp(X), Y))
    s[s > 1] = 1
    s = acos(s)
    return norm(s)
end

proj(s::Grassman, X, U) = U - multiprod(X, multiprod(multitransp(X), U))

egrad2rgrad = proj

function ehess2rhess(s::Grassman, X, egrad, ehess, H)
    PXehess = proj(s, X, ehess)
    XtG = multiprod(multitransp(X), egrad)
    HXtG = multiprod(H, XtG)
    return PXehess - HXtG
end

# TODO fix this
function exp(s::Grassman, X, U)
    norm_U = norm(s,nothing,U)
    if norm_U > 1e-3
        return X * cos(norm_U) + U * sin(norm_U) / norm_U
    else
        return retr(s,U)
    end
end

function retr(s::Grassman, X, G)
    u, s, vt = svd(X + G, full_matrices=False)
    return multiprod(u, vt)
end

# TODO fix this
function log(s::Grassman, X, Y)
    ytx = multiprod(multitransp(Y), X)
    At = multitransp(Y) - multiprod(ytx, multitransp(X))
    #Bt = np.linalg.solve(ytx, At)
    #u, s, vt = svd(multitransp(Bt))
    # arctan_s = np.expand_dims(atan(s), -2)

    U = multiprod(u * arctan_s, vt)
    return U
end

# TODO check if can vectorize qr
# then combine two parts into 1
function rand(s::Grassman)
    if s.k == 1
        X = randn(s.n, s.p)
        q, r = qr(X)
        return q

    X = zeros(s.k, s.n, s.p)

    for i in 1:s.k
        X[i], r = qr(randn(s.n, s.p))
    return X
end

function randvec(s::Grassman,X)
    U = randn(*size(X))
    U = proj(s, X, U)
    return  U / norm(U)
end

transp(s::Grassman, X, Y, U) = proj(s, Y, U)

pairmean(s::Grassman, X, Y) = @assert false "Not implemented"
