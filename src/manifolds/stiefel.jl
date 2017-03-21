# TODO Check that initialization is valid (legal values of k)
struct Stiefel{} <: AbstractManifold
    n::Int
    p::Int
    k::Int
end
Stiefel(n, p) = Stiefel(n, p, 1)


function dim!(s::Stiefel)
    return s.k * (s.n * s.p - 0.5 * s.p * (s.p + 1))
end

function typicaldist!(s::Stiefel)
    return sqrt(s.p * s.k)
end

function inner!(s::Stiefel, X, G, H)
    return dot(G,H)
end

function norm!(s::Stiefel, X, G)
    return norm(G)
end

function dist!(s::Stiefel, U, V)
    throw("not implemented")
end

# TODO create multi implementations in proj nd ehess2rhess
# fix the 3 functions below
function proj!(s::Stiefel, X, U)
    throw("not implemented")
    return U - X * (permutedims(X) * U)
end

egrad2rgrad = proj

function ehess2rhess!(s::Stiefel, X, egrad, ehess, U)
    throw("not implemented")
    return self.proj(s,X,ehess) - self.inner(s,nothing, X, egrad) * U
end

function exp!(s::Stiefel, X, U)
    norm_U = norm(s,nothing,U)
    if norm_U > 1e-3
        return X * cos(norm_U) + U * sin(norm_U) / norm_U
    else
        return retr(s,U)
    end
end








function retr!(s::Stiefel, X, G)
    if s.k == 1
        q, r = qr(X + G)
        xNew = dot(q, diag(sign(sign(diag(r))+.5)))
    else:
        xNew = X + G
        for i = 1:s.k
            q, r = qr(xNew[i])
            xNew[i]  = dot(q, diag(sign(sign(diag(r)+.5))))
        end
    end
    return xNew
end

function log!(s::Stiefel, X, Y)
    throw("not implemented")
end

function rand!(s::Stiefel)
    if s.k == 1
        X = randn(s.n, s.p)
        q, r = qr(X)
        return q
    end

    X = zeros(s.k, s.n, s.p)
    for i = 1:s.k
        X[i], r = qr(randn(s.n, s.p))
    return X
end

function randvec!(s::Stiefel, X)
    U = randn(size(X)...)
    U = proj(s,X,U)
    U = U / norm(U)
    return U
end

function transp!(s::Stiefel, x1, x2, d)
    return proj(x2 d)
end

function pairmean!(s::Stiefel, X, Y)
    throw("not implemented")
end
