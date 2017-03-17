abstract type AbstractManifold end

struct Sphere{} <: AbstractManifold
    m::Int
    n::Int
end

# TODO consider s<: AbstractManifold instead
# with if s::Sphere

function dim!(s::Sphere)
    return length(s) - 1
end

function typicaldist!(s::Sphere)
    return pi
end

function inner!(s::Sphere, X, U, V)
    return Float(dot(U,V))
end

function norm!(s::Sphere, X, U)
    return norm(U)
end

function dist!(s::Sphere, U, V)
    inner_prod = max(min(inner(s,nothing, U, V), 1), -1)
    return acos(inner_prod)
end

function proj!(s::Sphere, X, H)
    return H - self.inner(s,nothing, X, H) * X
end

function ehess2rhess!(s::Sphere, X, egrad, ehess, U)
    return self.proj(s,X,ehess) - self.inner(s,nothing, X, egrad) * U
end

function exp!(s::Sphere, X, U)
    norm_U = norm(s,nothing,U)
    if norm_U > 1e-3
        return X * cos(norm_U) + U * sin(norm_U) / norm_U
    else
        return retr(s,U)
    end
end

function retr!(s::Sphere, X, U)
    Y = X + U
    return _normalize(s,Y)

function log!(s::Sphere, X, Y)
    P = proj(s,X,Y-X)
    distance = dist(s,X,Y)
    if dist > 1e-6
        P *= distance / norm(s,nothing,P)
    return P
end

function rand!(s::Sphere)
    Y = randn(s.m,s.n)
    return self._normalize(Y)
end

function randvec!(s::Sphere,X)
    H = randn(s.m, s.n)
    P = proj(s,X,H)
    return self._normalize(P)
end


function transp!(s::Sphere, X, Y, U)
    return proj(s, Y, U)
end

function pairmean!(s::Sphere, X, Y)
    return _normalize(s, X + Y)
end

function _normalize!(s::Sphere, X)
    return X / norm(s,nothing, X)
end


# TODO SphereSubspaceComplementIntersection at the bottom of sphere.py
