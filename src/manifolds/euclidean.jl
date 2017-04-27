struct Euclidean <: AbstractManifold
    shape::Vector{Int}
    function Euclidean(shape)
        @assert shape != 0 "cannot initialize zero as the dimension"
        return new(shape)
    end
end

dim(s::Euclidean) = prod(s.shape)

typicaldist(s::Euclidean) = sqrt(dim(s))

inner(::Euclidean, ::Any, U, V) = vecdot(U,V)

Base.norm(::Euclidean, ::Any, U) = vecnorm(U)

dist(::Euclidean, X, Y) = norm(X .- Y)

proj(::Euclidean, ::Any, U) = U

egrad2rgrad(::Euclidean, ::Any, U) = U

ehess2rhess(::Euclidean, ::Any, ::Any, ehess, ::Any) = ehess

Base.exp(::Euclidean, X, U) = X + U

Base.log(::Euclidean, X, Y) = Y - X

retr(s::Euclidean, X, U) = exp(s, X, U)

Base.rand(s::Euclidean) = randn(s.shape...)

function randvec(s::Euclidean, X)
    Y = rand(s)
    return Y / norm(s, X, Y)
end

transp(s::Euclidean, X1, X2, G) = G

pairmean(s::Euclidean, X, Y) = .5 * (X .+ Y)

struct Symmetric <: AbstractManifold
    n::Int
    k::Int
end
Symmetric(n) = Symmetric(n,1)

dim(s::Symmetric) = 0.5 * s.k * s.n * (s.n + 1)

typicaldist(s::Symmetric) = sqrt(dim(s))

inner(::Symmetric, ::Any, U, V) = vecdot(U,V)

Base.norm(::Symmetric, ::Any, U) = norm(U)

dist(::Symmetric, ::Any, Y) = norm(X .- Y)

proj(::Symmetric, ::Any, U) = multisym(U)

egrad2rgrad(::Symmetric, ::Any, U) = multisym(U)

ehess2rhess(::Symmetric, ::Any, ::Any, ehess, ::Any) = multisym(ehess)

Base.exp(::Symmetric, X, U) = X + U

Base.log(::Symmetric, X, Y) = Y - X

retr(s::Symmetric, X, U) = exp(s::Symmetric, X, U)

Base.rand(s::Symmetric) = multisym(randn(s.shape...))

randvec(s::Symmetric, X) = multisym(Y / norm(X, rand()))

transp(s::Symmetric, X1, X2, G) = G

pairmean(s::Symmetric, X, Y) = .5 * (X .+ Y)
