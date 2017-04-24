struct Euclidean <: AbstractManifold
    shape::Vector{Int}
    function Euclidean(shape)
        @assert shape != 0 "cannot initialize zero as the dimension"
        return new(shape)
    end
end

dim(s::Euclidean) = prod(size(s)) - 1

typicaldist(s::Euclidean) = sqrt(dim(s))

inner(::Euclidean, ::Any, U, V) = vecdot(U,V)

Base.norm(::Euclidean, ::Any, U) = norm(U)

dist(::Euclidean, X, Y) = norm(X .- Y)

proj(::Euclidean, ::Any, U) = U

egrad2grad(::Euclidean, ::Any, U) = U

ehess2rhess(::Euclidean, ::Any, ::Any, ehess, ::Any) = ehess

Base.exp(::Euclidean, X, U) = X + U

Base.log(::Euclidean, X, Y) = Y - X

retr(s::Euclidean, X, U) = exp(s, X, U)

Base.rand(s::Euclidean) = randn(size(s)...)

function randvec(::Euclidean, X)
    Y = rand(s)
    return Y / norm(X, Y)
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

inner(s::Symmetric, X, U, V) = vecdot(U,V)

Base.norm(s::Symmetric, X, U) = norm(U)

dist(s::Symmetric, X, Y) = norm(X .- Y)

proj(s::Symmetric, X, U) = multisym(U)

egrad2grad(s::Symmetric, X, U) = multisym(U)

ehess2rhess(s::Symmetric, X, egrad, ehess, U) = multisym(ehess)

Base.exp(s::Symmetric, X, U) = X + U

Base.log(s::Symmetric, X, Y) = Y - X

retr(s::Symmetric, X, U) = exp(s::Symmetric, X, U)

Base.rand(s::Symmetric) = multisym(randn(s.shape...))

randvec(s::Symmetric,X) = multisym(Y / norm(X, rand()))

transp(s::Symmetric, X1, X2, G) = G

pairmean(s::Symmetric, X, Y) = .5 * (X .+ Y)
