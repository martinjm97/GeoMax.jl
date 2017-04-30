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

dist(::Euclidean, x, y) = vecnorm(x .- y)

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
    shape::Array{Int64,1}
end
Symmetric(n::Int64) = Symmetric([1,n])
Symmetric(k::Int64, n::Int64)= k==1 ? Symmetric(n): Symmetric([k, n, n])

dim(s::Symmetric) = 0.5 * s.shape[1] * s.shape[2] * (s.shape[2] + 1)

typicaldist(s::Symmetric) = sqrt(dim(s))

inner(::Symmetric, ::Any, U, V) = vecdot(U,V)

Base.norm(::Symmetric, ::Any, U) = vecnorm(U)

dist(::Symmetric, X, Y) = vecnorm(X .- Y)

proj(::Symmetric, ::Any, U) = multisym(U)

egrad2rgrad(::Symmetric, ::Any, U) = multisym(U)

ehess2rhess(::Symmetric, ::Any, ::Any, ehess, ::Any) = multisym(ehess)

Base.exp(::Symmetric, X, U) = X + U

Base.log(::Symmetric, X, Y) = Y - X

retr(s::Symmetric, X, U) = exp(s::Symmetric, X, U)

Base.rand(s::Symmetric) = multisym(randn(s.shape...))

function randvec(s::Symmetric, X)
    Y = rand(s)
    return multisym( Y / norm(s, X, Y))
end

transp(s::Symmetric, X1, X2, G) = G

pairmean(s::Symmetric, X, Y) = .5 * (X .+ Y)
