# TODO add undefined number of args
struct Euclidean <: AbstractManifold
    @assert shape != 0 "cannot initialize zero as the dimension"
    shape::Vector{Int}
end

dim(s::Euclidean) = prod(s.shape) - 1

typicaldist(s::Euclidean) = sqrt(dim(s))

# TODO FIGURE OUT TENSORDOT
# why are these the sames
# ones(5) - ones(5)
# ones(5) .- ones(5)

inner(s::Euclidean, X, U, V) = dot(U,V)

norm(s::Euclidean, X, U) = norm(U)

dist(s::Euclidean, X, Y) = norm(X .- Y)

proj(s::Euclidean, X, U) = U

egrad2grad(s::Euclidean, X, U) = U

ehess2rhess(s::Euclidean, X, egrad, ehess, U) = ehess

exp(s::Euclidean, X, U) = X + U

log(s::Euclidean, X, Y) = Y - X
# TODO is writing another exp alright?
retr(s::Euclidean, X, U) = exp(s::Euclidean, X, U)

rand(s::Euclidean) = randn(s.shape...)

function randvec(s::Euclidean,X)
    Y = rand()
    return Y / norm(X, Y)

transp(s::Euclidean, X1, X2, G) = G

pairmean(s::Euclidean, X, Y) = .5 * (X .+ Y)
