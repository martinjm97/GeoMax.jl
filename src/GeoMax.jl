module GeoMax

using StaticArrays

include("tools/multi.jl")

include("manifolds/sphere.jl")
include("manifolds/grassmann.jl")
include("manifolds/oblique.jl")
include("manifolds/product.jl")

include("manifolds/euclidean.jl")
include("manifolds/fixed_rank.jl")
include("manifolds/stiefel.jl")
include("manifolds/positive_definite.jl")

include("problem.jl")

end
