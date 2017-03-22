module JManOpt

include("tools/multi.jl")

include("manifolds/sphere.jl")
include("manifolds/grassman.jl")
include("manifolds/oblique.jl")
include("manifolds/product.jl")

include("manifolds/euclidian.jl")
include("manifolds/fixed_rank.jl")
include("manifolds/stiefel.jl")
include("manifolds/positive_definite.jl")

include("problem.jl")

end
