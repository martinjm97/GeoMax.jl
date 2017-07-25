#using ForwardDiff

# TODO: Figure out how none checking works in julia

# abstract type AbstractProblem end
#
# struct Problem{M <: AbstractManifold} <: AbstractProblem
#     manifold::M
# #    cost::C
# #    egrad::Vector{T}
# #    ehess::Matrix{T}
# #    grad::Vector{T}
# #    hess::Matrix{T}
# #    verbosity :: Int8
# end
# Problem(manifold, cost, egrad=nothing, ehess=nothing, grad=nothing, hess=nothing, arg=nothing, precon=nothing)
#
# #function grad!(p::Problem)
# #    if p._grad == nothing
# #        egrad = p.egrad
# #    end
# #    return nothing
# # TODO: grad compute works
# # can do this without using the gradient!
# #        grad(x) = p.manifold.egrad2rgrad(x, egrad(x))
# #        self._grad = grad
# #end
#
# #function hess!(p::Problem)
# #    return nothing
# #    if p._hess == nothing
# #        ehess = ForwardDiff.hessian(p.manifold)
# # TODO: compute hessian
#     #        hess(x, a) = self.manifold.ehess2rhess(x, self.egrad(x), ehess(x, a), a)
# #        self._hess = hess
# #    end
# #    return self._hess
# #end
