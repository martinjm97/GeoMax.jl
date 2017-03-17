using ForwardDiff

# TODO: Figure out how none checking works in julia

abstract type AbstractProblem end

struct Problem{M <: AbstractManifold,C, T} <: AbstractProblem
    manifold::M
    _cost::C
    _grad::Vector{T}
    hess::Matrix{T}
    verbosity :: Int8
end

function grad!(p::Problem)
    if p._grad == nothing
        egrad = p.egrad
# TODO: figure out how manifolds work
# can do this without using the gradient!
#        grad(x) = p.manifold.egrad2rgrad(x, egrad(x))
        self._grad = grad

function hess!(p::Problem)
    if p._hess == nothing
        ehess = ForwardDiff.hessian(p.manifold)
# TODO: compute hessian
    #        hess(x, a) = self.manifold.ehess2rhess(x, self.egrad(x), ehess(x, a), a)
#        self._hess = hess
    end
#    return self._hess
end
