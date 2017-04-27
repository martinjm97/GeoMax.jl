#############
# contract! #
#############

macro contract!(c_expr, a_expr, b_expr)
    @assert c_expr.head == :ref && a_expr.head == :ref && b_expr.head == :ref

    a_name = a_expr.args[1]
    b_name = b_expr.args[1]
    c_name = c_expr.args[1]

    a_subscripts = a_expr.args[2:end]
    b_subscripts = b_expr.args[2:end]
    c_subscripts = c_expr.args[2:end]

    @assert setdiff(a_subscripts, b_subscripts) == setdiff(c_subscripts, b_subscripts)
    @assert setdiff(b_subscripts, a_subscripts) == setdiff(c_subscripts, a_subscripts)

    contraction_indices = findin(a_subscripts, b_subscripts)

    # Here, we build up the kernel which performs the sum
    # reduction over the matching `a` and `b` indices.
    kernel_result = gensym()
    kernel = :($kernel_result += $a_expr * $b_expr)
    for i in contraction_indices
        kernel = quote
            @inbounds for $(a_subscripts[i]) in indices($a_name, $i)
                $kernel
            end
        end
    end
    kernel = quote
        $kernel_result = zero(eltype($c_name))
        $kernel
        $c_expr = $kernel_result
    end

    # Here, we build up the loop over the `c` indices.
    # Note our indexing pattern: looping over the left-most
    # index in the inner-most loop, and the second left-most
    # index in the second inner-most loop, etc. This should
    # linearly index over `c`, due to how Julia lays out the
    # array in memory. However, this indexing is not necessarily
    # linear over `a` and `b` (figuring that out how to do that
    # would be more complicated).
    loop = kernel
    i = 1
    for subscript in c_subscripts
        loop = quote
            @inbounds for $subscript in indices($c_name, $i)
                $loop
            end
        end
        i += 1
    end

    return esc(loop)
end

################
# tensor_*_dot #
################

function tensor_single_dot(a::AbstractArray{<:Real,3},
                           b::AbstractArray{<:Real,3})
    c = similar(a, size(a, 1), size(b, 3))
    return tensor_double_dot!(c, a, b)
end

function tensor_single_dot!(c::AbstractArray{<:Real,4},
                            a::AbstractArray{<:Real,3},
                            b::AbstractArray{<:Real,3})
    return @contract!(c[i,j,l,m], a[i,j,k], b[k,l,m])
end

# same as np.tensordot(x, y)
function tensor_double_dot(a::AbstractArray{<:Real,3},
                           b::AbstractArray{<:Real,3})
    c = similar(a, size(a, 1), size(b, 3))
    return tensor_double_dot!(c, a, b)
end

function tensor_double_dot!(c::AbstractArray{<:Real,2},
                            a::AbstractArray{<:Real,3},
                            b::AbstractArray{<:Real,3})
    return @contract!(c[i,l], a[i,j,k], b[j,k,l])
end

# TODO: fix this
# function tensor_double_dot(a::AbstractArray{<:Real,2},
#                            b::AbstractArray{<:Real,2})
#     c = 0
#     return tensor_double_dot!(c, a, b)
# end

# TODO: fix this
# function tensor_double_dot!(c::AbstractArray{<:Real,1},
#                             a::AbstractArray{<:Real,2},
#                             b::AbstractArray{<:Real,2})
#     return @contract!(c, a[i,j], b[i,j])
# end

tensor_full_dot(x, y) = vecdot(x, y)

#####################
# multiprod methods #
#####################

multiprod(A, B) = A * B

function multiprod(a::AbstractArray{A,3}, b::AbstractArray{B,3}) where {A,B}
    c = similar(a, promote_type(A, B), size(a, 1), size(a, 2), size(b, 3))
    return multiprod!(c, a, b)
end

function multiprod!(c::AbstractArray{<:Real,3},
                    a::AbstractArray{<:Real,3},
                    b::AbstractArray{<:Real,3})
    return @contract!(c[i, j, r], a[i, j, k], b[i, k, r])
end

##################
# multi* methods #
##################

# TODO rewrite with ::AbstractMatrix
function multitransp(A::AbstractArray)
    if ndims(A) == 2
        return A'
    end
    return permutedims(A, [1, 3, 2])
end

function multisym(A::AbstractArray)
    return 0.5 * (A + multitransp(A))
end

function multieye(k,n)
    return repeat(eye(n), (k, 1, 1))
end

# TODO maybe do type checking of matrices. Split into errors and Hermetian
function multilog(A::AbstractArray)
    @assert LinAlg.issymmetric(A) "not implemented"
    @assert LinAlg.isposdef(A) "not implemented"
    l, v = Linalg.eig(Hermetian(A))
    l =  reshape(log.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end

function multiexp(A::AbstractArray)
    @assert LinAlg.issymmetric(A) "not implemented"
    l, v = Linalg.eig(Hermetian(A))
    l =  reshape(exp.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end
