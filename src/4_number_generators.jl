import Sobol.next!

"""
    A NumberGenerator must always return a vector of uniforms when called with the function next!(::NumberGenerator).
    It can contain whatever it wants as long as it does that method.
"""
abstract type NumberGenerator end

"""
    A NumberGenerator wrapper for SobolSeq.
"""
struct SobolGen <: NumberGenerator
    seq_::SobolSeq
end
function next!(number_generator::SobolGen)
    return next!(number_generator.seq_)
end

"""
    A NumberGenerator wrapper for MersenneTwister.
"""
struct Mersenne <: NumberGenerator
    twister_::MersenneTwister
    number_of_itos_::UInt
end
function next!(number_generator::Mersenne)
    return rand(number_generator.twister_, number_generator.number_of_itos_)
end

#"""
#    ImportanceSampler(underlying_generator_::NumberGenerator, conversion_function_::Function)
#    The importance sampler takes in a NumberGenerator that is used to get a random vector of uniforms,  [0,1]^d.
#    Then it inputs this vector into a function (the conversion_function): [0,1]^d -> [0,1]^d.
#    Note that if the identity function is input then the numbers generated are the same as the input
#    number generator. If on the other hand a convex function (in some dimension) is input it is less likely to
#    get a high number for that dimension.
#    If you are doing Importance Sampling remember to multiply the results by the inverse of the pdf later. This
#    package doesn't do that for you.
#"""
#struct ImportanceSampler <: NumberGenerator
#    underlying_generator_::NumberGenerator
#    conversion_function_::Function
#end
#function next!(number_generator::ImportanceSampler)
#    array_of_uniforms = next!(number_generator.underlying_generator_)
#    return number_generator.conversion_function_(array_of_uniforms)
#end
