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
