import Sobol.next!

"""
    A `NumberGenerator` must always return a vector of uniforms when called with the function `next!(::NumberGenerator)`.
    It can contain whatever it wants as long as it does that method.
"""
abstract type NumberGenerator end

"""
    A NumberGenerator wrapper for `SobolSeq`. This makes quasirandom numbers
"""
struct SobolGen <: NumberGenerator
    seq_::SobolSeq
end

"""
    A NumberGenerator wrapper for `MersenneTwister`. This makes pseudorandom numbers.
"""
struct Mersenne <: NumberGenerator
    twister_::MersenneTwister
    number_of_itos_::UInt
end

"""
    next!(number_generator::NumberGenerator)
This extracts a random draw given a number generator struct (like a `SobolGen` or a `Mersenne`).

"""
function next!(number_generator::SobolGen)
    return next!(number_generator.seq_)
end
function next!(number_generator::Mersenne)
    return rand(number_generator.twister_, number_generator.number_of_itos_)
end
