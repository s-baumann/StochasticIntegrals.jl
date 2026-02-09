using Test

@testset "Regression Tests - Pinned Numerical Values" begin
    using UnivariateFunctions
    using StochasticIntegrals
    using LinearAlgebra
    using StableRNGs

    tol = 10 * eps()

    # Setup: Two ItoIntegrals with constant vols on correlated Brownians
    # vol_A = 0.1, vol_B = 0.2, correlation = 0.5
    ito_A = ItoIntegral(:BM1, 0.1)
    ito_B = ItoIntegral(:BM2, 0.2)

    brownian_corr = Hermitian([1.0 0.5; 0.0 1.0])
    brownian_ids = [:BM1, :BM2]
    itos = Dict(:A => ito_A, :B => ito_B)

    ito_set = ItoSet(brownian_corr, brownian_ids, itos)

    # Analytical expected values for constant vols over [0,1]:
    #   var_A = 0.1^2 * 1 = 0.01
    #   var_B = 0.2^2 * 1 = 0.04
    #   cov_AB = 0.5 * 0.1 * 0.2 * 1 = 0.01
    #   det = var_A * var_B - cov_AB^2 = 0.0004 - 0.0001 = 0.0003

    # --- ForwardCovariance over [0.0, 1.0] ---
    fc = ForwardCovariance(ito_set, 0.0, 1.0)

    @testset "Covariance matrix elements" begin
        @test abs(variance(fc, :A) - 0.010000000000000002) < tol
        @test abs(variance(fc, :B) - 0.04000000000000001) < tol
        @test abs(covariance(fc, :A, :B) - 0.010000000000000002) < tol
    end

    @testset "Correlation" begin
        @test abs(correlation(fc, :A, :B) - 0.5) < tol
    end

    @testset "Determinant" begin
        @test abs(fc.determinant_ - 0.0003000000000000001) < tol
    end

    @testset "Inverse matrix" begin
        # Look up indices for :A and :B
        idx_A = findfirst(fc.covariance_labels_ .== :A)
        idx_B = findfirst(fc.covariance_labels_ .== :B)
        @test abs(fc.inverse_[idx_A, idx_A] - 133.33333333333331) < tol
        @test abs(fc.inverse_[idx_A, idx_B] - (-33.33333333333333)) < tol
        @test abs(fc.inverse_[idx_B, idx_B] - 33.33333333333333) < tol
    end

    @testset "Cholesky decomposition" begin
        idx_A = findfirst(fc.covariance_labels_ .== :A)
        idx_B = findfirst(fc.covariance_labels_ .== :B)
        @test abs(fc.chol_[idx_A, idx_A] - 0.1) < tol
        @test abs(fc.chol_[idx_B, idx_A] - 0.10000000000000002) < tol
        @test abs(fc.chol_[idx_B, idx_B] - 0.17320508075688776) < tol
        @test abs(fc.chol_[idx_A, idx_B] - 0.0) < tol
    end

    @testset "Volatility at a point" begin
        @test abs(volatility(ito_A, 0.5) - 0.1) < tol
        @test abs(volatility(ito_B, 0.5) - 0.2) < tol
    end

    @testset "PDF at known point" begin
        coords = Dict{Symbol,Real}(:A => 0.1, :B => 0.2)
        pdf_val = StochasticIntegrals.pdf(fc, coords)
        @test abs(pdf_val - 4.71769488544796) < 1e-10
    end

    @testset "Log-likelihood at known point" begin
        coords = Dict{Symbol,Real}(:A => 0.1, :B => 0.2)
        ll_val = log_likelihood(fc, coords)
        @test abs(ll_val - 5.227074441396715) < 1e-10
    end

    @testset "Seeded random draws (Stable_RNG)" begin
        rng = Stable_RNG(StableRNG(42), UInt(2))
        draws = get_draws(fc, 5; number_generator = rng)

        expected = [
            (:A, 1, 0.02032107963392913),
            (:B, 1, -0.1309442007257826),
            (:A, 2, 0.18977193393213676),
            (:B, 2, 0.30305111762346953),
            (:A, 3, -0.0950344767696638),
            (:B, 3, -0.0018177877896891348),
            (:A, 4, -0.014832386513930089),
            (:B, 4, 0.1334128796733172),
            (:A, 5, 0.07467208151283099),
            (:B, 5, 0.1699246290695593),
        ]

        for (sym, idx, val) in expected
            @test abs(draws[idx][sym] - val) < tol
        end
    end

    @testset "Conditional distribution" begin
        cond_draws = Dict{Symbol,Float64}(:A => 0.05)
        cond_mu, cond_sigma, cond_labels = generate_conditioned_distribution(fc, cond_draws)

        @test cond_labels == [:B]
        @test abs(cond_mu[1] - 0.05) < tol
        @test abs(cond_sigma[1, 1] - 0.030000000000000006) < tol
    end

    @testset "SimpleCovariance matches ForwardCovariance" begin
        sc = SimpleCovariance(ito_set, 0.0, 1.0)

        # Same covariance matrix as ForwardCovariance
        @test abs(variance(fc, :A) - sc.covariance_[findfirst(sc.covariance_labels_ .== :A), findfirst(sc.covariance_labels_ .== :A)]) < tol
        @test abs(variance(fc, :B) - sc.covariance_[findfirst(sc.covariance_labels_ .== :B), findfirst(sc.covariance_labels_ .== :B)]) < tol
    end

    @testset "SimpleCovariance after update!" begin
        sc = SimpleCovariance(ito_set, 0.0, 1.0)
        update!(sc, 0.0, 2.0)

        idx_A = findfirst(sc.covariance_labels_ .== :A)
        idx_B = findfirst(sc.covariance_labels_ .== :B)

        # Covariance scales linearly with duration (constant vols)
        @test abs(sc.covariance_[idx_A, idx_A] - 0.020000000000000004) < tol
        @test abs(sc.covariance_[idx_A, idx_B] - 0.020000000000000004) < tol
        @test abs(sc.covariance_[idx_B, idx_B] - 0.08000000000000002) < tol

        # Determinant scales as duration^n where n = number of dimensions
        @test abs(sc.determinant_ - 0.0012000000000000003) < tol

        # Cholesky scales as sqrt(duration)
        @test abs(sc.chol_[idx_A, idx_A] - 0.14142135623730953) < tol
        @test abs(sc.chol_[idx_B, idx_A] - 0.14142135623730953) < tol
        @test abs(sc.chol_[idx_B, idx_B] - 0.24494897427831785) < tol

        # Inverse scales as 1/duration
        @test abs(sc.inverse_[idx_A, idx_A] - 66.66666666666666) < tol
        @test abs(sc.inverse_[idx_A, idx_B] - (-16.666666666666664)) < tol
        @test abs(sc.inverse_[idx_B, idx_B] - 16.666666666666664) < tol
    end

    # ================================================================
    # Additional regression tests for coverage gaps
    # ================================================================

    @testset "ItoIntegral-level variance and covariance" begin
        # Direct ItoIntegral variance: integral of f^2 from 0 to 1
        @test abs(variance(ito_A, 0.0, 1.0) - 0.010000000000000002) < tol
        @test abs(variance(ito_B, 0.0, 1.0) - 0.04000000000000001) < tol

        # Direct ItoIntegral covariance: corr * integral of f1*f2
        @test abs(covariance(ito_A, ito_B, 0.0, 1.0, 0.5) - 0.010000000000000002) < tol

        # Edge case: from == to returns 0.0
        @test variance(ito_A, 1.0, 1.0) == 0.0
        @test covariance(ito_A, ito_B, 1.0, 1.0, 0.5) == 0.0
    end

    @testset "Date-based API variants" begin
        using Dates
        base_date = Date(2000, 1, 1)
        from_date = Date(2020, 1, 1)
        to_date   = Date(2021, 1, 1)

        # Date-based ItoIntegral variance (constant vol, ~1 year duration)
        @test abs(variance(ito_A, base_date, from_date, to_date) - 0.010020533880903487) < tol

        # Date-based ForwardCovariance constructor
        fc_date = ForwardCovariance(ito_set, from_date, to_date)
        @test abs(fc_date.from_ - 49.998631074606436) < tol
        @test abs(fc_date.to_ - 51.00068446269678) < tol
        @test abs(variance(fc_date, :A) - 0.010020533880903404) < tol
        @test abs(covariance(fc_date, :A, :B) - 0.010020533880903404) < tol
    end

    @testset "Non-constant integrand (PE_Function with mean reversion)" begin
        # vol * exp(-mr * t), mr=0.15, vol=0.01
        # variance = integral of (0.01 * exp(-0.15*t))^2 from 0 to T
        using Dates
        pe_func = PE_Function(0.01, 0.15, Date(2000, 1, 1), 0)
        ito_mr = ItoIntegral(:BM1, pe_func)
        @test abs(variance(ito_mr, 0.0, 1.0) - 1.439791398735765e-8) < tol
        @test abs(variance(ito_mr, 0.0, 5.0) - 1.432836866756899e-7) < tol
    end

    @testset "ForwardCovariance rolling to new period" begin
        fc_rolled = ForwardCovariance(fc, 0.0, 0.5)
        idx_A = findfirst(fc_rolled.covariance_labels_ .== :A)
        idx_B = findfirst(fc_rolled.covariance_labels_ .== :B)

        # Half the duration → half the variance (constant vols)
        @test abs(variance(fc_rolled, :A) - 0.005000000000000001) < tol
        @test abs(variance(fc_rolled, :B) - 0.020000000000000004) < tol
        @test abs(covariance(fc_rolled, :A, :B) - 0.005000000000000001) < tol
        @test abs(fc_rolled.determinant_ - 7.500000000000001e-5) < tol
        @test abs(fc_rolled.chol_[idx_A, idx_A] - 0.07071067811865477) < tol
        @test abs(fc_rolled.chol_[idx_B, idx_A] - 0.07071067811865475) < tol
        @test abs(fc_rolled.chol_[idx_B, idx_B] - 0.12247448713915891) < tol
    end

    @testset "get_zero_draws" begin
        zd = get_zero_draws(fc)
        @test zd[:A] == 0.0
        @test zd[:B] == 0.0
        @test Set(keys(zd)) == Set([:A, :B])

        zd_multi = get_zero_draws(fc, 3)
        @test length(zd_multi) == 3
        for d in zd_multi
            @test d[:A] == 0.0
            @test d[:B] == 0.0
        end
    end

    @testset "ForwardCovariance optional flags" begin
        fc_nochol = ForwardCovariance(ito_set, 0.0, 1.0; calculate_chol = false)
        fc_noinv  = ForwardCovariance(ito_set, 0.0, 1.0; calculate_inverse = false)
        fc_nodet  = ForwardCovariance(ito_set, 0.0, 1.0; calculate_determinant = false)

        # Disabled fields are missing
        @test ismissing(fc_nochol.chol_)
        @test ismissing(fc_noinv.inverse_)
        @test ismissing(fc_nodet.determinant_)

        # Covariance matrix still computed correctly
        @test abs(variance(fc_nochol, :A) - 0.010000000000000002) < tol
        @test abs(variance(fc_noinv, :A) - 0.010000000000000002) < tol
        @test abs(variance(fc_nodet, :A) - 0.010000000000000002) < tol
    end

    @testset "Number generator next! pinned values" begin
        using Distributions: MersenneTwister
        using Sobol

        # Mersenne
        rng_m = Mersenne(MersenneTwister(42), UInt(2))
        m1 = StochasticIntegrals.next!(rng_m)
        @test abs(m1[1] - 0.7108238673434464) < tol
        @test abs(m1[2] - 0.0644852510983267) < tol
        m2 = StochasticIntegrals.next!(rng_m)
        @test abs(m2[1] - 0.477842641066915) < tol
        @test abs(m2[2] - 0.17770930557953246) < tol

        # Sobol (deterministic quasi-random)
        rng_s = SobolGen(SobolSeq(2))
        s1 = StochasticIntegrals.next!(rng_s)
        @test abs(s1[1] - 0.5) < tol
        @test abs(s1[2] - 0.5) < tol
        s2 = StochasticIntegrals.next!(rng_s)
        @test abs(s2[1] - 0.75) < tol
        @test abs(s2[2] - 0.25) < tol
        s3 = StochasticIntegrals.next!(rng_s)
        @test abs(s3[1] - 0.25) < tol
        @test abs(s3[2] - 0.75) < tol

        # Stable_RNG
        rng_st = Stable_RNG(StableRNG(42), UInt(2))
        st1 = StochasticIntegrals.next!(rng_st)
        @test abs(st1[1] - 0.5805148626851955) < tol
        @test abs(st1[2] - 0.19124147945818315) < tol
    end

    @testset "Antithetic variates with seed" begin
        rng_anti = Stable_RNG(StableRNG(42), UInt(2))
        draws_anti = get_draws(fc, 4; number_generator = rng_anti, antithetic_variates = true)

        # Pinned values
        @test abs(draws_anti[1][:A] - 0.02032107963392913) < tol
        @test abs(draws_anti[1][:B] - (-0.1309442007257826)) < tol

        # Antithetic pair: draw 2 = -draw 1
        @test abs(draws_anti[2][:A] - (-0.02032107963392913)) < tol
        @test abs(draws_anti[2][:B] - 0.1309442007257826) < tol

        # Exact antithetic cancellation
        @test draws_anti[1][:A] + draws_anti[2][:A] == 0.0
        @test draws_anti[1][:B] + draws_anti[2][:B] == 0.0
        @test draws_anti[3][:A] + draws_anti[4][:A] == 0.0
        @test draws_anti[3][:B] + draws_anti[4][:B] == 0.0
    end

    @testset "Data conversion round-trip preserves values" begin
        using DataFrames

        rng_conv = Stable_RNG(StableRNG(99), UInt(2))
        draws_orig = get_draws(fc, 3; number_generator = rng_conv)

        # Pin original draw values
        @test abs(draws_orig[1][:A] - 0.09074337541261285) < tol
        @test abs(draws_orig[1][:B] - (-0.1042853544867152)) < tol
        @test abs(draws_orig[2][:A] - (-0.08309299899760555)) < tol
        @test abs(draws_orig[2][:B] - 0.1568458688781476) < tol
        @test abs(draws_orig[3][:A] - 0.2273528974422485) < tol
        @test abs(draws_orig[3][:B] - 0.31672221183358557) < tol

        # Round-trip: draws → array → draws
        arr, labs = to_array(draws_orig; labels = [:A, :B])
        @test abs(arr[1, 1] - 0.09074337541261285) < tol
        @test abs(arr[1, 2] - (-0.1042853544867152)) < tol
        draws_rt = to_draws(arr; labels = labs)
        for i in 1:3
            @test abs(draws_rt[i][:A] - draws_orig[i][:A]) < tol
            @test abs(draws_rt[i][:B] - draws_orig[i][:B]) < tol
        end

        # Round-trip: draws → dataframe → draws
        df = to_dataframe(draws_orig; labels = [:A, :B])
        draws_rt2 = to_draws(df; labels = [:A, :B])
        for i in 1:3
            @test abs(draws_rt2[i][:A] - draws_orig[i][:A]) < tol
            @test abs(draws_rt2[i][:B] - draws_orig[i][:B]) < tol
        end
    end

    @testset "make_covariance_matrix" begin
        cov_mat, labels = make_covariance_matrix(ito_set, 0.0, 1.0)
        @test Set(labels) == Set([:A, :B])
        idx_A = findfirst(labels .== :A)
        idx_B = findfirst(labels .== :B)
        @test abs(cov_mat[idx_A, idx_A] - 0.010000000000000002) < tol
        @test abs(cov_mat[idx_B, idx_B] - 0.04000000000000001) < tol
        @test abs(cov_mat[idx_A, idx_B] - 0.010000000000000002) < tol
    end

    @testset "get_draws_matrix pinned values (Stable_RNG)" begin
        rng = Stable_RNG(StableRNG(42), UInt(2))
        mat, labels = get_draws_matrix(fc, 5; number_generator = rng)

        idx_A = findfirst(labels .== :A)
        idx_B = findfirst(labels .== :B)

        @test abs(mat[1, idx_A] - 0.02032107963392913) < tol
        @test abs(mat[1, idx_B] - (-0.1309442007257826)) < tol
        @test abs(mat[2, idx_A] - 0.18977193393213676) < tol
        @test abs(mat[2, idx_B] - 0.30305111762346953) < tol
        @test abs(mat[3, idx_A] - (-0.0950344767696638)) < tol
        @test abs(mat[3, idx_B] - (-0.0018177877896891348)) < tol
        @test abs(mat[4, idx_A] - (-0.014832386513930089)) < tol
        @test abs(mat[4, idx_B] - 0.1334128796733172) < tol
        @test abs(mat[5, idx_A] - 0.07467208151283099) < tol
        @test abs(mat[5, idx_B] - 0.1699246290695593) < tol
    end

    @testset "get_draws_matrix antithetic pinned values" begin
        rng = Stable_RNG(StableRNG(42), UInt(2))
        mat, labels = get_draws_matrix(fc, 4; number_generator = rng, antithetic_variates = true)

        idx_A = findfirst(labels .== :A)
        idx_B = findfirst(labels .== :B)

        @test abs(mat[1, idx_A] - 0.02032107963392913) < tol
        @test abs(mat[1, idx_B] - (-0.1309442007257826)) < tol
        @test abs(mat[2, idx_A] - (-0.02032107963392913)) < tol
        @test abs(mat[2, idx_B] - 0.1309442007257826) < tol

        # Exact cancellation
        @test mat[1, idx_A] + mat[2, idx_A] == 0.0
        @test mat[1, idx_B] + mat[2, idx_B] == 0.0
        @test mat[3, idx_A] + mat[4, idx_A] == 0.0
        @test mat[3, idx_B] + mat[4, idx_B] == 0.0
    end

    @testset "get_draws_matrix matches get_draws element-by-element" begin
        rng1 = Stable_RNG(StableRNG(42), UInt(2))
        rng2 = Stable_RNG(StableRNG(42), UInt(2))
        draws_dict = get_draws(fc, 5; number_generator = rng1)
        mat, labels = get_draws_matrix(fc, 5; number_generator = rng2)

        for i in 1:5
            for (j, lab) in enumerate(labels)
                @test abs(mat[i, j] - draws_dict[i][lab]) < tol
            end
        end
    end

    @testset "Confidence hypercube" begin
        hc = get_confidence_hypercube(fc, 0.95, 1000000)

        # Symmetric bounds around zero
        @test hc[:A][1] == -hc[:A][2]
        @test hc[:B][1] == -hc[:B][2]

        # Pinned cutoff values
        @test abs(hc[:A][2] - 0.22133666382055353) < 1e-10
        @test abs(hc[:B][2] - 0.44267332764110706) < 1e-10

        # Cutoff proportional to std dev (same number of std devs for each dimension)
        @test abs(hc[:A][2] / 0.1 - hc[:B][2] / 0.2) < 1e-10
    end
end

@testset "Regression Tests - ItoProcess" begin
    using UnivariateFunctions
    using StochasticIntegrals
    using LinearAlgebra
    using StableRNGs
    using Distributions: Exponential
    using DataFrames

    tol = 10 * eps()

    ito_A = ItoIntegral(:BM1, 0.1)
    ito_B = ItoIntegral(:BM2, 0.2)
    brownian_corr = Hermitian([1.0 0.5; 0.0 1.0])
    brownian_ids = [:BM1, :BM2]
    itos = Dict(:A => ito_A, :B => ito_B)
    ito_set = ItoSet(brownian_corr, brownian_ids, itos)
    drift_func = PE_Function(0.0)

    @testset "evolve! single process" begin
        # Zero drift, starting at 100.0, stochastic draw of 0.05
        proc = ItoProcess(0.0, 100.0, drift_func, ito_A)
        evolve!(proc, 0.05, 1.0)
        @test abs(proc.value - 100.05) < tol
        @test abs(proc.t0 - 1.0) < tol
    end

    @testset "evolve! with non-zero drift" begin
        # Constant drift 0.02, integral from 0 to 1 = 0.02
        drift_func2 = PE_Function(0.02)
        proc = ItoProcess(0.0, 100.0, drift_func2, ito_A)
        evolve!(proc, 0.05, 1.0)
        @test abs(proc.value - 100.07) < tol
        @test abs(proc.t0 - 1.0) < tol
    end

    @testset "evolve! Dict of processes" begin
        proc_A = ItoProcess(0.0, 100.0, drift_func, ito_A)
        proc_B = ItoProcess(0.0, 100.0, drift_func, ito_B)
        procs = Dict(:A => proc_A, :B => proc_B)
        stochs = Dict(:A => 0.05, :B => -0.03)
        evolve!(procs, stochs, 1.0)
        @test abs(procs[:A].value - 100.05) < tol
        @test abs(procs[:B].value - 99.97) < tol
    end

    @testset "evolve_covar_and_ito_processes! (ForwardCovariance)" begin
        proc_A = ItoProcess(0.0, 100.0, drift_func, ito_A)
        proc_B = ItoProcess(0.0, 100.0, drift_func, ito_B)
        procs = Dict(:A => proc_A, :B => proc_B)
        fc = ForwardCovariance(ito_set, 0.0, 1.0)
        rng = Stable_RNG(StableRNG(42), UInt(2))

        procs, fc_new = evolve_covar_and_ito_processes!(procs, fc, 2.0; number_generator = rng)
        @test abs(procs[:A].value - 100.02032107963393) < tol
        @test abs(procs[:B].value - 99.86905579927422) < tol
        @test abs(procs[:A].t0 - 2.0) < tol
        @test abs(fc_new.from_ - 1.0) < tol
        @test abs(fc_new.to_ - 2.0) < tol
    end

    @testset "evolve_covar_and_ito_processes! (SimpleCovariance)" begin
        proc_A = ItoProcess(0.0, 100.0, drift_func, ito_A)
        proc_B = ItoProcess(0.0, 100.0, drift_func, ito_B)
        procs = Dict(:A => proc_A, :B => proc_B)
        sc = SimpleCovariance(ito_set, 0.0, 1.0)
        rng = Stable_RNG(StableRNG(42), UInt(2))

        procs, sc_new = evolve_covar_and_ito_processes!(procs, sc, 2.0; number_generator = rng)
        # Should match ForwardCovariance result (same seed, same constant vols)
        @test abs(procs[:A].value - 100.02032107963393) < tol
        @test abs(procs[:B].value - 99.86905579927422) < tol
        @test abs(sc_new.from_ - 1.0) < tol
        @test abs(sc_new.to_ - 2.0) < tol
    end

    @testset "Synchronous time series (ForwardCovariance)" begin
        procs = Dict(:A => ItoProcess(0.0, 100.0, drift_func, ito_A),
                     :B => ItoProcess(0.0, 100.0, drift_func, ito_B))
        fc = ForwardCovariance(ito_set, 0.0, 1.0)
        rng = Stable_RNG(StableRNG(42), UInt(2))
        ts = make_ito_process_syncronous_time_series(procs, fc, 1.0, 5; number_generator = rng)

        @test nrow(ts) == 10  # 5 ticks × 2 assets
        # Pin first tick values
        row_A1 = findfirst(r -> r.Time == 2.0 && r.Name == :A, eachrow(ts))
        row_B1 = findfirst(r -> r.Time == 2.0 && r.Name == :B, eachrow(ts))
        @test abs(ts[row_A1, :Value] - 100.02032107963393) < tol
        @test abs(ts[row_B1, :Value] - 99.86905579927422) < tol
        # Pin last tick values
        row_A5 = findfirst(r -> r.Time == 6.0 && r.Name == :A, eachrow(ts))
        row_B5 = findfirst(r -> r.Time == 6.0 && r.Name == :B, eachrow(ts))
        @test abs(ts[row_A5, :Value] - 100.1748982317953) < tol
        @test abs(ts[row_B5, :Value] - 100.47362663785087) < tol
    end

    @testset "Synchronous time series (SimpleCovariance)" begin
        procs = Dict(:A => ItoProcess(0.0, 100.0, drift_func, ito_A),
                     :B => ItoProcess(0.0, 100.0, drift_func, ito_B))
        sc = SimpleCovariance(ito_set, 0.0, 1.0)
        rng = Stable_RNG(StableRNG(42), UInt(2))
        ts = make_ito_process_syncronous_time_series(procs, sc, 1.0, 5; number_generator = rng)

        @test nrow(ts) == 10
        # Should match ForwardCovariance results (constant vols)
        row_A1 = findfirst(r -> r.Time == 2.0 && r.Name == :A, eachrow(ts))
        row_B1 = findfirst(r -> r.Time == 2.0 && r.Name == :B, eachrow(ts))
        @test abs(ts[row_A1, :Value] - 100.02032107963393) < tol
        @test abs(ts[row_B1, :Value] - 99.86905579927422) < tol
    end

    @testset "Non-synchronous time series" begin
        procs = Dict(:A => ItoProcess(0.0, 100.0, drift_func, ito_A),
                     :B => ItoProcess(0.0, 100.0, drift_func, ito_B))
        fc = ForwardCovariance(ito_set, 0.0, 1.0)
        update_rates = Dict(:A => Exponential(5.0), :B => Exponential(5.0))
        ts = make_ito_process_non_syncronous_time_series(procs, fc, update_rates, 10;
            timing_twister = StableRNG(42), ito_number_generator = Stable_RNG(StableRNG(99), UInt(2)))

        @test nrow(ts) == 10
        # Pin first and last row
        @test abs(ts[1, :Time] - 3.414714738165395) < tol
        @test ts[1, :Name] == :B
        @test abs(ts[1, :Value] - 99.83794731237703) < tol
        @test abs(ts[10, :Time] - 34.348763079102085) < tol
        @test ts[10, :Name] == :A
        @test abs(ts[10, :Value] - 100.19794141009473) < tol
    end
end
