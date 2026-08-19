using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_snapshots, load_subdomain_snapshot, isotropic_powerspectrum, δ
set_theme!(Theme(fontsize = 12))

# =============================================================================
# --- File directories ---
# =============================================================================
hy_filehead  = "./LESStudySetup/"
hy_filename  = hy_filehead * "hydrostatic_snapshots_hydrostatic_twin_simulation.jld2"
hy_metadata  = hy_filehead * "experiment_hydrostatic_twin_simulation_metadata.jld2"

nhy_filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/"

filesave = "figures/20260811_spectra/"

# =============================================================================
# --- Shared physical parameters ---
# =============================================================================
f  = parameters.f               # Coriolis parameter
Q  = 40                         # surface sensible heat flux
h₀ = 60                         # height of the convective BL / mixing depth
ρ₀ = parameters.ρ₀              # reference density
cₚ = parameters.cp              # heat capacity
α  = parameters.α               # thermal expansion
g  = parameters.g               # gravity
wₛ = (α * g * Q * h₀ / (ρ₀ * cₚ))^(1/3)   # convective velocity scale [m/s]

# --- Grid spacing: each sim uses its own Δh (set right before it's used) ---
hy_dx, hy_dy   = 156.25, 156.25       # [m] hydrostatic horizontal grid spacing
nhy_dx, nhy_dy = 4.88281, 4.88281     # [m] nonhydrostatic horizontal grid spacing
dz = 1.125                             # [m] vertical grid size (both sims)

# --- klev matched so both sims are compared at the same z ---
hy_klev  = 223       # z = -8.4375 m in the hydrostatic full-domain grid
nhy_klev = 65        # z = -8.4375 m in the nonhydrostatic subdomain grid

# --- Slice of the hydrostatic full domain matching nhy subdomain 97 ---
# (nhy subdomain 97 is a 10km x 10km tile; this xrange wraps the periodic x-boundary)
subdomain = 97
hy_xrange = vcat(583:640, 1:7)
hy_yrange = 391:455

# --- Day <-> snapshot_number (hy) / iteration (nhy) correspondence ---
days              = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
hy_snapshot_nums  = [9, 25, 41, 57, 73, 89, 105, 121]
nhy_iterations    = [22865, 42238, 62484, 82586, 103348, 123463, 143293, 164410]

day_labels = ["Day $(d)" for d in days]

colors = cgrad(:dense, length(days), categorical = true)

# =============================================================================
# --- Hydrostatic spectra (full-domain snapshots, sliced to match nhy tile) ---
# =============================================================================
function hyspectrum_uvwT(snapshots, snapshot_number, klev, xrange, yrange, dx, dy)
    times = snapshots[:T].times
    nday = @sprintf("%2.1f", (times[snapshot_number]) / 60^2 / 24)
    println("Reading hy snapshot $snapshot_number on day $(nday)...")

    T = snapshots[:T][snapshot_number]
    u = snapshots[:u][snapshot_number]
    v = snapshots[:v][snapshot_number]
    w = snapshots[:w][snapshot_number]

    Su = isotropic_powerspectrum(interior(u, xrange, yrange, klev), interior(u, xrange, yrange, klev); Δx=dx, Δy=dy)
    Sv = isotropic_powerspectrum(interior(v, xrange, yrange, klev), interior(v, xrange, yrange, klev); Δx=dx, Δy=dy)
    wk = (interior(w, xrange, yrange, klev) + interior(w, xrange, yrange, klev+1)) / 2
    Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
    St = isotropic_powerspectrum(interior(T, xrange, yrange, klev), interior(T, xrange, yrange, klev); Δx=dx, Δy=dy)

    return St, Su, Sv, Sw
end

# =============================================================================
# --- Nonhydrostatic spectra (subdomain tile snapshots) ---
# =============================================================================
function nhyspectrum_uvwT(iteration, nday, subdomain, klev, dx, dy)
    println("Reading nhy iteration $(iteration) on day $(nday), subdomain $(subdomain)...")
    t0 = now()

    fileparam = "subdomain" * string(subdomain)
    output_filename = nhy_filehead * fileparam * "_iter$(iteration).jld2"
    snapshot = load_subdomain_snapshot(output_filename)

    T = snapshot[:T]
    u = snapshot[:u]
    v = snapshot[:v]
    w = snapshot[:w]

    println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")

    Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev); Δx=dx, Δy=dy)
    Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev); Δx=dx, Δy=dy)
    wk = (interior(w, :, :, klev) + interior(w, :, :, klev+1)) / 2
    Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
    St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)

    return St, Su, Sv, Sw
end

# # =============================================================================
# # --- Compute spectra across all days for a chosen sim: "hy", "nhy", or "both" ---
# # Returns a NamedTuple of Vectors, one entry per day, for T, u, v, w spectra.
# # =============================================================================
# function compute_spectra_evol(sim::String)
#     if sim == "hy"
#         set_value!(; Δh = hy_dx)
#         println("Loading hydrostatic data from $hy_filename...")
#         snapshots = load_snapshots(hy_filename; metadata=hy_metadata)

#         St, Su, Sv, Sw = ntuple(_ -> Vector{Any}(undef, length(days)), 4)
#         for (i, snum) in enumerate(hy_snapshot_nums)
#             St[i], Su[i], Sv[i], Sw[i] = hyspectrum_uvwT(snapshots, snum, hy_klev, hy_xrange, hy_yrange, hy_dx, hy_dy)
#         end
#         return (; St, Su, Sv, Sw)

#     elseif sim == "nhy"
#         set_value!(; Δh = nhy_dx)
#         St, Su, Sv, Sw = ntuple(_ -> Vector{Any}(undef, length(days)), 4)
#         for (i, iter) in enumerate(nhy_iterations)
#             St[i], Su[i], Sv[i], Sw[i] = nhyspectrum_uvwT(iter, days[i], subdomain, nhy_klev, nhy_dx, nhy_dy)
#         end
#         return (; St, Su, Sv, Sw)

#     else
#         error("compute_spectra_evol expects \"hy\" or \"nhy\" (call both separately for overlay plots).")
#     end
# end

# =============================================================================
# --- In-memory cache: avoids recomputing hy/nhy spectra across multiple
#     spectra_uvwT(...) calls within the same Julia session. ---
# =============================================================================
const SPECTRA_CACHE = Dict{String, NamedTuple}()
 
function compute_spectra_evol(sim::String; force_recompute=false)
    if !force_recompute && haskey(SPECTRA_CACHE, sim)
        println("Using cached $sim spectra (already computed this session)...")
        return SPECTRA_CACHE[sim]
    end
 
    result = _compute_spectra_evol_uncached(sim)
    SPECTRA_CACHE[sim] = result
    return result
end
 
function _compute_spectra_evol_uncached(sim::String)
    if sim == "hy"
        set_value!(; Δh = hy_dx)
        println("Loading hydrostatic data from $hy_filename...")
        snapshots = load_snapshots(hy_filename; metadata=hy_metadata)
 
        St, Su, Sv, Sw = ntuple(_ -> Vector{Any}(undef, length(days)), 4)
        for (i, snum) in enumerate(hy_snapshot_nums)
            St[i], Su[i], Sv[i], Sw[i] = hyspectrum_uvwT(snapshots, snum, hy_klev, hy_xrange, hy_yrange, hy_dx, hy_dy)
        end
        return (; St, Su, Sv, Sw)
 
    elseif sim == "nhy"
        set_value!(; Δh = nhy_dx)
        St, Su, Sv, Sw = ntuple(_ -> Vector{Any}(undef, length(days)), 4)
        for (i, iter) in enumerate(nhy_iterations)
            St[i], Su[i], Sv[i], Sw[i] = nhyspectrum_uvwT(iter, days[i], subdomain, nhy_klev, nhy_dx, nhy_dy)
        end
        return (; St, Su, Sv, Sw)
 
    else
        error("compute_spectra_evol expects \"hy\" or \"nhy\" (call both separately for overlay plots).")
    end
end
 
# =============================================================================
# --- Cache the nhy normalization factors (avoids recomputing full nhy spectra
#     every time you just want to plot "hy") ---
# =============================================================================
using JLD2
 
norm_cache_file = filesave * "nhy_norm_reference_subdomain$(subdomain).jld2"
 
function get_nhy_norm_factors(; force_recompute=false)
    if !force_recompute && isfile(norm_cache_file)
        println("Loading cached nhy normalization factors from $norm_cache_file...")
        return load(norm_cache_file, "norm_factors")
    end
 
    println("Computing nhy normalization factors (will cache to $norm_cache_file)...")
    nhy_data = compute_spectra_evol("nhy")
    norm_factors = Dict(
        :St => nhy_data.St[end].spec[1],
        :Su => nhy_data.Su[end].spec[1],
        :Sv => nhy_data.Sv[end].spec[1],
        :Sw => nhy_data.Sw[end].spec[1],
    )
    # Also cache the freq axis for the k^(-5/3) reference line
    freq_ref = nhy_data.Sv[1].freq
 
    mkpath(filesave)
    jldsave(norm_cache_file; norm_factors, freq_ref)
    return norm_factors
end
 
function get_nhy_freq_ref()
    if isfile(norm_cache_file)
        return load(norm_cache_file, "freq_ref")
    end
    error("No cached freq_ref found — call get_nhy_norm_factors() first to populate the cache.")
end

# =============================================================================
# --- Plot one field's spectra evolution across days, for one or both sims ---
# field ∈ (:St, :Su, :Sv, :Sw); varname is used for the y-axis label / filename
# =============================================================================
function plot_spectra_evol(sim::String, field::Symbol, varname::String;
                            hy_data=nothing, nhy_data=nothing,
                            zlabel="z=-8.4375 m")

    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                    ylabel = L"E_{%$varname}(k)/E_{%$varname} (k_{min},\text{day}=0.5)",
                    xscale = log10, yscale = log10,
                    limits = ((10^-3.5, 10^0.5), (1e-10, 1e2)))

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title = zlabel, axis_kwargs1...)

    # Normalization reference is ALWAYS the nhy last-day spectrum, regardless
    # of which sim(s) are being plotted. If nhy_data wasn't computed this run
    # (e.g. sim == "hy"), fall back to the cached nhy normalization factor.
    if nhy_data !== nothing
        norm_spec1 = getfield(nhy_data, field)[end].spec[1]
        freq_ref = getfield(nhy_data, field)[1].freq
    else
        norm_factors = get_nhy_norm_factors()
        norm_spec1 = norm_factors[field]
        freq_ref = get_nhy_freq_ref()
    end
 
    # Draw reference k^(-5/3) line (freq axis taken from nhy day-0.5 spectrum,
    # which is always available regardless of sim)
    lines!(ax, freq_ref, 1e-8 .* freq_ref.^(-2), linestyle = :dash, color = :black)
    lines!(ax, freq_ref, 1e-8 .* freq_ref.^(-5/3), linestyle = :dash, color = :gray)
    
    if sim in ("hy", "both") && hy_data !== nothing
        S = getfield(hy_data, field)
        for (i, d) in enumerate(days)
            lines!(ax, S[i].freq, Real.(S[i].spec ./ norm_spec1),
                color = colors[i], linestyle = :dash,
                label = "hy " * day_labels[i])
        end
    end

    if sim in ("nhy", "both") && nhy_data !== nothing
        S = getfield(nhy_data, field)
        for (i, d) in enumerate(days)
            lines!(ax, S[i].freq, Real.(S[i].spec ./ norm_spec1),
                color = colors[i], linestyle = :solid,
                label = "nhy " * day_labels[i])
        end
    end

    xlims!(ax, (10^-3.5, 10^0.5))
    # 10 km scale boundary (submesoscale / mesoscale)
    vlines!(ax, [2π/10^4]; color = :red, linewidth = 0.5)
    axislegend(ax, labelsize = 10, patchsize = (20, 5), position = (:left, :bottom))

    fname = filesave * "spectra$(varname)_" * "subdomain$(subdomain)_" * sim * "_evol.png"
    save(fname, fig)
    println("Saved $fname")
end

# =============================================================================
# --- Top-level dispatcher ---
# =============================================================================
# function spectra_uvwT(sim::String)
#     hy_data, nhy_data = nothing, nothing

#     if sim == "hy"
#         println("------ Computing spectra from hydrostatic simulation ------")
#         hy_data = compute_spectra_evol("hy")
#     elseif sim == "nhy"
#         println("------ Computing spectra from nonhydrostatic simulation ------")
#         nhy_data = compute_spectra_evol("nhy")
#     elseif sim == "both"
#         println("------ Computing spectra from both simulations ------")
#         hy_data  = compute_spectra_evol("hy")
#         nhy_data = compute_spectra_evol("nhy")
#     else
#         error("Unknown sim type: $sim. Expected \"hy\", \"nhy\", or \"both\".")
#     end

#     for (field, varname) in [(:St, "T"), (:Su, "u"), (:Sv, "v"), (:Sw, "w")]
#         plot_spectra_evol(sim, field, varname; hy_data, nhy_data)
#     end
# end

function spectra_uvwT(sim::String)
    hy_data, nhy_data = nothing, nothing
 
    if sim == "hy"
        println("------ Computing spectra from hydrostatic simulation ------")
        hy_data = compute_spectra_evol("hy")
        # nhy_data left as `nothing` — plot_spectra_evol will use the cached
        # nhy normalization factor instead of recomputing the full nhy pipeline.
    elseif sim == "nhy"
        println("------ Computing spectra from nonhydrostatic simulation ------")
        nhy_data = compute_spectra_evol("nhy")
    elseif sim == "both"
        println("------ Computing spectra from both simulations ------")
        hy_data  = compute_spectra_evol("hy")
        nhy_data = compute_spectra_evol("nhy")
    else
        error("Unknown sim type: $sim. Expected \"hy\", \"nhy\", or \"both\".")
    end
 
    for (field, varname) in [(:St, "T"), (:Su, "u"), (:Sv, "v"), (:Sw, "w")]
        plot_spectra_evol(sim, field, varname; hy_data, nhy_data)
    end
end

# =============================================================================
# --- Run ---
# =============================================================================
# spectra_uvwT("both")   # or "hy" / "nhy" individually
spectra_uvwT("nhy")
spectra_uvwT("hy")
