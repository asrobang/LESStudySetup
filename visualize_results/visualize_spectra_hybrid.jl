using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_snapshots, load_subdomain_snapshot, isotropic_powerspectrum, δ
using JLD2
set_theme!(Theme(fontsize = 12))

# =============================================================================
# --- File directories ---
# =============================================================================
hy_filehead  = "/orcd/data/abodner/002/shared_datasets/hyles_output/"
hy_filename  = hy_filehead * "hydrostatic_snapshots_hydrostatic_twin_simulation.jld2"
hy_metadata  = hy_filehead * "experiment_hydrostatic_twin_simulation_metadata.jld2"

nhy_filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/"

filesave = "/home/asrobang/orcd/scratch/figures/20261019_hy/"

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
hy_klev  = 222       # z = -2.8125 m in the hydrostatic full-domain grid
nhy_klev = 70        # z = -2.8125 m in the nonhydrostatic subdomain grid

# --- Slice of the hydrostatic full domain matching region A, B, or C ---
# (if left index > right index, the range wraps the periodic boundary)
function hy_region_ranges(fileparam)
    if fileparam == "regionA"
        xL, xR = 352, 608
        yB, yT = 192, 448
    elseif fileparam == "regionB"
        xL, xR = 512, 128
        yB, yT = 480, 96
    elseif fileparam == "regionC"
        xL, xR = 64, 320
        yB, yT = 192, 448
    else
        error("Subdomain not detected: fileparam = \"$fileparam\"; expected \"regionA\", \"regionB\", or \"regionC\".")
    end

    xrange = xL > xR ? vcat(xL:640, 1:xR) : (xL:xR)
    yrange = yB > yT ? vcat(yB:640, 1:yT) : (yB:yT)
    return xrange, yrange
end

# ! ! ! ! ! EDIT THIS ONE TO SWITCH REGION ! ! ! ! !
fileparam = "regionB"

# --- Day <-> snapshot_number (hy) / iteration (nhy) correspondence ---
# days              = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
days              = [5.5, 6.5, 7.5]
# hy_snapshot_nums  = [9, 25, 41, 57, 73, 89, 105, 121]
hy_snapshot_nums  = [89, 105, 121]
# nhy_iterations    = [22865, 42238, 62484, 82586, 103348, 123463, 143293, 164410]
nhy_iterations    = [123463, 143293, 164410]

# --- Normalization reference: nhy u spectrum, k_min bin, on norm_day in norm_region.
#     Fixed across runs, so every variable and every region shares one factor. ---
norm_day       = 7.5
norm_iteration = 164410      # nhy iteration for day 7.5 (see full day<->iteration table above)
norm_region    = "regionC"   # reference region, independent of fileparam

# --- Depth of nhy_klev, read from the nhy output grid (cf. grid.jl) ---
function nhy_depth(klev; iteration = nhy_iterations[1])
    fname = contains(fileparam, "region") ?
        nhy_filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2" :
        nhy_filehead * fileparam * "_iter$(iteration).jld2"
    grid = jldopen(file -> file["grid"], fname, "r")      # grid only, no field data
    return znodes(grid, Center())[klev]             # cell-center depth; matches T/u/v and the w average
end
depth = nhy_depth(nhy_klev)
println("nhy_klev = $(nhy_klev) → z = $(depth) m")

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
# --- Nonhydrostatic spectrum of ONE variable from a region's per-variable file
#     (subdomain_<var>_<region>_iter<iteration>.jld2) ---
# =============================================================================
function nhy_region_spectrum(var::Symbol, iteration, region, klev, dx, dy)
    output_filename = nhy_filehead * "subdomain_$(var)_" * region * "_iter$(iteration).jld2"
    snapshot = load_subdomain_snapshot(output_filename)
    field = snapshot[var]
    if var == :w
        # w at cell faces in z: interpolate between neighboring faces to get depth of cell center
        slice = (interior(field, :, :, klev) + interior(field, :, :, klev+1)) / 2
    else
        slice = interior(field, :, :, klev)
    end
    S = isotropic_powerspectrum(slice, slice; Δx=dx, Δy=dy)
    println("Freeing large variables...")
    field = nothing
    snapshot = nothing      # free the variable
    GC.gc()                 # force the garbage collector to run immediately
    return S
end

# =============================================================================
# --- Nonhydrostatic spectra (subdomain tile snapshots) ---
# =============================================================================
function nhyspectrum_uvwT(iteration, nday, fileparam, klev, dx, dy)
    println("Reading nhy iteration $(iteration) on day $(nday), $(fileparam)...")
    t0 = now()

    # Distinguish between regions and 10kmx10km tiles
    if contains(fileparam, "region")
        St, Su, Sv, Sw = (nhy_region_spectrum(var, iteration, fileparam, klev, dx, dy) for var in (:T, :u, :v, :w))
    else
        output_filename = nhy_filehead * fileparam * "_iter$(iteration).jld2"

        # 3. Load the snapshot
        snapshot = load_subdomain_snapshot(output_filename)

        T = snapshot[:T]
        u = snapshot[:u]
        v = snapshot[:v]
        w = snapshot[:w]

        println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")

        # Coordinate arrays
        xu, yu, zu = nodes(u)       # u at cell faces in x
        xv, yv, zv = nodes(v)       # v at cell faces in y
        xw, yw, zw = nodes(w)       # w at cell faces in z
        xT, yT, zT = nodes(T)       # T at cell centers 

        # Compute the auto-spectrum (co-spectrum of field with itself) of T, u, v, w 
        Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev); Δx=dx, Δy=dy)
        Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev); Δx=dx, Δy=dy)
        wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2      # interpolates between neighboring cells to get depth of cell center
        Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
        St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)
    end

    return St, Su, Sv, Sw
end

# =============================================================================
# --- On-disk cache of the full St/Su/Sv/Sw spectra vectors (all days), so a
#     fresh Julia session can reload previously-computed spectra evolution
#     instead of recomputing it from raw simulation snapshots. ---
# =============================================================================
spectra_cache_file(sim::String) = filesave * "$(sim)_spectra_evol_$(fileparam)_k$(sim == "hy" ? hy_klev : nhy_klev).jld2"

function save_spectra_cache(sim::String, data::NamedTuple)
    mkpath(filesave)
    file = spectra_cache_file(sim)
    to_plain(S) = [(spec = s.spec, freq = s.freq) for s in S]
    St, Su, Sv, Sw = to_plain(data.St), to_plain(data.Su), to_plain(data.Sv), to_plain(data.Sw)
    println("Caching $sim spectra evolution to $file...")
    if sim == "nhy"
        jldsave(file; St, Su, Sv, Sw, iterations = nhy_iterations)
    else
        jldsave(file; St, Su, Sv, Sw)
    end
end

# Cached nhy spectra are only reused if they were computed for the current nhy_iterations
function spectra_cache_is_current(sim::String)
    sim == "nhy" || return true          # hy behavior unchanged
    stored = jldopen(file -> haskey(file, "iterations") ? file["iterations"] : nothing,
                     spectra_cache_file(sim), "r")
    stored == nhy_iterations && return true
    println("Cached $sim spectra are stale (iterations $(stored) ≠ $(nhy_iterations)), recomputing...")
    return false
end

function load_spectra_cache(sim::String)
    file = spectra_cache_file(sim)
    println("Loading cached $sim spectra evolution from $file...")
    St, Su, Sv, Sw = load(file, "St", "Su", "Sv", "Sw")
    return (; St, Su, Sv, Sw)
end

function compute_spectra_evol(sim::String; force_recompute=false)
    sim == "hy" && return _compute_spectra_evol_uncached("hy")     # hy spectra are never cached

    if !force_recompute && isfile(spectra_cache_file(sim)) && spectra_cache_is_current(sim)
        return load_spectra_cache(sim)
    end

    result = _compute_spectra_evol_uncached(sim)
    save_spectra_cache(sim, result)
    return result
end
 
function _compute_spectra_evol_uncached(sim::String)
    if sim == "hy"
        hy_xrange, hy_yrange = hy_region_ranges(fileparam)    # errors before loading data if no region is selected
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
            St[i], Su[i], Sv[i], Sw[i] = nhyspectrum_uvwT(iter, days[i], fileparam, nhy_klev, nhy_dx, nhy_dy)
        end
        return (; St, Su, Sv, Sw)
 
    else
        error("compute_spectra_evol expects \"hy\" or \"nhy\" (call both separately for overlay plots).")
    end
end
 
# =============================================================================
# --- Single normalization factor shared by T, u, v, w and by every region:
#     k_min bin of the nhy u spectrum on norm_day in norm_region. Cached on disk
#     under norm_region (not fileparam) so all region runs reuse the same value. ---
# =============================================================================
norm_cache_file = filesave * "nhy_norm_reference_u_$(norm_region)_k$(nhy_klev).jld2"

function get_norm_factor(; force_recompute=false)
    if !force_recompute && isfile(norm_cache_file)
        stored = jldopen(norm_cache_file, "r") do file
            all(haskey(file, key) for key in ("norm_region", "norm_iteration", "nhy_klev")) ?
                (file["norm_region"], file["norm_iteration"], file["nhy_klev"]) : nothing
        end
        if stored == (norm_region, norm_iteration, nhy_klev)
            norm_factor = load(norm_cache_file, "norm_factor")
            println("Loaded cached norm factor = $(norm_factor) from $norm_cache_file")
            return norm_factor
        end
        println("Cached norm factor is stale ($(stored) ≠ $((norm_region, norm_iteration, nhy_klev))), recomputing...")
    end

    println("Computing norm factor from nhy u, $(norm_region), iteration $(norm_iteration) (will cache to $norm_cache_file)...")
    set_value!(; Δh = nhy_dx)
    norm_factor = nhy_region_spectrum(:u, norm_iteration, norm_region, nhy_klev, nhy_dx, nhy_dy).spec[1]
    println("norm factor = $(norm_factor)")

    mkpath(filesave)
    jldsave(norm_cache_file; norm_factor, norm_region, norm_iteration, nhy_klev)
    return norm_factor
end

# =============================================================================
# --- Plot one field's spectra evolution across days, for one or both sims ---
# field ∈ (:St, :Su, :Sv, :Sw); varname is used for the y-axis label / filename
# =============================================================================
function plot_spectra_evol(sim::String, field::Symbol, varname::String;
                            hy_data=nothing, nhy_data=nothing,
                            zlabel="z=$(depth) m")

    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                    ylabel = L"E_{%$varname}(k)/E_{u} (k_{min},\text{day}=%$(norm_day),\text{%$(norm_region)})",
                    xscale = log10, yscale = log10,
                    limits = ((10^-4, 10^0), (1e-8, 1e2)))

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title = zlabel, axis_kwargs1...)

    # Shaded wavenumber bands (drawn first so they sit behind the lines)
    k_10km  = 2π / 10000     # λ = 10 km
    k_300m  = 2π / 300       # λ = 300 m
    k_right = 10^0.5         # right edge of the x-limits
    vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))   # light blue: k > 2π/10000
    vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))  # darker: k > 2π/300 (layers on top of the light band)

    # Same reference for every field, sim, and region (see get_norm_factor)
    norm_spec1 = get_norm_factor()

    # k^(-2) and k^(-5/3) reference slopes (tune amplitudes A23, A13 by hand)
    kref = 10 .^ range(-4, 0, length = 100)
    A23, A53 = 1e-9, 1e-4
    lines!(ax, kref, A23 .* kref .^ (-2); linestyle = :dot, color = :grey, label = L"k^{-2}")
    lines!(ax, kref, A53 .* kref .^ (-5/3); linestyle = :dot, color = :black, label = L"k^{-5/3}")
    
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

    xlims!(ax, (10^-4, 10^0))
    # # 10 km scale boundary (submesoscale / mesoscale)
    # vlines!(ax, [2π/10^4]; color = :red, linewidth = 0.5)
    axislegend(ax, labelsize = 10, patchsize = (20, 5), position = (:right, :top))

    fname = filesave * "spectra$(varname)_" * "$(fileparam)_" * sim * "_evol.png"
    save(fname, fig)
    println("Saved $fname")
end

# =============================================================================
# --- Top-level dispatcher ---
# =============================================================================

function spectra_uvwT(sim::String)
    hy_data, nhy_data = nothing, nothing
 
    if sim == "hy"
        println("------ Computing spectra from hydrostatic simulation ------")
        hy_data = compute_spectra_evol("hy")
        # nhy_data left as `nothing` — plot_spectra_evol only needs the cached
        # norm factor, not the full nhy pipeline.
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
# --- Last-snapshot comparison across regions: hy (dashed) vs nhy (solid) ---
# =============================================================================
regions = ["regionA", "regionB", "regionC"]

# Reuse a region's cached nhy spectra (written by spectra_uvwT for that region) if they
# contain nhy_iter; otherwise compute them from the raw subdomain files
function nhy_region_spectra_lastday(region, nhy_iter, nday)
    cache_file = filesave * "nhy_spectra_evol_$(region)_k$(nhy_klev).jld2"
    if isfile(cache_file)
        stored = jldopen(jld -> haskey(jld, "iterations") ? jld["iterations"] : nothing, cache_file, "r")
        idx = stored === nothing ? nothing : findfirst(==(nhy_iter), stored)
        if idx !== nothing
            println("Loading cached nhy $region spectra (iteration $nhy_iter) from $cache_file...")
            St, Su, Sv, Sw = load(cache_file, "St", "Su", "Sv", "Sw")
            return (; St = St[idx], Su = Su[idx], Sv = Sv[idx], Sw = Sw[idx])
        end
        println("Cached nhy $region spectra don't include iteration $nhy_iter, recomputing...")
    end
    St, Su, Sv, Sw = nhyspectrum_uvwT(nhy_iter, nday, region, nhy_klev, nhy_dx, nhy_dy)
    return (; St, Su, Sv, Sw)
end

function compute_region_spectra_lastday()
    hy_snum, nhy_iter, nday = hy_snapshot_nums[end], nhy_iterations[end], days[end]

    set_value!(; Δh = hy_dx)
    println("Loading hydrostatic data from $hy_filename...")
    snapshots = load_snapshots(hy_filename; metadata=hy_metadata)
    hy_region_data = map(regions) do region
        xrange, yrange = hy_region_ranges(region)
        St, Su, Sv, Sw = hyspectrum_uvwT(snapshots, hy_snum, hy_klev, xrange, yrange, hy_dx, hy_dy)
        (; St, Su, Sv, Sw)
    end

    set_value!(; Δh = nhy_dx)
    nhy_region_data = map(region -> nhy_region_spectra_lastday(region, nhy_iter, nday), regions)
    return hy_region_data, nhy_region_data
end

function plot_spectra_regions(field::Symbol, varname::String, hy_region_data, nhy_region_data;
                              zlabel="z=$(depth) m, Day $(days[end])")

    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                    ylabel = L"E_{%$varname}(k)/E_{u} (k_{min},\text{day}=%$(norm_day),\text{%$(norm_region)})",
                    xscale = log10, yscale = log10,
                    limits = ((10^-4, 10^0), (1e-8, 1e2)))

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title = zlabel, axis_kwargs1...)

    # Shaded wavenumber bands (drawn first so they sit behind the lines)
    k_10km  = 2π / 10000     # λ = 10 km
    k_300m  = 2π / 300       # λ = 300 m
    k_right = 10^0.5         # right edge of the x-limits
    vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))
    vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))

    # Same reference for every field, sim, and region (see get_norm_factor)
    norm_spec1 = get_norm_factor()

    # k^(-2) and k^(-5/3) reference slopes
    kref = 10 .^ range(-4, 0, length = 100)
    A23, A53 = 1e-9, 1e-4
    lines!(ax, kref, A23 .* kref .^ (-2); linestyle = :dot, color = :grey, label = L"k^{-2}")
    lines!(ax, kref, A53 .* kref .^ (-5/3); linestyle = :dot, color = :black, label = L"k^{-5/3}")

    colors = Makie.wong_colors()[1:length(regions)]
    for (i, region) in enumerate(regions)
        S_hy  = getfield(hy_region_data[i], field)
        S_nhy = getfield(nhy_region_data[i], field)
        lines!(ax, S_hy.freq, Real.(S_hy.spec ./ norm_spec1),
            color = colors[i], linestyle = :dash, label = "hy " * region)
        lines!(ax, S_nhy.freq, Real.(S_nhy.spec ./ norm_spec1),
            color = colors[i], linestyle = :solid, label = "nhy " * region)
    end

    xlims!(ax, (10^-4, 10^0))
    axislegend(ax, labelsize = 10, patchsize = (20, 5), position = (:right, :top))

    fname = filesave * "spectra$(varname)_regionsABC_both_day$(replace(string(days[end]), "." => "-")).png"
    save(fname, fig)
    println("Saved $fname")
end

function spectra_regions_lastday()
    println("------ Computing day $(days[end]) spectra for $(join(regions, ", ")) ------")
    hy_region_data, nhy_region_data = compute_region_spectra_lastday()
    for (field, varname) in [(:St, "T"), (:Su, "u"), (:Sv, "v"), (:Sw, "w")]
        plot_spectra_regions(field, varname, hy_region_data, nhy_region_data)
    end
end

# =============================================================================
# --- Run ---
# =============================================================================
# spectra_uvwT("both")   # or "hy" / "nhy" individually
spectra_regions_lastday()   # day 7.5, hy vs nhy for regions A, B, C
# spectra_uvwT("nhy")
# spectra_uvwT("hy")
