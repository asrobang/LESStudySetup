using CairoMakie
using JLD2
set_theme!(Theme(fontsize = 12))

# =============================================================================
# Overlay nhy spectra evolution for regions A, B, C — one figure per variable.
# Reads only the on-disk caches written by visualize_spectra_hybrid.jl
# (nhy_spectra_evol_<region>_k<klev>.jld2 + the shared norm factor); no raw
# snapshots are touched. Run visualize_spectra_hybrid.jl first if a cache is
# missing or stale.
# =============================================================================

# =============================================================================
# --- File directories ---
# =============================================================================
filesave = "/home/asrobang/orcd/scratch/figures/20260927_regionABC_redo/"

# =============================================================================
# --- Settings (must match those used to build the caches) ---
# =============================================================================
regions        = ["regionA", "regionB", "regionC"]
region_colors  = Makie.wong_colors()[1:3]     # blue, orange, green (colorblind-safe)
region_labels  = ["Region A", "Region B", "Region C"]

nhy_klev = 70           # z = -2.8125 m in the nonhydrostatic subdomain grid
depth    = -2.8125      # [m] cell-center depth of nhy_klev (see nhy_depth in visualize_spectra_hybrid.jl)

days           = [5.5, 6.5, 7.5]              # must match the cache layout
nhy_iterations = [123463, 143293, 164410]

plot_day = 7.5                                # only this day is plotted
iday     = findfirst(==(plot_day), days)
day_tag  = replace(string(plot_day), "." => "-")   # "7-5", avoids a period in the filename

# --- Normalization reference (see get_norm_factor in visualize_spectra_hybrid.jl) ---
norm_day    = 7.5
norm_region = "regionC"


# =============================================================================
# --- Load cached spectra and norm factor ---
# =============================================================================
spectra_cache_file(region) = filesave * "nhy_spectra_evol_$(region)_k$(nhy_klev).jld2"
norm_cache_file = filesave * "nhy_norm_reference_u_$(norm_region)_k$(nhy_klev).jld2"

function load_region_spectra(region)
    file = spectra_cache_file(region)
    isfile(file) || error("Missing cache $file — run visualize_spectra_hybrid.jl with fileparam = \"$region\" first.")
    St, Su, Sv, Sw, iterations = load(file, "St", "Su", "Sv", "Sw", "iterations")
    iterations == nhy_iterations ||
        error("Cache $file is stale (iterations $(iterations) ≠ $(nhy_iterations)).")
    println("Loaded cached nhy spectra from $file")
    return (; St, Su, Sv, Sw)
end

function load_norm_factor()
    isfile(norm_cache_file) || error("Missing norm factor cache $norm_cache_file — run visualize_spectra_hybrid.jl first.")
    stored = load(norm_cache_file, "norm_region", "nhy_klev")
    stored == (norm_region, nhy_klev) ||
        error("Norm factor cache is for $(stored), expected $((norm_region, nhy_klev)).")
    return load(norm_cache_file, "norm_factor")
end

# =============================================================================
# --- Plot one field's spectrum on plot_day, all regions overlaid ---
# field ∈ (:St, :Su, :Sv, :Sw); varname is used for the y-axis label / filename
# color = region
# =============================================================================
function plot_regions_spectra(field::Symbol, varname::String, region_data, norm_spec1;
                              zlabel = "z=$(depth) m")

    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                    ylabel = L"E_{%$varname}(k)/E_{u} (k_{min},\text{day}=%$(norm_day),\text{%$(norm_region)})",
                    xscale = log10, yscale = log10,
                    limits = ((10^-4.5, 10^0.5), (1e-12, 1e6)))

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title = "nhy, Day $(plot_day), " * zlabel, axis_kwargs1...)

    # k^(-2) and k^(-5/3) reference slopes (tune amplitudes A23, A53 by hand)
    kref = 10 .^ range(-4.5, 0.5, length = 100)
    A23, A53 = 1e-9, 1e-5
    lines!(ax, kref, A23 .* kref .^ (-2); linestyle = :dash, color = :grey, label = L"k^{-2}")
    lines!(ax, kref, A53 .* kref .^ (-5/3); linestyle = :dash, color = :black, label = L"k^{-5/3}")

    for (r, data) in enumerate(region_data)
        S = getfield(data, field)[iday]
        lines!(ax, S.freq, Real.(S.spec ./ norm_spec1);
               color = region_colors[r], label = region_labels[r])
    end

    xlims!(ax, (10^-4.5, 10^0.5))
    
    # 10 km scale boundary (submesoscale / mesoscale)
    # vlines!(ax, [2π/10^4]; color = :red, linewidth = 0.5)
    vspan!(ax, 2π/10^4, 10^0.5; color = (:lightblue, 0.2))

    axislegend(ax, labelsize = 10, patchsize = (20, 5), position = (:left, :bottom))

    fname = filesave * "spectra$(varname)_regionABC_nhy_$(day_tag).png"
    save(fname, fig)
    println("Saved $fname")
end

function plot_unnorm_regions_spectra(field::Symbol, varname::String, region_data, norm_spec1;
                              zlabel = "z=$(depth) m")

    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                    ylabel = L"E_{%$varname}(k)/E_{u} (k_{min},\text{day}=%$(norm_day),\text{%$(norm_region)})",
                    xscale = log10, yscale = log10,
                    limits = ((10^-4.5, 10^0.5), (1e-12, 1e6)))

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title = "nhy, Day $(plot_day), " * zlabel, axis_kwargs1...)

    # k^(-2) and k^(-5/3) reference slopes (tune amplitudes A23, A53 by hand)
    kref = 10 .^ range(-4.5, 0.5, length = 100)
    A23, A53 = 1e-9, 1e-5
    lines!(ax, kref, A23 .* kref .^ (-2); linestyle = :dash, color = :grey, label = L"k^{-2}")
    lines!(ax, kref, A53 .* kref .^ (-5/3); linestyle = :dash, color = :black, label = L"k^{-5/3}")

    for (r, data) in enumerate(region_data)
        S = getfield(data, field)[iday]
        lines!(ax, S.freq, Real.(S.spec);
               color = region_colors[r], label = region_labels[r])
    end

    xlims!(ax, (10^-4.5, 10^0.5))
    
    # 10 km scale boundary (submesoscale / mesoscale)
    # vlines!(ax, [2π/10^4]; color = :red, linewidth = 0.5)
    vspan!(ax, 2π/10^4, 10^0.5; color = (:lightblue, 0.2))

    axislegend(ax, labelsize = 10, patchsize = (20, 5), position = (:left, :bottom))

    fname = filesave * "1norm_spectra$(varname)_regionABC_nhy_$(day_tag).png"
    save(fname, fig)
    println("Saved $fname")
end

# =============================================================================
# --- Run ---
# =============================================================================
region_data = [load_region_spectra(region) for region in regions]
norm_spec1  = load_norm_factor()
println("norm factor = $(norm_spec1)")

for (field, varname) in [(:St, "T"), (:Su, "u"), (:Sv, "v"), (:Sw, "w")]
    # plot_regions_spectra(field, varname, region_data, norm_spec1)
    plot_unnorm_regions_spectra(field, varname, region_data, norm_spec1)
end
