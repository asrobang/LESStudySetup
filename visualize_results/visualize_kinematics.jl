using LESStudySetup
using CairoMakie
using Printf
using LaTeXStrings
using FFTW
using Oceananigans.Grids: xnodes, ynodes, znodes, xspacings, yspacings
using Oceananigans.Fields: interior
using LESStudySetup.Diagnostics: load_subdomain_snapshot, isotropic_powerspectrum
using MathTeXEngine
using JLD2 #, CUDA
set_theme!(theme_latexfonts(), fontsize=12, figure_padding = 10)

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "/home/asrobang/orcd/scratch/figures/20261014_regionABC_kinematics/"

# --- Set parameters ---
set_value!(; Δh = 4.8828125)    # horizontal spacing
f = parameters.f;               # Coriolis parameter
Q = 40                          # surface sensible heat flux? 
h₀ = 60                         # height of the convective BL or mixing depth? 
ρ₀ = parameters.ρ₀              # reference density
cₚ = parameters.cp              # heat capacity
α = parameters.α                # thermal expansion
g = parameters.g                # gravity
wₛ = (α * g * Q * h₀ / (ρ₀ * cₚ))^(1/3)     # convective velocity scale [m/s]?
dx = 4.88281                    # [m] horizontal grid size
dy = 4.88281                    # [m] horizontal grid size
dz = 1.125                      # [m] vertical grid size

# --- Helper Functions ---

function compute_dAdx_dAdy(A,dx,dy) 
    A_hpad = hcat(A, A[:, 1:1])      # pad the first column at the rightmost (Ny,Nx+1)
    A_vpad = vcat(A, A[1:1, :])      # pad the first row at the bottom (Ny+1,Nx)
    dAdx = (A_hpad[:, 2:end] .- A_hpad[:, 1:end-1]) ./ dx
    dAdy = (A_vpad[2:end, :] .- A_vpad[1:end-1, :]) ./ dy
    
    return dAdx, dAdy
end

function compute_dAdz(A_above,A_below,dz)
    dAdz = (A_above - A_below) ./ (2*dz)

    return dAdz
end

function lowpass_filter(field, k, smooth)
    """
        lowpass_filter(field, k, smooth)

    Apply a sharp spectral cutoff filter to `field`, removing all horizontal
    scales smaller than `smooth` (a physical length, e.g. in meters — NOT a
    number of grid cells).

    Cutoff wavenumber:
        K = 2π / smooth

    This generalizes the "4Δx smoothing → K = π/(2Δx)" recipe: if smooth = 4Δx,
    then K = π/(2·(smooth/4)) = 2π/smooth, so K = 2π/smooth works for any
    smoothing scale.

    Assumes a horizontally uniform grid. Works on 2D (Nx, Ny) slices or
    3D (Nx, Ny, Nz) fields (filtered level-by-level in the horizontal).
    """

    grid = field.grid
    Δx = minimum(xspacings(grid, Center()))
    Δy = minimum(yspacings(grid, Center()))

    data = Array(interior(field,:,:,k))  # bring to CPU physical-space array
    Nx, Ny = size(data, 1), size(data, 2)
    Lx, Ly = Nx * Δx, Ny * Δy

    # # ONLY if the domain being transformed is periodic 
    # # Angular wavenumbers associated with the horizontal FFT
    # kx = 2π .* fftfreq(Nx, 1/dx)
    # ky = 2π .* fftfreq(Ny, 1/dy)

    # DCT used for zero-gradient domain edges 
    # Cosine-transform wavenumbers (Neumann / zero-gradient assumption):
    # note π, not 2π — the fundamental mode has period 2L, not L
    kx = π .* (0:Nx-1) ./ Lx
    ky = π .* (0:Ny-1) ./ Ly

    K = 2π / smooth          # cutoff wavenumber for this smoothing scale
    κ² = (kx .^ 2) .+ (ky' .^ 2)
    mask = κ² .<= K^2        # true where κ ≤ K (keep), false where κ > K (zero)

    filtered = similar(data)

    # # ONLY if the domain being transformed is periodic 
    # f̂ = fft(data)
    # f̂ .*= mask
    # filtered .= real.(ifft(f̂))

    # DCT used for zero-gradient domain edges 
    f̂ = FFTW.r2r(data, FFTW.REDFT10)   # forward DCT-II
    f̂ .*= mask
    filtered = FFTW.r2r(f̂, FFTW.REDFT01) ./ (4 * Nx * Ny)  # inverse DCT-III + normalization

    return filtered
end

function gaussian_filter(field, k, smooth; periodic_x=true, periodic_y=true)
    """
        gaussian_filter(field, k, smooth; periodic_x=true, periodic_y=true)

    Gaussian low-pass filter (width Δ = `smooth`, physical length) applied in the
    horizontal to level `k` of `field`. Transfer function exp(-κ²Δ²/24).

    Non-periodic directions are mirror-padded (half-sample symmetric reflection,
    appropriate for cell-centered data) before the FFT. This is equivalent to a
    zero-gradient condition at those edges and avoids wraparound contamination.
    """
    grid = field.grid
    Δx = minimum(xspacings(grid, Center()))
    Δy = minimum(yspacings(grid, Center()))

    data = Array(interior(field, :, :, k))
    Nx, Ny = size(data)

    # Mirror-pad non-periodic directions
    padded = periodic_x ? data : vcat(data, reverse(data, dims=1))
    padded = periodic_y ? padded : hcat(padded, reverse(padded, dims=2))
    Mx, My = size(padded)

    # Wavenumbers for the padded array
    kx = 2π .* rfftfreq(Mx, 1/Δx)
    ky = 2π .* fftfreq(My, 1/Δy)
    κ² = (kx .^ 2) .+ (ky' .^ 2)
    Ĝ = @. exp(-κ² * smooth^2 / 24)

    f̂ = rfft(padded)
    f̂ .*= Ĝ
    filtered = irfft(f̂, Mx)

    return filtered[1:Nx, 1:Ny]   # drop the mirrored copy
end

function compute_load_gradients(grad_cache_file, filehead, fileparam, iteration, klev, dx, dy, filter_scale)
    if isfile(grad_cache_file)
        println("Loading cached gradient variables from $grad_cache_file...")
        x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f =
            load(grad_cache_file,   "x", "y", "dTdx", "dTdx_f", "dTdy", "dTdy_f", 
                                    "dudx", "dudx_f", "dudy", "dudy_f", "dvdx", "dvdx_f", "dvdy", "dvdy_f")
    else
        # --- Compute gradients for T ---

        output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)
        x, y, _ = nodes(snapshot[:T])                           # call this one time to get x and y coords

        T_uf = copy(interior(snapshot[:T],:,:,klev))              # 2048×2048 Matrix{Float32}
        dTdx, dTdy = compute_dAdx_dAdy(T_uf,dx,dy) 
        println("Computed unfiltered T gradients")
        T_uf = nothing                                          # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # T_f = lowpass_filter(snapshot[:T], klev, filter_scale)            # 2048×2048 Matrix{Float32}
        T_f = gaussian_filter(snapshot[:T], klev, filter_scale)            # 2048×2048 Matrix{Float32}
        dTdx_f, dTdy_f = compute_dAdx_dAdy(T_f,dx,dy) 
        println("Computed filtered T gradients")
        T_f = nothing                                           # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately
        
        # --- Compute gradients for u ---

        output_filename = filehead * "subdomain_u_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)

        u_uf = copy(interior(snapshot[:u],:,:,klev)[1:end-1, :])        # initially 2049×2048 Matrix{Float32}\
        dudx, dudy = compute_dAdx_dAdy(u_uf,dx,dy) 
        println("Computed unfiltered u gradients")
        u_uf = nothing                                          # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # u_f = lowpass_filter(snapshot[:u], klev, filter_scale)[1:end-1, :]  # initially 2049×2048 Matrix{Float32}
        u_f = gaussian_filter(snapshot[:u], klev, filter_scale)[1:end-1, :]  # initially 2049×2048 Matrix{Float32}
        dudx_f, dudy_f = compute_dAdx_dAdy(u_f,dx,dy) 
        println("Computed filtered u gradients")
        u_f = nothing                                           # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Compute gradients for v ---

        output_filename = filehead * "subdomain_v_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)

        v_uf = copy(interior(snapshot[:v],:,:,klev)[:, 1:end-1])        # initially 2048×2049 Matrix{Float32}
        dvdx, dvdy = compute_dAdx_dAdy(v_uf,dx,dy) 
        println("Computed unfiltered v gradients")
        v_uf = nothing                                          # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # v_f = lowpass_filter(snapshot[:v], klev, filter_scale)[:, 1:end-1]  # initially 2048×2049 Matrix{Float32}
        v_f = gaussian_filter(snapshot[:v], klev, filter_scale)[:, 1:end-1]  # initially 2048×2049 Matrix{Float32}
        dvdx_f, dvdy_f = compute_dAdx_dAdy(v_f,dx,dy)
        println("Computed filtered v gradients")
        v_f = nothing                                           # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Save computed gradient variables to cache so future runs can skip recomputation ---
        mkpath(filesave)
        jldsave(grad_cache_file; x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        println("Saved cached gradient variables to $grad_cache_file")
    end # if isfile(grad_cache_file)

    return x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f
end

function compute_frontogenesis(α, g, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
    dbdx = (α*g) .* dTdx
    dbdx_f = (α*g) .* dTdx_f
    dbdy = (α*g) .* dTdy
    dbdy_f = (α*g) .* dTdy_f
    
    # Compute unfiltered frontogenetic tendency (F_uf)
    F_uf = - ((dbdx.^2 .* dudx) .+ (dbdy.^2 .* dvdy)) .- (dbdx .* dbdy .* (dudy .+ dvdx))
    # Compute filtered frontogenetic tendency (F_f)
    F_f = - ((dbdx_f.^2 .* dudx_f) .+ (dbdy_f.^2 .* dvdy_f)) .- (dbdx_f .* dbdy_f .* (dudy_f .+ dvdx_f))

    return F_uf, F_f
end

function compute_vorticity(dudy, dudy_f, dvdx, dvdx_f)
    # Compute unfiltered vorticity (zeta_uf)
    zeta_uf = dvdx .- dudy
    # Compute filtered vorticity (zeta_f)
    zeta_f = dvdx_f .- dudy_f

    return zeta_uf, zeta_f
end

function compute_divergence(dudx, dudx_f, dvdy, dvdy_f)
    # Compute unfiltered divergence (delta_uf)
    delta_uf = dudx .+ dvdy
    # Compute filtered divergence (delta_f)
    delta_f = dudx_f .+ dvdy_f

    return delta_uf, delta_f
end

function compute_strain(dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
    # Compute unfiltered strain
    sig_n_uf = dudx .- dvdy
    sig_s_uf = dvdx .+ dudy
    sig_uf = sqrt.(sig_n_uf.^2 .+ sig_s_uf.^2)

    # Compute filtered strain
    sig_n_f = dudx_f .- dvdy_f
    sig_s_f = dvdx_f .+ dudy_f
    sig_f = sqrt.(sig_n_f.^2 .+ sig_s_f.^2)

    return sig_n_uf, sig_s_uf, sig_uf, sig_n_f, sig_s_f, sig_f
end

function compute_fluxes()
    if isfile(flux_cache_file)
        println("Loading cached flux variables from $flux_cache_file...")
        # x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f =
        #     load(grad_cache_file,   "x", "y", "dTdx", "dTdx_f", "dTdy", "dTdy_f", 
        #                             "dudx", "dudx_f", "dudy", "dudy_f", "dvdx", "dvdx_f", "dvdy", "dvdy_f")
    else
        # --- Obtain unfiltered and filtered copies of T, u, v, w ---

        # T
        output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)
        x, y, _ = nodes(snapshot[:T])                               # call this one time to get x and y coords
        T_uf = copy(interior(snapshot[:T],:,:,klev))                # 2048×2048 Matrix{Float32}
        T_f = gaussian_filter(snapshot[:T], klev, filter_scale)     # 2048×2048 Matrix{Float32}
        snapshot = nothing
        GC.gc()                                                     # force the garbage collector to run immediately

        # u
        output_filename = filehead * "subdomain_u_" * fileparam * "_iter$(iteration).jld2"      # Load u field
        snapshot = load_subdomain_snapshot(output_filename)
        u_uf = copy(interior(snapshot[:u],:,:,klev)[1:end-1, :])                # initially 2049×2048 Matrix{Float32}
        u_f = gaussian_filter(snapshot[:u], klev, filter_scale)[1:end-1, :]     # initially 2049×2048 Matrix{Float32}
        snapshot = nothing
        GC.gc()                                                     # force the garbage collector to run immediately

        # v
        output_filename = filehead * "subdomain_v_" * fileparam * "_iter$(iteration).jld2"      # Load v field
        snapshot = load_subdomain_snapshot(output_filename)
        v_uf = copy(interior(snapshot[:v],:,:,klev)[:, 1:end-1])                # initially 2048×2049 Matrix{Float32}
        v_f = gaussian_filter(snapshot[:v], klev, filter_scale)[:, 1:end-1]     # initially 2048×2049 Matrix{Float32}
        snapshot = nothing
        GC.gc()                                                     # force the garbage collector to run immediately

        # w
        output_filename = filehead * "subdomain_w_" * fileparam * "_iter$(iteration).jld2"      # Load w field
        snapshot = load_subdomain_snapshot(output_filename)
        w_uf = copy(interior(snapshot[:w],:,:,70))                  # 2048×2048 Matrix{Float32}
        w_f = gaussian_filter(snapshot[:w], 70, filter_scale)       # 2048×2048 Matrix{Float32}
        snapshot = nothing
        GC.gc()                                                     # force the garbage collector to run immediately

        # --- Compute fluxes ---

        # --- Save computed gradient variables to cache so future runs can skip recomputation ---
        mkpath(filesave)
        jldsave(grad_cache_file; x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        println("Saved cached gradient variables to $grad_cache_file")
    end # if isfile(flux_cache_file)

    # return x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f
end

### -------------------------------------------------------------------------
### Plotting functions

## Plots the heatmap for a region for a given indicated variable
## available variables: front, vort, div, strain_n, strain_s, strain
function plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, var)

    # Load gradients cache file if it exists, or compute gradients
    x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f = compute_load_gradients(
        grad_cache_file, filehead, fileparam, iteration, klev, dx, dy, filter_scale)

    # Set plotting variables
    if (var == "front")
        colormap = :balance
        colorscale = Makie.Symlog10(-5e-16, 5e-16)
        field, field_f = compute_frontogenesis(α, g, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        cmin, cmax = -5e-16, 5e-16
    elseif (var == "vort")
        colormap = :curl
        colorscale = identity
        field, field_f = compute_vorticity(dudy, dudy_f, dvdx, dvdx_f)
        cmin, cmax = -0.001, 0.001
    elseif (var == "div")
        colormap = :delta
        colorscale = identity
        field, field_f = compute_divergence(dudx, dudx_f, dvdy, dvdy_f)
        cmin, cmax = -0.001, 0.001
    elseif (var == "strain_n")
        colormap = :balance
        colorscale = identity
        # sig_n_uf, sig_s_uf, sig_uf, sig_n_f, sig_s_f, sig_f
        field, _, _, field_f, _, _ = compute_strain(dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        cmin, cmax = -0.001, 0.001
    elseif (var == "strain_s")
        colormap = :balance
        colorscale = identity
        _, field, _, _, field_f, _ = compute_strain(dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        cmin, cmax = -0.001, 0.001
    elseif (var == "strain")
        colormap = :amp
        colorscale = log10
        # sig_n_uf, sig_s_uf, sig_uf, sig_n_f, sig_s_f, sig_f
        _, _, field, _, _, field_f = compute_strain(dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        cmin, cmax = 1e-5, 1e-2
    else
        colormap = :dense
    end

    # Clean up memory
    dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f = ntuple(i -> nothing, 12)
    GC.gc()                                                 # force the garbage collector to run immediately

    # println(extrema(field))

    # Plot unfiltered field
    fig = Figure(size = (700, 640))
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    hm = heatmap!(ax, 1e-3x, 1e-3y, field;
                rasterize = true, colormap = colormap,
                colorscale = colorscale,
                colorrange = (cmin, cmax))
                # )
    Colorbar(fig[1, 2], hm)
    save(filesave * var * "_" * fileparam * "_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting unfiltered " * var * " heatmap")

    # Plot filtered field
    fig = Figure(size = (700, 640))
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    hm = heatmap!(ax, 1e-3x, 1e-3y, field_f;
                rasterize = true, colormap = colormap,
                colorscale = colorscale, 
                colorrange = (cmin, cmax))
                # )
    Colorbar(fig[1, 2], hm)
    save(filesave * var * "_f_" * fileparam * "_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting filtered " * var * " heatmap")
end

function plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, var; regions=["A", "B", "C"])
    
    S_uf = Dict{String, Any}()
    S_f  = Dict{String, Any}()

    for region in regions
        # Cache of computed gradient variables (dTdx, dTdy, dudx, dudy, dvdx, dvdy, filtered & unfiltered)
        fileparam = "region" * string(region)
        grad_cache_file = filesave * fileparam * "_iter$(iteration)_grad.jld2"

        # Load gradients cache file if it exists, or compute gradients
        x, y, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f = compute_load_gradients(
            grad_cache_file, filehead, fileparam, iteration, klev, dx, dy, filter_scale)

        # Compute unfiltered and filtered field
        if var == "front"
            field, field_f = compute_frontogenesis(α, g, dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
        elseif var == "vort"
            field, field_f = compute_vorticity(dudy, dudy_f, dvdx, dvdx_f)
        elseif var == "div"
            field, field_f = compute_divergence(dudx, dudx_f, dvdy, dvdy_f)
        elseif var in ("strain_n", "strain_s", "strain")
            # returns: sig_n_uf, sig_s_uf, sig_uf, sig_n_f, sig_s_f, sig_f
            out = compute_strain(dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f)
            k = var == "strain_n" ? 1 : var == "strain_s" ? 2 : 3
            field, field_f = out[k], out[k + 3]
        else
            error("Unknown var: $var")
        end

        # Clean up gradient memory
        dTdx, dTdx_f, dTdy, dTdy_f, dudx, dudx_f, dudy, dudy_f, dvdx, dvdx_f, dvdy, dvdy_f = ntuple(i -> nothing, 12)
        GC.gc()

        # Compute spectra and store by region
        S_uf[region] = isotropic_powerspectrum(field, field;     Δx=dx, Δy=dy)
        S_f[region]  = isotropic_powerspectrum(field_f, field_f; Δx=dx, Δy=dy)

        # Free the fields before the next region
        field, field_f = nothing, nothing
        GC.gc()
    end

    # Plot
    colors = Makie.wong_colors()[1:length(regions)]   # one color per region
    S0 = S_f["C"].spec[1]                              # normalize by filtered field of region C at k_min

    # LaTeX labels for each variable: (title, symbol used in the y-label)
    var_labels = Dict(
        "front"    => (L"F",          "F"),
        "vort"     => (L"\zeta",      "\\zeta"),
        "div"      => (L"\delta",     "\\delta"),
        "strain_n" => (L"\sigma_n",   "\\sigma_n"),
        "strain_s" => (L"\sigma_s",   "\\sigma_s"),
        "strain"   => (L"\sigma",     "\\sigma"),
    )
    ttl, sym = var_labels[var]

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title = ttl, xlabel = "Wavenumber (rad⋅m⁻¹)",
            ylabel = latexstring("E(k)/E_{$(sym),f}^{C} (k_{min})"),
            xscale = log10, yscale = log10)

    # Shaded wavenumber bands (drawn first so they sit behind the lines)
    k_10km  = 2π / 10000     # λ = 10 km
    k_300m  = 2π / 300       # λ = 300 m
    k_right = 10^0.5         # right edge of the x-limits
    vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))   # light blue: k > 2π/10000
    vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))  # darker: k > 2π/300 (layers on top of the light band)

    for (n, region) in enumerate(regions)
        Suf, Sfilt = S_uf[region], S_f[region]
        lines!(ax, collect(Suf.freq), Real.(Suf.spec ./ S0); color = colors[n], linewidth = 2,
            linestyle = :dash, label = "Region $(region), unfiltered")
        lines!(ax, collect(Sfilt.freq), Real.(Sfilt.spec ./ S0); color = colors[n], linewidth = 2,
            label = "Region $(region), filtered")
    end

    limits!(ax, (10^-4, 10^0), (1e-4, 1e8))
    axislegend(ax; position = :lb)

    save(filesave * "regionABC_iter$(iteration)_spectra_$(var).png", fig)
    println("Saved figure of $(var) spectra for regions $(join(regions, ", "))")
end

### -------------------------------------------------------------------------
## Main

# 1. Define subdomain

# Identify iteration (which denotes timestep)
iteration = 164410
println("Iteration: $(iteration)")
klev = 70

# Region A/B/C
region = "A"
println("--- Region $(region) ---")
fileparam = "region" * string(region)

# filter_scale = 300                                      # meters, bound between submesoscale and BLT
filter_scale = 100                                      # meters, for Gaussian filter to match spectra of 300m sharp spectral cutoff

# Cache of computed gradient variables (dTdx, dTdy, dudx, dudy, dvdx, dvdy, filtered & unfiltered)
grad_cache_file = filesave * fileparam * "_iter$(iteration)_grad.jld2"

# # 2. Load/compute variables and plot
# plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, "front")
# plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, "vort")
# plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, "div")
# plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, "strain_n")
# plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, "strain_s")
# plot_image(grad_cache_file, filehead, fileparam, iteration, klev, α, g, dx, dy, filter_scale, "strain")

# 2. Load/compute variables' spectra and plot spectra
plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, "front")
# plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, "vort")
# plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, "div")
# plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, "strain_n")
# plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, "strain_s")
# plot_spectra(iteration, klev, α, g, dx, dy, filter_scale, "strain")

# --- 
