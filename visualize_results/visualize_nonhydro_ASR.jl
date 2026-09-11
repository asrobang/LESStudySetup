using LESStudySetup
using CairoMakie, Makie
using FFTW
using Printf, Dates, StatsBase
using Statistics: mean, std, quantile
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes, xspacings, yspacings
using Oceananigans.Fields: interior
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_subdomain_snapshot
using LESStudySetup.Diagnostics: load_snapshots, isotropic_powerspectrum, δ
using MathTeXEngine
set_theme!(theme_latexfonts(), fontsize=12, figure_padding = 10)
using JLD2 #, CUDA

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "/home/asrobang/orcd/scratch/figures/20260908_banded_bgradmag/"

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

## Compute vorticity (\zeta) from 2D u and v velocity matrices
## \zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
## assumes uniform constant dx and dy
function vorticity2d(u,v,dx,dy) 
    v_pad = hcat(v, v[:, 1:1])      # pad the first column at the rightmost (Ny,Nx+1)
    u_pad = vcat(u, u[1:1, :])      # pad the first row at the bottom (Ny+1,Nx)
    dvdx = (v_pad[:, 2:end] .- v_pad[:, 1:end-1]) ./ dx
    dudy = (u_pad[2:end, :] .- u_pad[1:end-1, :]) ./ dy
    vort = dvdx .- dudy

    return vort
end

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

# |\nabla b_h|: the buoyancy gradient magnitude typically only includes the horizontal buoyancy gradient
function compute_bgradmag(T,α,g,dx,dy)
    dTdx, dTdy = compute_dAdx_dAdy(T,dx,dy) 

    bgradmag = (α*g) .* sqrt.(dTdx.^2 .+ dTdy.^2)

    return bgradmag
end

"""
    filter(field, smooth)

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
function filter(field, k, smooth)
    grid = field.grid
    Δx = minimum(xspacings(grid, Center()))
    Δy = minimum(yspacings(grid, Center()))

    data = Array(interior(field,:,:,k))  # bring to CPU physical-space array
    Nx, Ny = size(data, 1), size(data, 2)

    # Angular wavenumbers associated with the horizontal FFT
    kx = 2π .* fftfreq(Nx, 1/dx)
    ky = 2π .* fftfreq(Ny, 1/dy)

    K = 2π / smooth          # cutoff wavenumber for this smoothing scale
    κ² = (kx .^ 2) .+ (ky' .^ 2)
    mask = κ² .<= K^2        # true where κ ≤ K (keep), false where κ > K (zero)

    filtered = similar(data)

    f̂ = fft(data)
    f̂ .*= mask
    filtered .= real.(ifft(f̂))

    return filtered
end

using FFTW

"""
    bandpass_filter(field, dx, dy, cutoffs)

Split `field` into multiple horizontal bands using a series of sharp
spectral cutoffs (κ = √(k² + l²), periodic FFT in x, y).

`cutoffs` should be given as physical smoothing scales (same units as
dx, dy), sorted from largest scale to smallest, e.g.:

    cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]

which corresponds to wavenumber cutoffs K = 2π / cutoffs, i.e.
K = [2π/10⁴, 2π/10³, 2π/10², 2π/10¹, 2π/10⁰].

Returns a Dict mapping each band (as a string label) to the filtered field
containing only that band's wavenumbers. Bands are:

    band 1: κ ≤ K[1]                     (largest scales, > cutoffs[1])
    band 2: K[1] < κ ≤ K[2]
    band 3: K[2] < κ ≤ K[3]
    ...
    band N+1: κ > K[N]                   (smallest scales, < cutoffs[end])
"""
function bandpass_filter(field, dx, dy, cutoffs)
    Nx, Ny = size(field, 1), size(field, 2)

    kx = 2π .* fftfreq(Nx, 1/dx)
    ky = 2π .* fftfreq(Ny, 1/dy)
    κ² = (kx .^ 2) .+ (ky' .^ 2)

    # cutoff wavenumbers, sorted increasing (since cutoffs is sorted decreasing)
    K = 2π ./ cutoffs
    @assert issorted(K) "cutoffs must be sorted from largest scale to smallest (e.g. [1e4, 1e3, ..., 1e0])"

    f̂ = fft(field)
    results = Dict{String, Array{Float64}}()

    # band 1: everything below the first (smallest) cutoff wavenumber
    mask = κ² .<= K[1]^2
    results["κ ≤ 2π/$(cutoffs[1])"] = real.(ifft(f̂ .* mask))

    # middle bands: between consecutive cutoffs
    for n in 1:length(K)-1
        mask = (κ² .> K[n]^2) .& (κ² .<= K[n+1]^2)
        results["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])"] = real.(ifft(f̂ .* mask))
    end

    # last band: everything above the largest cutoff wavenumber
    mask = κ² .> K[end]^2
    results["κ > 2π/$(cutoffs[end])"] = real.(ifft(f̂ .* mask))

    return results
end

# ## Compute stratification (N^2) from vertical gradient of buoyancy 
# function strat(T,dz)
#     # ! ! ! fill in 
#     b = (α * g) .* T
#     N2 = ??? ./ dz # N2 = db/dz
#     return N2
# end

### -------------------------------------------------------------------------

## Plots a vertical slice of the vertical velocity w
function plot_w(snapshot, fileparam)
    # 3. Plot figure 
    fig = Figure(size = (640, 320))
    gab = fig[1, 1] = GridLayout()
    x, y, z = nodes(snapshot[:w]);
    yloc = round(Int, length(y)/2)
    ax_a = Axis(gab[1,1]; titlealign = :left, title=L"\text{(a)}~w~\text{(mm s^{-1})}", xlabel=L"x~\text{(km)}", ylabel=L"z~\text{(m)}",limits=(nothing,(-90,0)))
    ax_b = Axis(gab[1,3]; titlealign = :left, title=L"\text{(b)}", xlabel=L"10^6\langle\text{KE}_w\rangle~\text{(m^2~s^{-2})}",limits=(nothing,(-90,0)))
    hm_a = heatmap!(ax_a, 1e-3x, z, abs.(1e3interior(snapshot[:w], :, yloc, :)); rasterize = true, colormap = :delta, colorrange = (0, 20))
    Colorbar(gab[1,2], hm_a)
    # vlines!(ax_a, 1e-3x[[512,1024,1536]], color = [:orange, :green, :purple], linewidth = 0.8)
    hideydecorations!(ax_b, ticks = false)
    lines!(ax_b, 1e6*vec(mean(interior(snapshot[:w], :, yloc, :).^2/2, dims =(1))), z; linewidth = 1)
    # lines!(ax_b, 1e6*interior(snapshot[:w], 512, yloc, :).^2/2, z; linewidth = 1)
    # lines!(ax_b, 1e6*interior(snapshot[:w], 1024, yloc, :).^2/2, z; linewidth = 1)
    # lines!(ax_b, 1e6*interior(snapshot[:w], 1536, yloc, :).^2/2, z; linewidth = 1)
    Label(gab[0, 1:3], "Vertical velocity snapshot and turbulent KE with depth: $(fileparam)", fontsize = 12)
    colsize!(gab, 3, Relative(0.3))
    colgap!(gab, 1, 1)
    resize_to_layout!(fig)
    save(filesave * "wvslice_abs_" * fileparam * "_iter$(iteration).pdf", fig; pt_per_unit = 1)
    println("Finished plotting w fields")
end

## Plots horizontal and vertical slices of T, w, u, v fields at a specified iteration
## depth? 
## part at which vertical slice is made? 

## 1. Define the filename of the saved snapshot
## output_filename = filehead * "subdomains/" * fileparam * "_snapshot_iter$(iteration).jld2"
## 2. Load the snapshot using the new function
## snapshot = load_subdomain_snapshot(output_filename)
function plot_Twuv(snapshot)
    # 3. Plot figure
    x, y, z = nodes(snapshot[:T]);
    _, _, zw = nodes(snapshot[:w]);
    k = 72      # takes surface depth=0m
    Tmap, wmap, vmap = :thermal,:delta,:balance
    wmax,umax,vmax=0.01,0.15,0.2
    #fig = Figure(size = (640, 450))
    fig = Figure(size = (640, 750))
    gabc = fig[1, 1] = GridLayout()
    aspect = 1
    axis_kwargs = (ylabel = L"y~\text{(km)}", aspect=aspect)
    ax_a = Axis(gabc[1,1]; titlealign = :left, title=L"\text{(a)}~T~\text{({^\circ}C)}", axis_kwargs...)
    ax_b = Axis(gabc[1,3]; titlealign = :left, title=L"\text{(b)}~w~\text{(m s^{-1})}", aspect=aspect)
    ax_c = Axis(gabc[3,1]; titlealign = :left, title=L"\text{(c)}~u~\text{(m s^{-1})}", axis_kwargs...) 
    ax_d = Axis(gabc[3,3]; titlealign = :left, title=L"\text{(d)}~v~\text{(m s^{-1})}", aspect=aspect) 
    hm_a = heatmap!(ax_a, 1e-3x, 1e-3y, (interior(snapshot[:T],:,:,k)); rasterize = true, colormap = Tmap)
    hm_b = heatmap!(ax_b, 1e-3x, 1e-3y, (interior(snapshot[:w],:,:,k)); rasterize = true, colormap = wmap, colorrange = (-wmax, wmax))
    hm_c = heatmap!(ax_c, 1e-3x, 1e-3y, (interior(snapshot[:u],:,:,k)); rasterize = true, colormap = vmap, colorrange = (-umax, umax))
    hm_d = heatmap!(ax_d, 1e-3x, 1e-3y, (interior(snapshot[:v],:,:,k)); rasterize = true, colormap = vmap, colorrange = (-vmax, vmax))
    Colorbar(gabc[1,2], hm_a)
    Colorbar(gabc[1,4], hm_b)
    Colorbar(gabc[3,2], hm_c)
    Colorbar(gabc[3,4], hm_d)
    hidexdecorations!(ax_a, ticks = false)
    hidexdecorations!(ax_b, ticks = false)
    hidexdecorations!(ax_c, ticks = false)
    hidexdecorations!(ax_d, ticks = false)
    hideydecorations!(ax_b, ticks = false)
    hideydecorations!(ax_d, ticks = false)

    # T̄ = (mean(snapshot[:T], dims = 2));
    # w̄ = (mean(snapshot[:w], dims = 2));
    # ū = (mean(snapshot[:u], dims = 2));
    # v̄ = (mean(snapshot[:v], dims = 2));
    zmin = -75
    kz = findfirst(z .≥ zmin)
    Nz = length(z)
    axis_kwargs0 = (xlabel = L"x~\text{(km)}", ylabel = L"z~\text{(m)}", limits = (nothing, (zmin, 0)))
    axis_kwargs1 = NamedTuple{(:xlabel,:ylabel)}(axis_kwargs0)
    ax_a = Axis(gabc[2,1]; titlealign = :left, axis_kwargs0...)
    ax_b = Axis(gabc[2,3]; titlealign = :left, xlabel = L"x~\text{(km)}", limits = (nothing, (zmin, 0)))
    ax_c = Axis(gabc[4,1]; titlealign = :left,  limits = (nothing, (zmin, 0)), axis_kwargs1...)
    ax_d = Axis(gabc[4,3]; titlealign = :left, xlabel = L"x~\text{(km)}", limits = (nothing, (zmin, 0)))
    wmax,umax,vmax=0.01,0.1,0.2
    hm_a = heatmap!(ax_a, 1e-3x, z[kz:Nz], (interior(snapshot[:T],:,3*640,kz:Nz)); rasterize = true, colormap = Tmap)#, colorrange = (vmin, vmax))
    hm_b = heatmap!(ax_b, 1e-3x, zw[kz:Nz], (interior(snapshot[:w],:,3*640,kz:Nz)); rasterize = true, colormap = wmap, colorrange = (-wmax, wmax))
    hm_c = heatmap!(ax_c, 1e-3x, z[kz:Nz], (interior(snapshot[:u],:,3*640,kz:Nz)); rasterize = true, colormap = vmap, colorrange = (-umax, umax))
    hm_d = heatmap!(ax_d, 1e-3x, z[kz:Nz], (interior(snapshot[:v],:,3*640,kz:Nz)); rasterize = true, colormap = vmap, colorrange = (-vmax, vmax))
    hideydecorations!(ax_b, ticks = false)
    hideydecorations!(ax_d, ticks = false)
    Colorbar(gabc[2,2], hm_a)
    Colorbar(gabc[2,4], hm_b)
    Colorbar(gabc[4,2], hm_c)
    Colorbar(gabc[4,4], hm_d)
    rowgap!(gabc, 4)
    colgap!(gabc, 1, 1)
    colgap!(gabc, 3, 1)
    colgap!(gabc, 2, 5)
    for row = [2,4]
        rowsize!(gabc, row, Relative(0.14))
    end
    resize_to_layout!(fig)
    save(filesave * "Twuv_" * fileparam * "_3d_iter$(iteration).pdf", fig; pt_per_unit = 1)
    println("Finished plotting Twuv fields")
end

### -------------------------------------------------------------------------

## Plots the temperature heatmap for a subdomain tile with no padding or colorbar
function plot_T_image(snapshot, fileparam; Tmin=19, Tmax=21, k=70, colormap=:thermal)
    x, y, _ = nodes(snapshot[:T])

    fig = Figure(size = (640, 640), figure_padding = 0)
    # fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())

    hm = heatmap!(ax, 1e-3x, 1e-3y, interior(snapshot[:T], :, :, k);
             rasterize = true, colormap = colormap,
             colorrange = (Tmin, Tmax))
            # )

    # Colorbar(fig[1, 2], hm)             # if you want colorbar
    # hlines!(ax, 1e-3y[round(Int, length(y)/2)], color = :white, linewidth = 2)

    # comment out to bring back axes
    hidedecorations!(ax)
    hidespines!(ax)
    tightlimits!(ax)

    save(filesave * "T_" * fileparam * "_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting T heatmap")
end

## Plots the colorbar for a heatmap of temperature for a subdomain tile
function plot_T_colorbar(; Tmin=19, Tmax=21, colormap=:thermal, vertical=true)
    fig = Figure(size = vertical ? (120, 500) : (500, 120), figure_padding = 5)

    # limits = (Tmin, Tmax), 
    Colorbar(fig[1, 1]; limits = (Tmin, Tmax), colormap = colormap,
              vertical = vertical, label = L"T~\text{({^\circ}C)}")

    save(filesave * "T_colorbar_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting T colorbar")
end

## Plots the heatmap for a subdomain tile with no padding or colorbar
## available variables: u, v, w, vort, vortf, hke
function plot_image(snapshot, var, fileparam; Tmin=19, Tmax=21, k=70, colormap=:thermal)
    if (var == :u || var == :v || var == :w)
        x, y, _ = nodes(snapshot[var])
    else
        x, y, _ = nodes(snapshot[:T])
    end

    fig = Figure(size = (640, 640), figure_padding = 0)
    # fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())

    if (var == :u)
        varlabel = "u"
        colormap = :balance
        Tmin, Tmax = -0.4, 0.4
    elseif (var == :v)
        varlabel = "v"
        colormap = :balance
        Tmin, Tmax = -0.4, 0.4
    elseif (var == :w)
        varlabel = "w"
        colormap = :delta
        Tmin, Tmax = -0.05, 0.05
    elseif (var == :vort)
        varlabel = "vort"
        colormap = :curl
        Tmin, Tmax = -0.05, 0.05
    elseif (var == :vortf)
        varlabel = "vortf"
        colormap = :curl
        Tmin, Tmax = -600, 600
    elseif (var == :hke)
        varlabel = "hke"
        colormap = :viridis
        Tmin, Tmax = 0.0, 0.08
    else
        varlabel = "unknown"
        colormap = :dense
    end

    if (var == :vort)
        whole_u = copy(interior(snapshot[:u],:,:,k)[1:end-1, :])
        whole_v = copy(interior(snapshot[:v],:,:,k)[:, 1:end-1])
        field = vorticity2d(whole_u,whole_v,dx,dy)
    elseif (var == :vortf)
        whole_u = copy(interior(snapshot[:u],:,:,k)[1:end-1, :])
        whole_v = copy(interior(snapshot[:v],:,:,k)[:, 1:end-1])
        field = vorticity2d(whole_u,whole_v,dx,dy) ./ f       
    elseif (var == :hke)
        # Extract u and v velocity fields 
        whole_u = copy(interior(snapshot[:u],:,:,k)[1:end-1, :])
        whole_v = copy(interior(snapshot[:v],:,:,k)[:, 1:end-1])
        # Calculate horizontal KE
        field = (whole_u.^2 .+ whole_v.^2) ./ 2 
    else
        field = interior(snapshot[var], :, :, k)
    end

    hm = heatmap!(ax, 1e-3x, 1e-3y, field;
                rasterize = true, colormap = colormap,
                 colorrange = (Tmin, Tmax))
                # )

    # Colorbar(fig[1, 2], hm)             # if you want colorbar

    hidedecorations!(ax)
    hidespines!(ax)
    tightlimits!(ax)

    save(filesave * varlabel * "_" * fileparam * "_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting " * varlabel * " heatmap")

    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    varlabel = "q (potential vorticity)"
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_f;
                rasterize = true, colormap = colormap)
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * "q_f.png", fig; px_per_unit=4)
end

## Plots the colorbar for a heatmap of temperature for a subdomain tile
## available variables: u, v, w, vort, vortf, hke
function plot_colorbar(var; Tmin=19, Tmax=21, colormap=:thermal, vertical=true)
    fig = Figure(size = vertical ? (120, 500) : (500, 120), figure_padding = 5)

    if (var == :u)
        varlabel = "u"
        axlabel = L"u~\text{(m/s)}"
        colormap = :balance
        Tmin, Tmax = -0.4, 0.4
    elseif (var == :v)
        varlabel = "v"
        axlabel = L"v~\text{(m/s)}"
        colormap = :balance
        Tmin, Tmax = -0.4, 0.4
    elseif (var == :w)
        varlabel = "w"
        axlabel = L"w~\text{(m/s)}"
        colormap = :delta
        Tmin, Tmax = -0.05, 0.05
    elseif (var == :vort)
        varlabel = "vort"
        axlabel = L"\text{vorticity}"
        colormap = :curl
        Tmin, Tmax = -0.05, 0.05
    elseif (var == :vortf)
        varlabel = "vortf"
        axlabel = L"\frac{\zeta}{f} \text{effective Rossby number}"
        colormap = :curl
        Tmin, Tmax = -600, 600
    elseif (var == :hke)
        varlabel = "hke"
        axlabel = L"\text{horizontal KE} (J)"
        colormap = :viridis
        Tmin, Tmax = 0.0, 0.08
    else
        varlabel = "unknown"
        colormap = :dense
    end
 
    Colorbar(fig[1, 1]; limits = (Tmin, Tmax), colormap = colormap,
              vertical = vertical, label = axlabel)

    save(filesave * varlabel * "_colorbar_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting " * varlabel * " colorbar")
end

## Plots the heatmap of a vertical slice for a subdomain tile with no padding or colorbar
## available variables: u, v, w, vort
## vertical slice at fixed y: cross-front slice ! preferred
## vertical slice at fixed x: along-front slice
function plot_vimage(snapshot, var, fileparam; Tmin=19, Tmax=21, colormap=:thermal)

    # if (var == :u || var == :v || var == :w)
    #     x, y, _ = nodes(snapshot[var])
    # else
    #     x, y, _ = nodes(snapshot[:T])
    # end
    x, y, z = nodes(snapshot[:w]);      # grid-node coordinates associated with :w

    # fig = Figure(size = (640, 640), figure_padding = 0)
    # # fig = Figure(size = (700, 640))     # if you want colorbar
    # ax = Axis(fig[1, 1]; aspect = DataAspect())
    fig = Figure(size = (640, 320))     # 640 pixels wide, 320 pixels high
    gab = fig[1, 1] = GridLayout()
    ax_a = Axis(gab[1,1];
            titlealign = :left, 
            title=L"\text{(a)}~w~\text{(mm s^{-1})}", 
            xlabel=L"x~\text{(km)}",        # labels horizontal axis 
            ylabel=L"z~\text{(m)}",         # labels vertical axis
            limits=(nothing,(-90,0))        # z-axis limits from -90m to 0m
            )
    ax_b = Axis(gab[1,3]; 
            titlealign = :left, 
            title=L"\text{(b)}", 
            xlabel=L"10^6\langle\text{KE}_w\rangle~\text{(m^2~s^{-2})}",    # vertical profiles of KE assoc. with w
            limits=(nothing,(-90,0))
            )

    # 1e3 to convert w from m/s to mm/s
    field = 1e3interior(snapshot[:w], :, 1497, :)   # use y-index 1497

    # hm = heatmap!(ax, 1e-3x, 1e-3y, field;
    #             rasterize = true, colormap = colormap,
    #              colorrange = (Tmin, Tmax))
    #             # )
    hm_a = heatmap!(ax_a, 1e-3x, z, field;      # x in km, z in m
                rasterize = true, 
                colormap = :delta, 
                colorrange = (-20, 20)          # colorscale between -20 and 20 mm/s
                )

    # Colorbar(fig[1, 2], hm)             # if you want colorbar
    Colorbar(gab[1,2], hm_a)

    # hidedecorations!(ax)
    # hidespines!(ax)
    # tightlimits!(ax)
    # draws vertical lines at x-coordinates at indices 2004, 1004
    vlines!(ax_a, 1e-3x[[2004,1004]], color = [:orange, :green], linewidth = 0.8)
    
    hideydecorations!(ax_b, ticks = false)
    lines!(ax_b, 1e6*vec(mean(interior(snapshot[:w], :, 1497, :).^2/2, dims =(1))), z; linewidth = 1)
    lines!(ax_b, 1e6*interior(snapshot[:w], 2004, 1497, :).^2/2, z; linewidth = 1)
    lines!(ax_b, 1e6*interior(snapshot[:w], 1004, 1497, :).^2/2, z; linewidth = 1)
    colsize!(gab, 3, Relative(0.3))
    colgap!(gab, 1, 1)
    resize_to_layout!(fig)

    # save(filesave * varlabel * "_" * fileparam * "_iter$(iteration).png", fig; px_per_unit = 4)
    # println("Finished plotting " * varlabel * " heatmap")
    save(filesave * "w_" * fileparam * "_2d_iter$(iteration).pdf", fig; px_per_unit = 4)
    println("Finished plotting w fields")
end

function plot_filteredPV(snapshot)
    # filtered
    T_f = filter(snapshot[:T], 70, 4*dx)                    # 2048×2048 Matrix{Float32}
    u_f = filter(snapshot[:u], 70, 4*dx)[1:end-1, :]        # initially 2049×2048 Matrix{Float32}
    v_f = filter(snapshot[:v], 70, 4*dx)[:, 1:end-1]        # initially 2048×2049 Matrix{Float32}
    w_f = filter(snapshot[:w], 70, 4*dx)                    # 2048×2048 Matrix{Float32}
    T_f_above = filter(snapshot[:T], 71, 4*dx)
    u_f_above = filter(snapshot[:u], 71, 4*dx)[1:end-1, :]
    v_f_above = filter(snapshot[:v], 71, 4*dx)[:, 1:end-1]
    w_f_above = filter(snapshot[:w], 71, 4*dx)
    T_f_below = filter(snapshot[:T], 69, 4*dx)
    u_f_below = filter(snapshot[:u], 69, 4*dx)[1:end-1, :]
    v_f_below = filter(snapshot[:v], 69, 4*dx)[:, 1:end-1]
    w_f_below = filter(snapshot[:w], 69, 4*dx)

    dTdx, dTdy = compute_dAdx_dAdy(T_f,dx,dy) 
    dudx, dudy = compute_dAdx_dAdy(u_f,dx,dy) 
    dvdx, dvdy = compute_dAdx_dAdy(v_f,dx,dy) 
    dwdx, dwdy = compute_dAdx_dAdy(w_f,dx,dy) 
    dTdz = compute_dAdz(T_f_above,T_f_below,dz)
    dudz = compute_dAdz(u_f_above,u_f_below,dz)
    dvdz = compute_dAdz(v_f_above,v_f_below,dz)

    q_f = (α*g) .* ( dTdx .* (dwdy.-dvdz) .+ dTdy .* (dudz.-dwdx) .+ dTdz .* (dvdx.-dudy.+f) ) 

    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_f;
                rasterize = true, colormap = colormap, colorrange = (-4e-6, 4e-6))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * "q_f.png", fig; px_per_unit=4)

    # unfiltered! 
    T_uf = copy(interior(snapshot[:T],:,:,70))                    # 2048×2048 Matrix{Float32}
    u_uf = copy(interior(snapshot[:u],:,:,70)[1:end-1, :])        # initially 2049×2048 Matrix{Float32}
    v_uf = copy(interior(snapshot[:v],:,:,70)[:, 1:end-1])        # initially 2048×2049 Matrix{Float32}
    w_uf = copy(interior(snapshot[:w],:,:,70))                    # 2048×2048 Matrix{Float32}
    T_above = copy(interior(snapshot[:T],:,:,71))
    u_above = copy(interior(snapshot[:u],:,:,71)[1:end-1, :])
    v_above = copy(interior(snapshot[:v],:,:,71)[:, 1:end-1])
    w_above = copy(interior(snapshot[:w],:,:,71))
    T_below = copy(interior(snapshot[:T],:,:,69))
    u_below = copy(interior(snapshot[:u],:,:,69)[1:end-1, :])
    v_below = copy(interior(snapshot[:v],:,:,69)[:, 1:end-1])
    w_below = copy(interior(snapshot[:w],:,:,69))

    dTdx, dTdy = compute_dAdx_dAdy(T_uf,dx,dy) 
    dudx, dudy = compute_dAdx_dAdy(u_uf,dx,dy) 
    dvdx, dvdy = compute_dAdx_dAdy(v_uf,dx,dy) 
    dwdx, dwdy = compute_dAdx_dAdy(w_uf,dx,dy) 
    dTdz = compute_dAdz(T_above,T_below,dz)
    dudz = compute_dAdz(u_above,u_below,dz)
    dvdz = compute_dAdz(v_above,v_below,dz)

    q_uf = (α*g) .* ( dTdx .* (dwdy.-dvdz) .+ dTdy .* (dudz.-dwdx) .+ dTdz .* (dvdx.-dudy.+f) ) 

    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_uf;
                rasterize = true, colormap = colormap, colorrange = (-4e-6, 4e-6))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * "q_uf.png", fig; px_per_unit=4)

    # filtered vs. unfiltered T 
    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, T_f;
                rasterize = true, colormap = colormap, colorrange = (19.5, 20.1))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * "T_f.png", fig; px_per_unit=4)

    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, T_uf;
                rasterize = true, colormap = colormap, colorrange = (19.5, 20.1))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * "T_uf.png", fig; px_per_unit=4)
end

### -------------------------------------------------------------------------

# ### Loop through multiple files and plot w, Twuv using Shirui's functions 
# # Get names of matching files in that specific folder
# file_arr = filter(f -> startswith(f, "subdomain"), readdir(filehead))
# println(file_arr)

# for curr_file in file_arr
#     println(curr_file)
#     # 1. Define the filename of the saved snapshot
#     # output_filename = filehead * "subdomains/" * fileparam * "_snapshot_iter$(iteration).jld2"
#     # output_filename = filehead * "subdomains/" * fileparam * "_iter$(iteration).jld2"
#     output_filename = filehead * curr_file

#     # 2. Load the snapshot using the new function
#     snapshot = load_subdomain_snapshot(output_filename)

#     # # 3. Plot figure
#     # plot_w(snapshot)
#     # plot_Twuv(snapshot)
# end

### -------------------------------------------------------------------------
## Plot the heatmap plots of each subdomain tile

# 1. Define file parameters
# Loop over subdomain files
iteration = 164410
println("Iteration: $(iteration)")

# max_T = 0.0
# min_T = 1000.0

# --- 10km x 10km Tiles ---
i = 97  # temporarily commented out for loop 
# for i in 97:97 #1:100
    println("--- Subdomain $(i) ---")

    # 2. Define the filename of the saved snapshot
    fileparam = "subdomain" * string(i)
    output_filename = filehead * fileparam * "_iter$(iteration).jld2"

    # 3. Load the snapshot
    snapshot = load_subdomain_snapshot(output_filename)

    # ---

    T_uf = copy(interior(snapshot[:T],:,:,70))
    T_above = copy(interior(snapshot[:T],:,:,71))
    T_below = copy(interior(snapshot[:T],:,:,69))

    field = compute_bgradmag(T_uf,α,g,dx,dy)
    dTdz = compute_dAdz(T_above,T_below,dz)
    dbdz = (α*g) .* dTdz

    # quick plot of field
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = L"$|\nabla b_h|$")
    colormap = :amp
    m_field = maximum(abs, field)  # symmetric bound around 0
    m_field = m_field == 0 ? 1e-10 : m_field   # avoid zero-width colorrange
    hm = heatmap!(ax, 1e-3x, 1e-3y, field;
                rasterize = true, colormap = colormap, colorrange=(0,m_field))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    # save(filesave * "bgradmag_uf.png", fig; px_per_unit=4)

    # quick plot of spectra 
    S_bgradmag = isotropic_powerspectrum(field, field; Δx=dx, Δy=dy)
    fig = Figure(size = (600, 500))
    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_{|\nabla b_h|}(k)/E_{|\nabla b_h|} (k_{min})",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
    ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)
    global St0 = S_bgradmag
    lines!(ax, S_bgradmag.freq, Real.(S_bgradmag.spec./St0.spec[1]))
    xlims!(ax, (10^-4.5, 10^0.5))
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    save(filesave * "spectra_bgradmag_uf.png", fig)

    cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]
    bands = bandpass_filter(field, dx, dy, cutoffs)

    # # access a specific band
    # bands["κ ≤ 2π/10000.0"]
    # bands["2π/10000.0 < κ ≤ 2π/1000.0"]
    # bands["2π/1000.0 < κ ≤ 2π/100.0"]
    # bands["2π/100.0 < κ ≤ 2π/10.0"]
    # bands["2π/10.0 < κ ≤ 2π/1.0"]

    # labels = collect(keys(bands))   # unsorted
    labels = ["κ ≤ 2π/10000.0", "2π/10000.0 < κ ≤ 2π/1000.0", "2π/1000.0 < κ ≤ 2π/100.0",   # sorted 
                "2π/100.0 < κ ≤ 2π/10.0", "2π/10.0 < κ ≤ 2π/1.0", "κ > 2π/1.0"]
    n = length(labels)
    ncols_ = 3
    nrows_ = ceil(Int, n / ncols_)

    fig = Figure(size = (400*ncols_, 350*nrows_))
    for (idx, label) in enumerate(labels)
        row = div(idx - 1, ncols_) + 1
        col = mod(idx - 1, ncols_) + 1

        ax = Axis(fig[row, col], title = label, aspect = DataAspect())
        data = bands[label]

        m = maximum(abs, data)  # symmetric bound around 0
        m = m == 0 ? 1e-10 : m   # avoid zero-width colorrange
        hm = heatmap!(ax, 1e-3x, 1e-3y, data, colormap = :balance, colorrange = (-2e-5, 2e-5))
        Colorbar(fig[row, col+ncols_], hm, width = 10; label = L"\text{reconstructed } $|\nabla b_h|$")
    end
    # save(filesave * "bands_bgradmag_uf.png", fig)

    # # Useless plot of bgradmag squared
    # fig = Figure(size = (400*ncols_, 350*nrows_))
    # for (idx, label) in enumerate(labels)
    #     row = div(idx - 1, ncols_) + 1
    #     col = mod(idx - 1, ncols_) + 1

    #     ax = Axis(fig[row, col], title = label, aspect = DataAspect())
    #     data = bands[label] .^ 2

    #     m = maximum(abs, data)  # symmetric bound around 0
    #     m = m == 0 ? 1e-16 : m   # avoid zero-width colorrange
    #     hm = heatmap!(ax, 1e-3x, 1e-3y, data, colormap = :amp, colorrange = (0, 1e-11))
    #     Colorbar(fig[row, col+ncols_], hm, width = 10; label = L"\text{reconstructed } $|\nabla b_h|^2$")
    # end
    # # save(filesave * "bands_bgradmagsquared_uf.png", fig)

    # Extrema of bands
    # julia> extrema(bands["κ ≤ 2π/10000.0"])
    # (1.367743253396196e-6, 1.367743253396196e-6)
    # julia> extrema(bands["2π/10000.0 < κ ≤ 2π/1000.0"])
    # (-1.4721677960247549e-6, 2.7241948733091438e-6)
    # julia> extrema(bands["2π/1000.0 < κ ≤ 2π/100.0"])
    # (-9.373607586752127e-6, 1.8252773497426324e-5)
    # julia> extrema(bands["2π/100.0 < κ ≤ 2π/10.0"])
    # (-2.9840358924229248e-5, 0.00016119601865659347)
    # julia> extrema(bands["2π/10.0 < κ ≤ 2π/1.0"])
    # (-1.8261886925726886e-5, 1.9952158482386392e-5)

    # Extrema of bands squared
    # julia> extrema(bands["κ ≤ 2π/10000.0"].^2)
    # (1.870721607210811e-12, 1.870721607210811e-12)
    # julia> extrema(bands["2π/10000.0 < κ ≤ 2π/1000.0"].^2)
    # (1.0653826322057696e-25, 7.421237707763823e-12)
    # julia> extrema(bands["2π/1000.0 < κ ≤ 2π/100.0"].^2)
    # (3.1800156390852164e-28, 3.331637403483488e-10)
    # julia> extrema(bands["2π/100.0 < κ ≤ 2π/10.0"].^2)
    # (2.7267589554592325e-25, 2.598415643073683e-8)
    # julia> extrema(bands["2π/10.0 < κ ≤ 2π/1.0"].^2)
    # (2.294106309503896e-26, 3.9808862810626327e-10)

    # Sum bands together 
    band_sum = (bands["κ ≤ 2π/10000.0"] + bands["2π/10000.0 < κ ≤ 2π/1000.0"] + bands["2π/1000.0 < κ ≤ 2π/100.0"]
                + bands["2π/100.0 < κ ≤ 2π/10.0"] + bands["2π/10.0 < κ ≤ 2π/1.0"])

    # quick plot of sum of bands 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = L"Sum of bands of $|\nabla b_h|$")
    colormap = :amp
    hm = heatmap!(ax, 1e-3x, 1e-3y, band_sum;
                rasterize = true, colormap = colormap, colorrange=(0,m_field))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    # save(filesave * "bgradmag_bandsum_uf.png", fig; px_per_unit=4)

    # julia> maximum(field-band_sum)
    # 8.131516293641283e-20

    # to-do 
    # - observe these in terms of spectra with much, much narrower bands
    # - color the spectra plot by its max frontal sharpness
    # 

    # ---

    # quick plot of db/dz
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = L"$\frac{\partial b}{\partial z}$")
    colormap = :balance
    m_dbdz = maximum(abs, dbdz)  # symmetric bound around 0
    m_dbdz = m_dbdz == 0 ? 1e-10 : m_dbdz   # avoid zero-width colorrange
    hm = heatmap!(ax, 1e-3x, 1e-3y, dbdz;
                rasterize = true, colormap = colormap, colorrange=(-m_dbdz,m_dbdz))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * "dbdz_uf.png", fig; px_per_unit=4)

    # quick plot of spectra of db/dz
    S_dbdz = isotropic_powerspectrum(dbdz, dbdz; Δx=dx, Δy=dy)
    fig = Figure(size = (600, 500))
    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_{db/dz}(k)/E_{db/dz} (k_{min})",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
    ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)
    global St0 = S_dbdz
    lines!(ax, S_dbdz.freq, Real.(S_dbdz.spec./St0.spec[1]))
    xlims!(ax, (10^-4.5, 10^0.5))
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    save(filesave * "spectra_dbdz_uf.png", fig)

    cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]
    bands_dbdz = bandpass_filter(dbdz, dx, dy, cutoffs)

    labels = ["κ ≤ 2π/10000.0", "2π/10000.0 < κ ≤ 2π/1000.0", "2π/1000.0 < κ ≤ 2π/100.0",   # sorted 
                "2π/100.0 < κ ≤ 2π/10.0", "2π/10.0 < κ ≤ 2π/1.0", "κ > 2π/1.0"]
    n = length(labels)
    ncols_ = 3
    nrows_ = ceil(Int, n / ncols_)

    fig = Figure(size = (400*ncols_, 350*nrows_))
    for (idx, label) in enumerate(labels)
        row = div(idx - 1, ncols_) + 1
        col = mod(idx - 1, ncols_) + 1

        ax = Axis(fig[row, col], title = label, aspect = DataAspect())
        data = bands_dbdz[label]

        m = maximum(abs, data)  # symmetric bound around 0
        m = m == 0 ? 1e-10 : m   # avoid zero-width colorrange
        hm = heatmap!(ax, 1e-3x, 1e-3y, data, colormap = :balance, colorrange = (-2e-5, 2e-5))
        Colorbar(fig[row, col+ncols_], hm, width = 10; label = L"\text{reconstructed } $\frac{\partial b}{\partial z}$")
    end
    save(filesave * "bands_dbdz_uf.png", fig)

    # ---

    # Color spectra plot according to max bgradmag in each band regime

    freqs = collect(S_bgradmag.freq)
    spec_normalized = Real.(S_bgradmag.spec ./ S_bgradmag.spec[1])

    # --- band-based coloring ---
    # cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]
    cutoffs = 10 .^ range(4, 0, length=100)   # 100 log-spaced values from 1e4 down to 1e0
    bands = bandpass_filter(field, dx, dy, cutoffs)

    K = 2π ./ cutoffs                      # band edge wavenumbers, increasing
    labels = ["κ ≤ 2π/$(cutoffs[1])",
            ["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])" for n in 1:length(cutoffs)-1]...,
            "κ > 2π/$(cutoffs[end])"]

    band_max_bgradmag = [maximum(abs, bands[label]) for label in labels]  # from your bandpass_filter output on `field`

    color_vals = similar(freqs, Float64)
    for (i, κ) in enumerate(freqs)
        if κ <= K[1]
            color_vals[i] = band_max_bgradmag[1]
        elseif κ > K[end]
            color_vals[i] = band_max_bgradmag[end]
        else
            band_idx = findfirst(n -> κ <= K[n+1], 1:length(K)-1) + 1
            color_vals[i] = band_max_bgradmag[band_idx]
        end
    end

    # --- plot ---
    fig = Figure(size = (600, 500))
    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E_{|\nabla b_h|}(k)/E_{|\nabla b_h|} (k_{min})",
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
    ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

    lp = lines!(ax, freqs, spec_normalized, color = color_vals,
                colormap = :amp, linewidth = 2)

    Colorbar(fig[1,2], lp, label = L"\max |\nabla b|\text{ in band}")

    xlims!(ax, (10^-4.5, 10^0.5))
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    save(filesave * "spectracolor_bgradmag_uf.png", fig)

    # ---

    # Color spectra plot according to max dbdz in each band regime

    freqs = collect(S_dbdz.freq)
    spec_normalized = Real.(S_dbdz.spec ./ S_dbdz.spec[1])

    # --- band-based coloring ---
    cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]
    # cutoffs = 10 .^ range(4, 0, length=100)   # 100 log-spaced values from 1e4 down to 1e0
    bands_dbdz = bandpass_filter(dbdz, dx, dy, cutoffs)

    K = 2π ./ cutoffs                      # band edge wavenumbers, increasing
    labels = ["κ ≤ 2π/$(cutoffs[1])",
            ["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])" for n in 1:length(cutoffs)-1]...,
            "κ > 2π/$(cutoffs[end])"]

    band_max_dbdz = [maximum(abs, bands_dbdz[label]) for label in labels]

    color_vals = similar(freqs, Float64)
    for (i, κ) in enumerate(freqs)
        if κ <= K[1]
            color_vals[i] = band_max_dbdz[1]
        elseif κ > K[end]
            color_vals[i] = band_max_dbdz[end]
        else
            band_idx = findfirst(n -> κ <= K[n+1], 1:length(K)-1) + 1
            color_vals[i] = band_max_dbdz[band_idx]
        end
    end

    # --- plot ---
    fig = Figure(size = (600, 500))
    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E_{\frac{\partial b}{\partial z}}(k)/E_{\frac{\partial b}{\partial z}} (k_{min})",
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
    ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

    lp = lines!(ax, freqs, spec_normalized, color = color_vals,
                colormap = :amp, linewidth = 2)

    Colorbar(fig[1,2], lp, label = L"\max {|\frac{\partial b}{\partial z}|} \text{ in band}")

    xlims!(ax, (10^-4.5, 10^0.5))
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    save(filesave * "spectracolor_dbdz_uf.png", fig)
    
    # ---

    # # update max and min T
    # curr_max = maximum(snapshot[:T])
    # curr_min = minimum(snapshot[:T])
    # global max_T = max(max_T, curr_max)
    # global min_T = min(min_T, curr_min)
    # println("Max T so far: $(max_T)")
    # println("Min T so far: $(min_T)")

    # 4. Plot figure
    # plot_T_image(snapshot, fileparam; Tmin=19.5, Tmax=20.1)
    # plot_image(snapshot, :u, fileparam; k=70)
    # plot_image(snapshot, :v, fileparam; k=70)
    # plot_image(snapshot, :w, fileparam; k=70)
    # plot_image(snapshot, :vort, fileparam; k=70)
    # plot_image(snapshot, :vortf, fileparam; k=70)
    # plot_image(snapshot, :hke, fileparam; k=70)
    # plot_image(snapshot, :pv, fileparam; k=70)
    # plot_w(snapshot, fileparam)
    # plot_filteredPV(snapshot)     # needs to be cleaned up 
# end

# println("FINAL Max T: $(max_T)")
# println("FINAL Min T: $(min_T)")

# plot_T_colorbar(; Tmin=19.5, Tmax=20.1)
# plot_colorbar(:u)
# plot_colorbar(:v)
# plot_colorbar(:w)
# plot_colorbar(:vort)
# plot_colorbar(:vortf)
# plot_colorbar(:hke)
# plot_colorbar(:pv)

# ### -------------------------------------------------------------------------
# ## Plot the heatmap plots of one of regions A, B, and C

# # --- Regions A, B, C ---
# region = "B"
# println("--- Region $(region) ---")

# # 2. Define the filename of the saved snapshot
# fileparam = "region" * string(region)

# # 3. Load the snapshot and plot image - T
# output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"
# snapshot = load_subdomain_snapshot(output_filename)
# plot_T_image(snapshot, fileparam; Tmin=19.5, Tmax=20.1)
# println("Freeing large variables...")
# snapshot = nothing      # free the variable
# GC.gc()                 # force the garbage collector to run immediately

# # 3. Load the snapshot and plot image - u
# output_filename = filehead * "subdomain_u_" * fileparam * "_iter$(iteration).jld2"
# snapshot = load_subdomain_snapshot(output_filename)
# plot_image(snapshot, :u, fileparam)
# println("Freeing large variables...")
# snapshot = nothing      # free the variable
# GC.gc()                 # force the garbage collector to run immediately

# # 3. Load the snapshot and plot image - v
# output_filename = filehead * "subdomain_v_" * fileparam * "_iter$(iteration).jld2"
# snapshot = load_subdomain_snapshot(output_filename)
# plot_image(snapshot, :v, fileparam)
# println("Freeing large variables...")
# snapshot = nothing      # free the variable
# GC.gc()                 # force the garbage collector to run immediately

# # 3. Load the snapshot and plot image - w
# output_filename = filehead * "subdomain_w_" * fileparam * "_iter$(iteration).jld2"
# snapshot = load_subdomain_snapshot(output_filename)
# plot_image(snapshot, :w, fileparam)
# println("Freeing large variables...")
# snapshot = nothing      # free the variable
# GC.gc()                 # force the garbage collector to run immediately

### -------------------------------------------------------------------------