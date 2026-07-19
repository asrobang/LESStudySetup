using LESStudySetup
using CairoMakie, Makie
using Printf, Dates, StatsBase
using Statistics: mean, std, quantile
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_subdomain_snapshot
using MathTeXEngine
set_theme!(theme_latexfonts(), fontsize=12, figure_padding = 10)
using JLD2 #, CUDA

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "figures/20260708_nhy_frontanalysis/tiles/"

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
iteration = 62484
println("Iteration: $(iteration)")

max_T = 0.0
min_T = 1000.0

for i in 1:100
    println("--- Subdomain $(i) ---")

    # 2. Define the filename of the saved snapshot
    fileparam = "subdomain" * string(i)
    output_filename = filehead * fileparam * "_iter$(iteration).jld2"

    # 3. Load the snapshot
    snapshot = load_subdomain_snapshot(output_filename)

    # update max and min T
    curr_max = maximum(snapshot[:T])
    curr_min = minimum(snapshot[:T])
    global max_T = max(max_T, curr_max)
    global min_T = min(min_T, curr_min)
    println("Max T so far: $(max_T)")
    println("Min T so far: $(min_T)")

    # 4. Plot figure
    plot_T_image(snapshot, fileparam; Tmin=19.5, Tmax=20.1)
    # plot_image(snapshot, :u, fileparam; k=70)
    # plot_image(snapshot, :v, fileparam; k=70)
    # plot_image(snapshot, :w, fileparam; k=70)
    # plot_image(snapshot, :vort, fileparam; k=70)
    # plot_image(snapshot, :vortf, fileparam; k=70)
    # plot_image(snapshot, :hke, fileparam; k=70)
    # plot_w(snapshot, fileparam)
end

println("FINAL Max T: $(max_T)")
println("FINAL Min T: $(min_T)")

# plot_T_colorbar(; Tmin=19.5, Tmax=20.1)
# plot_colorbar(:u)
# plot_colorbar(:v)
# plot_colorbar(:w)
# plot_colorbar(:vort)
# plot_colorbar(:vortf)
# plot_colorbar(:hke)

