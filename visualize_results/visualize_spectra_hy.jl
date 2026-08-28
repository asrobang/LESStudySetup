using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_snapshots, isotropic_powerspectrum, δ
set_theme!(Theme(fontsize = 12))

cooling, wind, dTf,a = 50, 0.1, -1,1.0
# Examples! (fill in the correct filename and metadata filename)
# cooling, wind, dTf = 25, 0.02, -1
cooling = @sprintf("%03d", cooling)
wind = replace("$(wind)","." => "" )
a = replace("$(a)","." => "" )
if dTf < 0
    fileparams = "hydrostatic_twin_simulation"
else
    if length(wind) < 2
        wind = "0" * wind
    end
    dTf = @sprintf("%1d", dTf)
    fileparams = "free_surface_short_test_$(cooling)_wind_$(wind)_dTf_$(dTf)_a_$(a)"
end
filehead = "./LESStudySetup/"
filename = filehead * "hydrostatic_snapshots_" * fileparams * ".jld2"
metadata = filehead * "experiment_" * fileparams * "_metadata.jld2"
filesave = "./figures/20260717_hy_spectra/"

# --- Set parameters ---
set_value!(; Δh = 156.25)    # horizontal spacing
f = parameters.f;               # Coriolis parameter
Q = 40                          # surface sensible heat flux? 
h₀ = 60                         # height of the convective BL or mixing depth? 
ρ₀ = parameters.ρ₀              # reference density
cₚ = parameters.cp              # heat capacity
α = parameters.α                # thermal expansion
g = parameters.g                # gravity
wₛ = (α * g * Q * h₀ / (ρ₀ * cₚ))^(1/3)     # convective velocity scale [m/s]?
dx = 156.25                    # [m] horizontal grid size
dy = 156.25                    # [m] horizontal grid size
dz = 1.125                      # [m] vertical grid size

# Double check grid spacing
# xu, yu, zu = nodes(snapshots[:u])
# Δx = xu[2] - xu[1]   # spacing between adjacent x-nodes
# Δy = yu[2] - yu[1]
# xT, yT, zT = nodes(snapshots[:T])   # or nodes(snapshots[:w]) for face-centered z
# Δz = zT[2]-zT[1] # diff(zT)                       # vector of spacings between consecutive z-levels

# load all the data!!
println("Loading data from $filename...")
snapshots = load_snapshots(filename; metadata)

# ### -------------------------------------------------------------------------

## Computes the isotropic power spectrums of u, v, w, T from the hydrostatic simulation output
function hyspectrum_uvwT(snapshots, snapshot_number, klev)

    times = snapshots[:T].times
    nday = @sprintf("%2.1f", (times[snapshot_number])/60^2/24)
    println("Reading snapshot $snapshot_number on day $(nday)...")

    T = snapshots[:T][snapshot_number]
    u = snapshots[:u][snapshot_number]
    v = snapshots[:v][snapshot_number]
    w = snapshots[:w][snapshot_number]

    # Coordinate arrays
    xu, yu, zu = nodes(u)       # u at cell faces in x
    xv, yv, zv = nodes(v)       # v at cell faces in y
    xw, yw, zw = nodes(w)       # w at cell faces in z
    xT, yT, zT = nodes(T)       # T at cell centers 

    # # Compute the auto-spectrum (co-spectrum of field with itself) of T, u, v, w 

    # Full domain 
    # Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev); Δx=dx, Δy=dy)
    # Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev); Δx=dx, Δy=dy)
    # wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2      # interpolates between neighboring cells to get depth of cell center
    # Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
    # St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)

    # # Sliced to match nhy subdomain 91
    # Su = isotropic_powerspectrum(interior(u, 583:640, 7:71, klev), interior(u, 583:640, 7:71, klev); Δx=dx, Δy=dy)
    # Sv = isotropic_powerspectrum(interior(v, 583:640, 7:71, klev), interior(v, 583:640, 7:71, klev); Δx=dx, Δy=dy)
    # wk = (interior(w, 583:640, 7:71, klev)+interior(w, 583:640, 7:71, klev+1))/2
    # Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
    # St = isotropic_powerspectrum(interior(T, 583:640, 7:71, klev), interior(T, 583:640, 7:71, klev); Δx=dx, Δy=dy)

    # # Sliced to match nhy subdomain 97
    # xrange = vcat(583:640, 1:7)
    # Su = isotropic_powerspectrum(interior(u, xrange, 391:455, klev), interior(u, xrange, 391:455, klev); Δx=dx, Δy=dy)
    # Sv = isotropic_powerspectrum(interior(v, xrange, 391:455, klev), interior(v, xrange, 391:455, klev); Δx=dx, Δy=dy)
    # wk = (interior(w, xrange, 391:455, klev)+interior(w, xrange, 391:455, klev+1))/2
    # Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
    # St = isotropic_powerspectrum(interior(T, xrange, 391:455, klev), interior(T, xrange, 391:455, klev); Δx=dx, Δy=dy)

    name = "C"

    if name == "A"
        xL, xR = 352, 608
        yB, yT = 192, 448
    elseif name == "B"
        xL, xR = 512, 128
        yB, yT = 480, 96
    elseif name == "C"
        xL, xR = 64, 320
        yB, yT = 192, 448
    end

    if xL > xR
        xrange = vcat(xL:640, 1:xR)
    else
        xrange = xL:xR
    end

    if yB > yT
        yrange = vcat(yB:640, 1:yT)
    else
        yrange = yB:yT
    end

    Su = isotropic_powerspectrum(interior(u, xrange, 391:455, klev), interior(u, xrange, 391:455, klev); Δx=dx, Δy=dy)
    Sv = isotropic_powerspectrum(interior(v, xrange, 391:455, klev), interior(v, xrange, 391:455, klev); Δx=dx, Δy=dy)
    wk = (interior(w, xrange, 391:455, klev)+interior(w, xrange, 391:455, klev+1))/2
    Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
    St = isotropic_powerspectrum(interior(T, xrange, 391:455, klev), interior(T, xrange, 391:455, klev); Δx=dx, Δy=dy)

    return St, Su, Sv, Sw
end

# ### -------------------------------------------------------------------------
# ### Figure 1: spectra of E_T, E_u, E_v, E_w

# # Let's pick the last snapshot!
# times = snapshots[:T].times
# # snapshot_number = length(times)÷2 + 48
# snapshot_number = 121   # corresponds to day 7.5
# nday = @sprintf("%2.1f", (times[snapshot_number])/60^2/24)
# println("Plotting snapshot $snapshot_number on day $(nday)...")
# t0 = now()

# T = snapshots[:T][snapshot_number]
# u = snapshots[:u][snapshot_number]
# v = snapshots[:v][snapshot_number]
# w = snapshots[:w][snapshot_number]

# ro = compute!(Field(ζ(snapshots, snapshot_number)/f))
# rd = compute!(Field(δ(snapshots, snapshot_number)/f))

# println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")

# # Coordinate arrays
# xu, yu, zu = nodes(u)
# xv, yv, zv = nodes(v)
# xw, yw, zw = nodes(w)
# xT, yT, zT = nodes(T)

# # Compute the horizontal spectrum of T, u, v, w 
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)", 
#                 ylabel = L"E_i(k)/E_{T,v}(k_{min},z=-8.4375~m)",
#                 xscale = log10, yscale = log10,
#                 limits = ((10^-3.5, 10^-1.5), (1e-10,1e3)))
# axis_kwargs2 = NamedTuple{(:xlabel,:xscale,:yscale,:limits)}(axis_kwargs1)

# fig = Figure(size = (800, 300))
# # zT[223] -8.4375
# # zT[191] -44.4375
# # zT[159] -80.4375
# for (i,klev) in enumerate([223, 191, 159])
#     println("Plotting spectra at z = $(zT[klev])m...")
#     if i == 1
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs1...)
#     else
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs2...)
#         hideydecorations!(ax, ticks = false)
#     end

#     # Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev); Δx=dx, Δy=dy)
#     # Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev); Δx=dx, Δy=dy)
#     # wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2
#     # Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
#     # St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)

#     # Sliced to match nhy subdomain 91
#     Su = isotropic_powerspectrum(interior(u, 583:640, 7:71, klev), interior(u, 583:640, 7:71, klev); Δx=dx, Δy=dy)
#     Sv = isotropic_powerspectrum(interior(v, 583:640, 7:71, klev), interior(v, 583:640, 7:71, klev); Δx=dx, Δy=dy)
#     wk = (interior(w, 583:640, 7:71, klev)+interior(w, 583:640, 7:71, klev+1))/2
#     Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
#     St = isotropic_powerspectrum(interior(T, 583:640, 7:71, klev), interior(T, 583:640, 7:71, klev); Δx=dx, Δy=dy)

#     if i == 1
#         global Sv0,St0 = Sv,St
#     end

#     lines!(ax, Su.freq, 1e-5Su.freq.^-2, linestyle = :dash, color = :black)
#     text!(ax, 10^-2.5, 1e1; text = L"k^{-2}")
#     lines!(ax, Su.freq, 1e-9Su.freq.^-3, linestyle = :dash, color = :gray)
#     text!(ax, 10^-3, 1e-2; text = L"k^{-3}")
#     lines!(ax, St.freq, Real.(St.spec./St0.spec[1]), color = :red, label = L"E_T")
#     lines!(ax, Su.freq, Real.(Su.spec./Sv0.spec[1]), color = :blue, label = L"E_u")
#     lines!(ax, Sv.freq, Real.(Sv.spec./Sv0.spec[1]), color = :green, label = L"E_v")
#     lines!(ax, Sw.freq, Real.(Sw.spec./Sv0.spec[1]), color = :black, label = L"E_w")
#     xlims!(ax, (10^-3.5, 10^-1.5))
#     vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
#     axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))
# end
# save(filesave * "spectra_" * fileparams * "_d$(replace(nday, "." => "-")).pdf", fig)
# println("Finished plotting spectra, wall time: $((now() - t0).value/1e3) seconds.")

# ### -------------------------------------------------------------------------
# ### Figure 2: spectra of E_{zeta/f}, E_{delta/f}

# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)", 
#                 ylabel = L"E_i(k)/E_{\zeta/f, \delta/f}(k_{min},z=-3~m)",
#                 xscale = log10, yscale = log10,
#                 limits = ((8e-5, 4e-2), (1e-1,1e4)))
# axis_kwargs2 = NamedTuple{(:xlabel,:xscale,:yscale,:limits)}(axis_kwargs1)
# fig = Figure(size = (800, 300))
# for (i,klev) in enumerate([127, 116, 98])
#     println("Plotting spectra at z = $(zT[klev])m...")
#     if i == 1
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs1...)
#     else
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs2...)
#         hideydecorations!(ax, ticks = false)
#     end

#     So = isotropic_powerspectrum(interior(ro, :, :, klev), interior(ro, :, :, klev); Δx=dx, Δy=dy)
#     Sd = isotropic_powerspectrum(interior(rd, :, :, klev), interior(rd, :, :, klev); Δx=dx, Δy=dy)
#     if i == 1
#         global So0,Sd0 = So,Sd
#     end

#     lines!(ax, So.freq, 10So.freq.^(1/3), linestyle = :dash, color = :black)
#     text!(ax, 10^-3.5, 0.2; text = L"k^{1/3}")
#     lines!(ax, So.freq, 1e3So.freq.^(2/3), linestyle = :dash, color = :gray)
#     text!(ax, 10^-2, 20; text = L"k^{2/3}")
#     lines!(ax, So.freq, Real.(So.spec./So0.spec[1]), color = :blue, label = L"E_{\zeta/f}")
#     lines!(ax, Sd.freq, Real.(Sd.spec./Sd0.spec[1]), color = :green, label = L"E_{\delta/f}")
#     xlims!(ax, (8e-5, 4e-2))
#     vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
#     axislegend(ax, labelsize=10, patchsize = (20, 5))
# end
# save(filesave * "spectra2_" * fileparams * "_d$(nday).pdf", fig)
# println("Finished plotting spectra, wall time: $((now() - t0).value/1e3) seconds.")


### -------------------------------------------------------------------------
### Figure 3: spectra of E_T across days

klev = 223       # corresponds to depth = -8.4375 m
i = 97
# fileparam = "subdomain" * string(i)
# fileparam = "fulldomain"
fileparam = "regionC"

St_d05, Su_d05, Sv_d05, Sw_d05 = hyspectrum_uvwT(snapshots, 9, klev)
St_d15, Su_d15, Sv_d15, Sw_d15 = hyspectrum_uvwT(snapshots, 25, klev)
St_d25, Su_d25, Sv_d25, Sw_d25 = hyspectrum_uvwT(snapshots, 41, klev)
St_d35, Su_d35, Sv_d35, Sw_d35 = hyspectrum_uvwT(snapshots, 57, klev)
St_d45, Su_d45, Sv_d45, Sw_d45 = hyspectrum_uvwT(snapshots, 73, klev)
St_d55, Su_d55, Sv_d55, Sw_d55 = hyspectrum_uvwT(snapshots, 89, klev)
St_d65, Su_d65, Sv_d65, Sw_d65 = hyspectrum_uvwT(snapshots, 105, klev)
St_d75, Su_d75, Sv_d75, Sw_d75 = hyspectrum_uvwT(snapshots, 121, klev)

# Plot figure
fig = Figure(size = (600, 500))

# Define axes for figures
# low k: oscillates over a long spatial distance, "large-scale motions"
# high k: oscillates over a short spatial distance, "small-scale motions"
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_T(k)/E_T (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-3.5, 10^0.5), (1e-10,1e2))
                )

ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

colors = [
    :black,
    RGBf(0.902, 0.624, 0.000),   # orange
    RGBf(0.337, 0.706, 0.914),   # sky blue
    RGBf(0.000, 0.620, 0.451),   # bluish green
    RGBf(0.941, 0.894, 0.259),   # yellow
    RGBf(0.000, 0.447, 0.698),   # blue
    RGBf(0.835, 0.369, 0.000),   # vermillion
    RGBf(0.800, 0.475, 0.655),   # reddish purple
]

global St0 = St_d75

# # Draw reference k^(-2) and k^(-3) lines
# lines!(ax, Sv_d05.freq, 1e-8Sv_d05.freq.^-2, linestyle = :dash, color = :black)
# text!(ax, 1e-3, 1e-2; text = L"k^{-2}")
# lines!(ax, Sv_d05.freq, 1e-15Sv_d05.freq.^-3, linestyle = :dash, color = :gray)
# text!(ax, 10^-3.5, 1e-5; text = L"k^{-3}")
# Draw spectra lines for E_T
lines!(ax, St_d05.freq, Real.(St_d05.spec./St0.spec[1]), color = colors[1], label = "Day 0.5")
lines!(ax, St_d15.freq, Real.(St_d15.spec./St0.spec[1]), color = colors[2], label = "Day 1.5")
lines!(ax, St_d25.freq, Real.(St_d25.spec./St0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, St_d35.freq, Real.(St_d35.spec./St0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, St_d45.freq, Real.(St_d45.spec./St0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, St_d55.freq, Real.(St_d55.spec./St0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, St_d65.freq, Real.(St_d65.spec./St0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, St_d75.freq, Real.(St_d75.spec./St0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-3.5, 10^0.5))

# Vertical line corresponding to wavenumber at wavelength = 10^4 m = 10 km (refers to a 10km spatial scale)
# Anything to the left is motions larger than 10km
# Anything to the right is motions smaller than 10km 
# 10km spatial scale is the loose boundary between submesoscale and mesoscale 
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)

axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "hyspectraT_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

# Plot figure u
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_u(k)/E_u (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-3.5, 10^0.5), (1e-10,1e2))
                )
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

global Su0 = Su_d75
lines!(ax, Su_d05.freq, Real.(Su_d05.spec./Su0.spec[1]), color = colors[1], label = "Day 0.5")
lines!(ax, Su_d15.freq, Real.(Su_d15.spec./Su0.spec[1]), color = colors[2], label = "Day 1.5")
lines!(ax, Su_d25.freq, Real.(Su_d25.spec./Su0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, Su_d35.freq, Real.(Su_d35.spec./Su0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, Su_d45.freq, Real.(Su_d45.spec./Su0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, Su_d55.freq, Real.(Su_d55.spec./Su0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, Su_d65.freq, Real.(Su_d65.spec./Su0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, Su_d75.freq, Real.(Su_d75.spec./Su0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-3.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "hyspectrau_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

# Plot figure v
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_v(k)/E_v (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-3.5, 10^0.5), (1e-10,1e2))
                )
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

global Sv0 = Sv_d75
lines!(ax, Sv_d05.freq, Real.(Sv_d05.spec./Sv0.spec[1]), color = colors[1], label = "Day 0.5")
lines!(ax, Sv_d15.freq, Real.(Sv_d15.spec./Sv0.spec[1]), color = colors[2], label = "Day 1.5")
lines!(ax, Sv_d25.freq, Real.(Sv_d25.spec./Sv0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, Sv_d35.freq, Real.(Sv_d35.spec./Sv0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, Sv_d45.freq, Real.(Sv_d45.spec./Sv0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, Sv_d55.freq, Real.(Sv_d55.spec./Sv0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, Sv_d65.freq, Real.(Sv_d65.spec./Sv0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, Sv_d75.freq, Real.(Sv_d75.spec./Sv0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-3.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "hyspectrav_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

# Plot figure w
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_w(k)/E_w (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-3.5, 10^0.5), (1e-10,1e2))
                )
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

global Sw0 = Sw_d75
lines!(ax, Sw_d05.freq, Real.(Sw_d05.spec./Sw0.spec[1]), color = colors[1], label = "Day 0.5")
lines!(ax, Sw_d15.freq, Real.(Sw_d15.spec./Sw0.spec[1]), color = colors[2], label = "Day 1.5")
lines!(ax, Sw_d25.freq, Real.(Sw_d25.spec./Sw0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, Sw_d35.freq, Real.(Sw_d35.spec./Sw0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, Sw_d45.freq, Real.(Sw_d45.spec./Sw0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, Sw_d55.freq, Real.(Sw_d55.spec./Sw0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, Sw_d65.freq, Real.(Sw_d65.spec./Sw0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, Sw_d75.freq, Real.(Sw_d75.spec./Sw0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-3.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "hyspectraw_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")
