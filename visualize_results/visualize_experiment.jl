# AUTHOR: Shirui Peng
# DESCRIPTION: 
# This script loads snapshot data from LES simulations and creates visualizations
# of temperature, velocity fields, and diagnostics like mixed layer depth (MLD).

# ============================================================================
# IMPORT REQUIRED PACKAGES
# ============================================================================
using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using Statistics: mean
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_snapshots, MLD 
set_theme!(Theme(fontsize = 12))

# ============================================================================
# SIMULATION PARAMETERS AND FILE PATHS
# ============================================================================
# Set simulation parameters (cooling rate, wind stress, surface temperature flux, etc.)
cooling, wind, dTf,a = 50, 0.1, -1,1.0
# Examples! (fill in the correct filename and metadata filename)
# cooling, wind, dTf = 25, 0.02, -1

# Format parameters as strings for filename construction
cooling = @sprintf("%03d", cooling)
wind = replace("$(wind)","." => "" )
a = replace("$(a)","." => "" )

# Construct filename based on whether free surface is included
if dTf < 0
    fileparams = "hydrostatic_twin_simulation"
else
    if length(wind) < 2
        wind = "0" * wind
    end
    dTf = @sprintf("%1d", dTf)
    fileparams = "free_surface_short_test_$(cooling)_wind_$(wind)_dTf_$(dTf)_a_$(a)"
end

# Define file paths
filehead = "./LESStudySetup/"
filename = filehead * "hydrostatic_snapshots_" * fileparams * ".jld2"
metadata = filehead * "experiment_" * fileparams * "_metadata.jld2"
filesave = "/home/asrobang/orcd/scratch/figures/"

# ============================================================================
# LOAD SIMULATION DATA
# ============================================================================
# load all the data!!
println("Loading data from $filename...")
snapshots = load_snapshots(filename; metadata)

# ============================================================================
# SELECT SNAPSHOT TO VISUALIZE
# ============================================================================
times = snapshots[:T].times
snapshot_number = length(times)÷2 + 48      # Snapshot 208 out of 321 (Day 13), why??
nday = @sprintf("%2.0f", (times[snapshot_number])/60^2/24)      # Technically, this is Day 12.875103
println("Plotting snapshot $snapshot_number on day $(nday)...")
t0 = now()

# ============================================================================
# EXTRACT FIELDS FROM SNAPSHOT
# ============================================================================
# Extract temperature and velocity components; compute mixed layer depth
T = snapshots[:T][snapshot_number]
u = snapshots[:u][snapshot_number]
v = snapshots[:v][snapshot_number]
w = snapshots[:w][snapshot_number]
h = compute!(MLD(snapshots,snapshot_number; threshold = 0.03))
println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")
println("mean MLD: $(mean(interior(h)))m")

# Reference to the computational grid
grid = T.grid

# ============================================================================
# EXTRACT COORDINATE ARRAYS FOR EACH FIELD
# ============================================================================
# Each field is staggered on different points of the Arakawa C-grid
# Coordinate arrays
xu, yu, zu = nodes(u)
xv, yv, zv = nodes(v)
xw, yw, zw = nodes(w)
xT, yT, zT = nodes(T)

# ============================================================================
# HORIZONTAL SLICE VISUALIZATION (Top-down view)
# ============================================================================
# Define depths and transect locations for slicing
kT, ks, kw = 116, 127, 116                    # Depth indices for slicing
jslices = [1, 2, 3] * 125                     # Transect locations in y-direction
Tmin, Tmax = minimum(interior(T)), maximum(interior(T))  # Temperature range for colormap
vbnd, wbnd = maximum(abs, interior(v)), 0.005            # Bounds for velocity colormaps

# Create figure with 3 subplots (Temperature, Meridional velocity, Vertical velocity)
fig = Figure(size = (800, 250))
axis_kwargs = (xlabel = "x (km)", ylabel = "y (km)",
            limits = ((0, 100), (0, 100)))
axis_kwargs2 = NamedTuple{(:xlabel,:limits)}(axis_kwargs)

# Define axes
ax_T = Axis(fig[1, 1][1,1]; title="Temperature (ᵒC), z=$(zT[kT])m", axis_kwargs...)
#ax_s = Axis(fig[1, 2][1,1]; title="Horizontal speed (m s⁻¹)", axis_kwargs2...)
ax_v = Axis(fig[1, 2][1,1]; title="Meridional velocity (m s⁻¹), z=$(zv[ks])m", axis_kwargs2...)
ax_w = Axis(fig[1, 3][1,1]; title="Vertical velocity (m s⁻¹), z=$(zw[kw])m", axis_kwargs2...)    

# Plot temperature (T) heatmap
hm_T = heatmap!(ax_T, 1e-3xT, 1e-3yT, interior(T,:,:,kT); rasterize = true, colormap = :thermal, colorrange = (Tmin, Tmax))
Colorbar(fig[1, 1][1, 2], hm_T)

#= 
# Plot horizontal speed (s) heatmap
hm_s = heatmap!(ax_s, 1e-3xs, 1e-3ys, interior(s,:,:,ks); colormap = :speed, colorrange = (0, 0.85))
hideydecorations!(ax_s, ticks = false)
Colorbar(fig[1, 2][1, 2], hm_s) 
=#

# Plot meridional velocity (v) heatmap
hm_v = heatmap!(ax_v, 1e-3xv, 1e-3yv, interior(v,:,:,ks); rasterize = true, colormap = :balance, colorrange = (-vbnd, vbnd))
hideydecorations!(ax_v, ticks = false)
Colorbar(fig[1, 2][1, 2], hm_v)

# Plot vertical velocity (w) heatmap
hm_w = heatmap!(ax_w, 1e-3xw, 1e-3yw, interior(w,:,:,kw); rasterize = true, colormap = :balance, colorrange = (-wbnd, wbnd))#, highclip = :red, lowclip = :blue)
hideydecorations!(ax_w, ticks = false)
Colorbar(fig[1, 3][1, 2], hm_w)

# Add horizontal lines to mark transect locations
hlines!(ax_T, 1e-3*yT[jslices]; color = :black, linewidth = 0.5)
hlines!(ax_v, 1e-3*yv[jslices]; color = :black, linewidth = 0.5)
hlines!(ax_w, 1e-3*yw[jslices]; color = :black, linewidth = 0.5)

# Save the horizontal slice figure
save(filesave * "fields_" * fileparams * "_d$(nday).pdf", fig)
println("Finished plotting fields, wall time: $((now() - t0).value/1e3) seconds.")

# ============================================================================
# VERTICAL CROSS-SECTION VISUALIZATION (Side view along x-direction)
# ============================================================================
# Define vertical axis limits and plot parameters
zmin = -250
axis_kwargs0 = (xlabel = "x (km)",
                ylabel = "z (m)",
                limits = ((0, 100), (zmin, 0)))
axis_kwargs1 = NamedTuple{(:xlabel,:limits)}(axis_kwargs0)
axis_kwargs2 = NamedTuple{(:ylabel,:limits)}(axis_kwargs0)
#axis_kwargs3 = NamedTuple{(:limits)}(axis_kwargs0)

# Create figure with 3 rows (one for each transect location)
fig = Figure(size = (800, 400))

# Loop over three transect locations
for j in [1,2,3]
    jT, jv, jw = 125*(4-j), 125*(4-j), 125*(4-j)

    # Set up axes with appropriate labels based on position in grid
    if j == 3
        local ax_T = Axis(fig[j, 1][1,1]; axis_kwargs0...)
        local ax_v = Axis(fig[j, 2][1,1]; axis_kwargs1...)
        local ax_w = Axis(fig[j, 3][1,1]; axis_kwargs1...)
    elseif j == 2
        local ax_T = Axis(fig[j, 1][1,1]; axis_kwargs2...)
        local ax_v = Axis(fig[j, 2][1,1]; limits = ((0, 100), (zmin, 0)))
        local ax_w = Axis(fig[j, 3][1,1]; limits = ((0, 100), (zmin, 0)))
    else
        local ax_T = Axis(fig[j, 1][1,1]; title="Temperature (ᵒC)", axis_kwargs2...)
        local ax_v = Axis(fig[j, 2][1,1]; title="Meridional velocity (m s⁻¹)", limits = ((0, 100), (zmin, 0)))
        local ax_w = Axis(fig[j, 3][1,1]; title="Vertical velocity (m s⁻¹)", limits = ((0, 100), (zmin, 0)))
    end
    
    # Create heatmaps for this transect
    hm_T = heatmap!(ax_T, 1e-3xT, zT, interior(T,:,jT,:); rasterize = true, colormap = :thermal, colorrange = (Tmin, Tmax))
    hm_v = heatmap!(ax_v, 1e-3xv, zv, interior(v,:,jv,:); rasterize = true, colormap = :balance, colorrange = (-vbnd, vbnd))
    hm_w = heatmap!(ax_w, 1e-3xw, zw, interior(w,:,jw,:); rasterize = true, colormap = :balance, colorrange = (-wbnd, wbnd))#, highclip = :red, lowclip = :blue)
    
    # Hide x-axis decorations for non-bottom rows
    if j < 3
        hidexdecorations!(ax_T, ticks = false)
        hidexdecorations!(ax_v, ticks = false)
        hidexdecorations!(ax_w, ticks = false)
    end

    # Hide y-axis decorations for velocity subplots
    hideydecorations!(ax_v, ticks = false)
    hideydecorations!(ax_w, ticks = false)

    # Add colorbars
    Colorbar(fig[j, 1][1, 2], hm_T)
    Colorbar(fig[j, 2][1, 2], hm_v)
    Colorbar(fig[j, 3][1, 2], hm_w)

    # Overlay depth of horizontal slice on vertical cross-section (black line)
    hlines!(ax_T, [zT[kT]]; color = :black, linewidth = 0.5)

    # Overlay mixed layer depth contour (white line)
    lines!(ax_T, xT/1e3, -interior(h,:,jT); color = :white, linewidth = 0.8)
    println(mean(-interior(h,:,jT)))

    # Overlay depth of horizontal velocity slice on vertical cross-sections
    hlines!(ax_v, [zv[ks]]; color = :black, linewidth = 0.5)
    hlines!(ax_w, [zw[kw]]; color = :black, linewidth = 0.5)
    
    # # Calculate isotherm depth
    #Tref = (interior(T,:,jT,120)+interior(T,:,jT,121))/2
    #hj = zeros(length(xT))
    #for i = 1:length(xT)
    #  hj[i] = -zT[findfirst(interior(T,i,jT,:) .>= Tref[i]-0.1)]
    #end
    #lines!(ax_T, xT/1e3, -hj; color = :white, linewidth = 0.8, linestyle = :dash)
    #println(mean(hj))
end

# Save the vertical cross-section figure
save(filesave * "fieldslices_" * fileparams * "_d$(nday).pdf", fig)
println("Finished plotting fieldslices, wall time: $((now() - t0).value/1e3) seconds.")

#=
# ============================================================================
# EDDY FLUX ANALYSIS
# ============================================================================
# This section computes eddy fluxes by separating fields into mean and 
# fluctuating components. 
# Filtering the fields to find mean values
T̅ = spatial_filtering(T)
U = spatial_filtering(u)
V = spatial_filtering(v)
W = spatial_filtering(w)

# Calculate the fluctuations
T′ = compute!(Field(T - T̅))
u′ = compute!(Field(u - U))
v′ = compute!(Field(v - V))
w′ = compute!(Field(w - W))

# Compute eddy fluxes 
u′T′ = compute!(Field(u′ * T′)) 
v′T′ = compute!(Field(v′ * T′)) 
w′T′ = compute!(Field(w′ * T′))  
=#