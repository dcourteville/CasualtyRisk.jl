using OrbitalElementModels, CasualtyRisk
using Test, Dates, StaticArrays, SatelliteToolboxBase, SatelliteToolboxAtmosphericModels, LinearAlgebra

# Build dynamics model
ref_date = DateTime(2025)
body = earth(ref_date)
jb0 = datetime2julian(ref_date)
zonal = ZonalAcc(4)
sol_acti = HistoricSolActi(jb0)
drag = Drag(2.25 * 15.54, Thermonets(sol_acti, body))
SpaceIndices.init(SpaceIndices.Celestrak)
elem = MCE()
model = ReentryModel(elem, body, zonal, drag)

# Population data
pop_data = load_gpwv4("POP_100_2024_GPW_V4_UN_ext.asc")
@test pop_data.ncols == 360
@test pop_data.nrows == 145
@test pop_data.xllcorner == -180
@test pop_data.yllcorner == -59.999999999992
@test pop_data.cellsize == 1.0
@test size(pop_data.grid) == (pop_data.nrows, pop_data.ncols)
cell_range = pop_data.grid[1][1]
@test rad2deg(cell_range[2] - cell_range[1]) ≈ pop_data.cellsize
@test rad2deg(cell_range[4] - cell_range[3]) ≈ pop_data.cellsize

# Casualty risk
X = SA[500e3, 120e3, deg2rad(0), deg2rad(98), deg2rad(94.4), 0.0, 750.0, 0.0, 0.0]
Σ = Diagonal(SA[10., 10., deg2rad(0.1), deg2rad(0.001), deg2rad(0.1), deg2rad(0.001), 0.1, 60., 0.01])

Rut = compute_casualty_risk(model, X, Σ, pop_data)
Rmc = mc_casualty_risk(model, X, Σ, pop_data; nsamples=10000)
@test Rut ≈ Rmc rtol=0.1