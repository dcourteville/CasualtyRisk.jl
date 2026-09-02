module CasualtyRisk

using StaticArrays, SatelliteToolboxBase, Distributions, OnlineStats

using Accessors

include("model.jl")
include("unscented.jl")
include("gpw.jl")
include("integration.jl")
include("search.jl")
include("plots.jl")

end # module CasualtyRisk
