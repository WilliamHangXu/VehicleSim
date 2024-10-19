module VehicleSim

using ColorTypes
using Dates
using GeometryBasics
using MeshCat
using MeshCatMechanisms
using Random
using Rotations
using RigidBodyDynamics
using Infiltrator
using LinearAlgebra
using SparseArrays
using Suppressor
using Sockets
using Serialization
using StaticArrays
using Ipopt
using Symbolics 
using GLMakie
using ProgressMeter
using LazySets
using AutomotiveDrivingModels

include("view_car.jl")
include("objects.jl")
include("sim.jl")
include("client.jl")
include("control.jl")
include("sink.jl")
include("measurements.jl")
include("map.jl")
include("example_project.jl")
include("ekf_local.jl")

export server, shutdown!, keyboard_client, my_client, get_lane_half_space, if_in_segments, get_mid_half_space_left, get_mid_half_space_right, collision_checker, test_check_collision

end
