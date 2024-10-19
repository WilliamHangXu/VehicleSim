using StaticArrays
using AStarSearch

include("../map.jl")

export training_map

map_segments = training_map()

# manhattan distance between positions in the maze matrix
manhattan(a::RoadSegment, b::RoadSegment) = sum(abs.(b.lane_boundaries[1].pt_a - a.lane_boundaries[1].pt_a))

# check out the children of each RoadSegment
function get_children(map, p)
    res = RoadSegment[]
    #println("p.children ", p.children)
    for child in p.children
        #println("map[child] ", map[child])
        push!(res, map[child])
    end
    return res
end

function a_star_solver(map, start, goal)
    current_get_children(state) = get_children(map, state)
    # Here you can use any of the exported search functions, they all share the same interface, but they won't use the heuristic and the cost
    return astar(current_get_children, start, goal, heuristic=manhattan)
end

#println("map_segments[51] ", map_segments[51])

start = map_segments[5]
goal = map_segments[102]

res = a_star_solver(map_segments, start, goal)

####### print out the whole path from start point to end point ######

# for i in res.path
#     print(i.id)
#     println(i.children)
# end