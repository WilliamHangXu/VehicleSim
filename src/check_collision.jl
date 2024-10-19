function overlap(poly_a::SMatrix{M, 2, Float64},
                poly_b::SMatrix{N, 2, Float64}) where {M,N}
  
    if find_separating_axis(poly_a, poly_b)
        return false
    end
    if find_separating_axis(poly_b, poly_a)
        return false
    end

    return true

end

""" 
    find_separating_axis(poly_a::SMatrix{4, 2, Float64}, poly_b::SMatrix{4, 2, Float64})
build a list of candidate separating axes from the edges of a
  /!\\ edges needs to be ordered
"""
function find_separating_axis(poly_a::SMatrix{M, 2, Float64},
                              poly_b::SMatrix{N, 2, Float64}) where {M, N}
    n_a = size(poly_a)[1]
    n_b = size(poly_b)[1]
    axis = zeros(2)
    @views previous_vertex = poly_a[end,:]
    for i=1:n_a # loop through all the edges n edges
        @views current_vertex = poly_a[i,:]
        # get edge vector
        edge = current_vertex - previous_vertex

        # rotate 90 degrees to get the separating axis
        axis[1] = edge[2]
        axis[2] = -edge[1]

        #  project polygons onto the axis
        a_min,a_max = polygon_projection(poly_a, axis)
        b_min,b_max = polygon_projection(poly_b, axis)

        # check separation
        if a_max < b_min
            return true#,current_vertex,previous_vertex,edge,a_max,b_min

        end
        if b_max < a_min
            return true#,current_vertex,previous_vertex,edge,a_max,b_min

        end

        @views previous_vertex = poly_a[i,:]
    end

    # no separation was found
    return false
end

"""
    polygon_projection(poly::SMatrix{4, 2, Float64}, axis::Vector{Float64})
return the projection interval for the polygon poly over the axis axis 
"""
function polygon_projection(poly::SMatrix{N, 2, Float64},
                            axis::Vector{Float64}) where {N}
        n_a = size(poly)[1]
        @inbounds @fastmath @views d1 = dot(poly[1,:],axis)
        @inbounds @fastmath @views d2 = dot(poly[2,:],axis)
        # initialize min and max
        if d1<d2
            out_min = d1
            out_max = d2
        else
            out_min = d2
            out_max = d1
        end

        for i=1:n_a
            @inbounds @fastmath @views d = dot(poly[i,:],axis)
            if d < out_min
                out_min = d
            elseif d > out_max
                out_max = d
            end
        end
        return out_min,out_max
end