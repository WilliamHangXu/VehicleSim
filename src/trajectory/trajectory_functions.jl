using("../measurement.jl")

"""
Create functions which accepts X¹, X², X³, r¹, r², r³, track_center, track_radius, lane_width, as input, and each return
one of the 5 callbacks which constitute an IPOPT problem: 
1. eval_f
2. eval_g
3. eval_grad_f
4. eval_jac_g
5. eval_hess_lag

Xⁱ is the vehicle_state of vehicle i at the start of the trajectory (t=0)
rⁱ is the radius of the i-th vehicle.
(track_center, track_radius, lane_width) define two circles boundaries.

The purpose of this function is to construct functions which can quickly turn 
updated world information into planning problems that IPOPT can solve.
"""
function create_callback_generator(vehicle_state, 
    latest_perception_state, 
    cur_seg)
    
    
    # trajectory_length=40, timestep=0.2, R = Diagonal([0.1, 0.5]), max_vel=10.0)
    # Define symbolic variables for all inputs, as well as trajectory
    # X¹, X², X³, r¹, r², r³, track_center, track_radius, lane_width, Z = let
    #     @variables(X¹[1:4], X²[1:4], X³[1:4], r¹, r², r³, track_center[1:2], track_radius, lane_width, Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    # end

    Z = let
         @variables(Z[1:Inf]) .|> Symbolics.scalarize
    end

    # states, controls = decompose_trajectory(Z)
    # all_states = [[X¹,]; states]
    # vehicle_2_prediction = constant_velocity_prediction(X², trajectory_length, timestep)
    # vehicle_3_prediction = constant_velocity_prediction(X³, trajectory_length, timestep)

    # cost_val = sum(stage_cost(x, u, R) for (x,u) in zip(states, controls))
    cost_val = 0
    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]
    curvature = seg.lane_boundaries[1].curvature
    curved = !isapprox(curvature, 0.0; atol=1e-6)
    while true
        # append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
        # append!(constraints_lb, zeros(4))
        # append!(constraints_ub, zeros(4))
        

        if curved
            append!(constraints_val, lane_constraint_curve_lower(vehicle_state, cur_seg))
            append!(constraints_val, lane_constraint_curve_upper(vehicle_state, cur_seg))
            append!(constraints_lb, [0.0, -Inf])
            append!(constraints_ub, [Inf, 0.0])
        else
            append!(constraints_val, lane_constraint_lower(vehicle_state, cur_seg))
            append!(constraints_val, lane_constraint_upper(vehicle_state, cur_seg))
            append!(constraints_lb, [0.0, -Inf])
            append!(constraints_ub, [Inf, 0.0])
        end
        
        
        for i in 1:length(latest_perception_state.x)
            other_vehicle = latest_perception_state.x[i]
            append!(constraints_val, collision_constraint(vehicle_state, other_vehicle, ϵ))
            append!(constraints_lb, 0)
            append!(constraints_ub, Inf)
        end
        
        vel = norm(vehicle_state.velocity)
        append!(constraints_val, vel)
        append!(constraints_lb, 0.0)
        append!(constraints_ub, cur_seg.speed_limit)
        
        #append!(constraints_val, states[k][4])
        #append!(constraints_lb, -pi/4)
        #append!(constraints_ub, pi/4)
    end

    constraints_jac = Symbolics.sparsejacobian(constraints_val, Z)
    (jac_rows, jac_cols, jac_vals) = findnz(constraints_jac)
    num_constraints = length(constraints_val)

    λ, cost_scaling = let
        @variables(λ[1:num_constraints], cost_scaling) .|> Symbolics.scalarize
    end

    lag = (cost_scaling * cost_val + λ' * constraints_val)
    lag_grad = Symbolics.gradient(lag, Z)
    lag_hess = Symbolics.sparsejacobian(lag_grad, Z)
    (hess_rows, hess_cols, hess_vals) = findnz(lag_hess)
    
    expression = Val{false}

    full_cost_fn = let
        cost_fn = Symbolics.build_function(cost_val, [Z; vehicle_state; latest_perception_state; cur_seg]; expression)
        (Z, vehicle_state, latest_perception_state, cur_seg) -> cost_fn([Z; vehicle_state; latest_perception_state; cur_seg])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [Z; vehicle_state; latest_perception_state; cur_seg]; expression)[2]
        (grad, Z, vehicle_state, latest_perception_state, cur_seg) -> cost_grad_fn!(grad, [Z; vehicle_state; latest_perception_state; cur_seg])
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [Z; vehicle_state; latest_perception_state; cur_seg]; expression)[2]
        (cons, Z, vehicle_state, latest_perception_state, cur_seg) -> constraint_fn!(cons, [Z; vehicle_state; latest_perception_state; cur_seg])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(constraints_val, [Z; vehicle_state; latest_perception_state; cur_seg]; expression)[2]
        (cons, Z, vehicle_state, latest_perception_state, cur_seg) -> constraint_jac_vals_fn!(cons, [Z; localization_state_channel; perception_state_channel; cur_seg; socket])
    end
    
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [Z; vehicle_state; latest_perception_state; cur_seg; λ; cost_scaling]; expression)[2]
        (vals, Z,  vehicle_state, latest_perception_state, cur_seg, λ, cost_scaling) -> hess_vals_fn!(vals, [Z; vehicle_state; latest_perception_state; cur_seg; λ; cost_scaling])
    end

    full_constraint_jac_triplet = (; jac_rows, jac_cols, full_constraint_jac_vals_fn)
    full_lag_hess_triplet = (; hess_rows, hess_cols, full_hess_vals_fn)
    
    return (; full_cost_fn, 
            full_cost_grad_fn, 
            full_constraint_fn, 
            full_constraint_jac_triplet, 
            full_lag_hess_triplet,
            constraints_lb,
            constraints_ub)
end

"""
Predict a dummy trajectory for other vehicles.
"""
# function constant_velocity_prediction(X0, trajectory_length, timestep)
#     X = X0
#     U = zeros(2)
#     states = []
#     for k = 1:trajectory_length
#         X = evolve_state(X, U, timestep)
#         push!(states, X)
#     end
#     states
# end

# """
# The physics model used for motion planning purposes.
# Returns X[k] when inputs are X[k-1] and U[k]. 
# Uses a slightly different vehicle model than presented in class for technical reasons.
# """
# function evolve_state(X, U, Δ)
#     V = X[3] + Δ * U[1] 
#     θ = X[4] + Δ * U[2]
#     X + Δ * [V*cos(θ), V*sin(θ), U[1], U[2]]
# end

function lane_constraint_curve_lower(vehicle_state, seg)
    #a'*(X[1:2] - a*r)-b
    ego_position = vehicle_state.position[1:2]
    lb_1 = seg.lane_boundaries[1]
    if length(seg.lane_boundaries) == 2
        lb_2 = seg.lane_boundaries[2]
    else
        lb_2 = seg.lane_boundaries[3]
    end

    pt_a = lb_1.pt_a
    pt_b = lb_1.pt_b
    pt_c = lb_2.pt_a
    pt_d = lb_2.pt_b

    curvature = lb_1.curvature
    
    rad = 1.0 / abs(curvature)
    dist = π*rad/2.0
    left = curvature > 0

    rad_1 = rad
    rad_2 = abs(pt_d[1]-pt_c[1])

    if left
        if sign(delta[1]) == sign(delta[2])
            center = pt_a + [0, delta[2]]
        else
            center = pt_a + [delta[1], 0]
        end
    else
        if sign(delta[1]) == sign(delta[2])
            center = pt_a + [delta[1], 0]
        else
            center = pt_a + [0, delta[2]]
        end
    end
    if rad_1 < rad_2
        min = rad_1
    else
        min = rad_2
    end
    (ego_position[1:2]-center[1:2])'*(ego_position[1:2]-center[1:2]) - (min+14.3781)^2
end


function lane_constraint_curve_upper(vehicle_state, seg)
    ego_position = vehicle_state.position[1:2]
    lb_1 = seg.lane_boundaries[1]
    if length(seg.lane_boundaries) == 2
        lb_2 = seg.lane_boundaries[2]
    else
        lb_2 = seg.lane_boundaries[3]
    end

    pt_a = lb_1.pt_a
    pt_b = lb_1.pt_b
    pt_c = lb_2.pt_a
    pt_d = lb_2.pt_b

    curvature = lb_1.curvature
    
    rad = 1.0 / abs(curvature)
    dist = π*rad/2.0
    left = curvature > 0

    rad_1 = rad
    rad_2 = abs(pt_d[1]-pt_c[1])

    if left
        if sign(delta[1]) == sign(delta[2])
            center = pt_a + [0, delta[2]]
        else
            center = pt_a + [delta[1], 0]
        end
    else
        if sign(delta[1]) == sign(delta[2])
            center = pt_a + [delta[1], 0]
        else
            center = pt_a + [0, delta[2]]
        end
    end
    if rad_1 > rad_2
        max = rad_1
    else
        max = rad_2
    end
    (ego_position[1:2]-center[1:2])'*(ego_position[1:2]-center[1:2]) - (max-14.3781)^2
end

function lane_constraint_lower(vehicle_state, seg)
    ego_position = vehicle_state.position[1:2]
    lb_1 = seg.lane_boundaries[1]

    pt_a = lb_1.pt_a
    b = -5.0

    pt_a'*(ego_position - pt_a*14.3781)-b
end

function lane_constraint_upper(vehicle_state, seg)
    ego_position = vehicle_state.position[1:2]
    if length(seg.lane_boundaries) == 2
        lb_2 = seg.lane_boundaries[2]
    else
        lb_2 = seg.lane_boundaries[3]
    end

    pt_a = lb_2.pt_a
    b = -5.0

    pt_a'*(ego_position - pt_a*14.3781)-b
end



"""
ϵ defines the min distance between two vehicles.
"""
function collision_constraint(X1::FullVehicleState, X2::SimpleVehicleState, ϵ = 0.5)
    I = [1 0; 0 1]
    P = [I -I; -I I]
    q = zeros(4) 
    quat = X1.orientation
    A1 = inv(Rot_from_quat(quat))
    A2 = inv([cos(X2.θ) -sin(X2.θ); sin(X2.θ) cos(X2.θ)])
    A = [A1 zeros(2,2);zeros(2,2) A2]
    l1 = [-13.2*0.5; -5.7*0.5] + A1*[X1.position[1]; X1.position[2]]
    l2 = [-X2.l*0.5; -X2.w*0.5] + A2*[X2.p1; X2.p2]
    u1 = [13.2*0.5; 5.7*0.5] + A1*[X1.position[1]; X1.position[2]]
    u2 = [X2.l*0.5; X2.w*0.5] + A2*[X2.p1; X2.p2]
    l = [l1;l2]
    u = [u1;u2]
    m = OSQP.Model()
    OSQP.setup!(m; P=sparse(P), q=q, A=sparse(A), l=l, u=u, polish = true)
    results = OSQP.solve!(m)
    results.info.obj_val - ϵ 
end

# """
# Cost at each stage of the plan
# """
# function stage_cost(X, U, R)
#     cost = -0.1*X[3] + U'*R*U
# end


"""
Assume z = [U[1];...;U[K];X[1];...;X[K]]
Return states = [X[1], X[2],..., X[K]], controls = [U[1],...,U[K]]
where K = trajectory_length
"""
function decompose_trajectory(z)
    K = Int(length(z) / 6)
    controls = [@view(z[(k-1)*2+1:k*2]) for k = 1:K]
    states = [@view(z[2K+(k-1)*4+1:2K+k*4]) for k = 1:K]
    return states, controls
end

function compose_trajectory(states, controls)
    K = length(states)
    z = [reduce(vcat, controls); reduce(vcat, states)]
end


