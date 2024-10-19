# function simulate(;
#         rng = MersenneTwister(420),
#         sim_steps = 100, 
#         timestep = 0.2, 
#         traj_length = 15,
#         R = Diagonal([0.1, 1.5]),
#         lane_width = 15, 
#         track_radius = 25,
#         num_vehicles = 7, 
#         min_r = 1.5, 
#         max_r = 3, 
#         max_vel = 12)

#     sim_records = []
#     track_center = [0.0,0.0]

#     # vehicle 1 is EGO
#     vehicles = [generate_random_vehicle(rng, lane_width, track_radius, track_center, min_r, max_r, max_vel),]
#     while length(vehicles) < num_vehicles
#         v = generate_random_vehicle(rng, lane_width, track_radius, track_center, min_r, max_r, max_vel/2)
#         if any(collision_constraint(v.state, v2.state, v.r, v2.r) < 1.0 for v2 in vehicles)
#             continue
#         else
#             push!(vehicles, v)
#         end
#     end
#     target_radii = [norm(v.state[1:2]-track_center) for v in vehicles]
#     # TODO Setup callbacks appropriately
#     callbacks = create_callback_generator(traj_length, timestep, R, max_vel)

#     @showprogress for t = 1:sim_steps
#         ego = vehicles[1]
#         dists = [Inf; [norm(v.state[1:2]-ego.state[1:2])-v.r-ego.r for v in vehicles[2:end]]]
#         closest = partialsortperm(dists, 1:2)
#         V2 = vehicles[closest[1]]
#         V3 = vehicles[closest[2]]
        
#         trajectory = (; states = repeat([ego.state,], traj_length), controls = repeat([zeros(2),], traj_length))
#         # TODO: replace with this when working
#         trajectory = generate_trajectory(ego, V2, V3, lane_width, track_radius, track_center, callbacks, timestep, traj_length)

#         push!(sim_records, (; vehicles=copy(vehicles), trajectory))
#         vehicles[1] = (; state = trajectory.states[1], r=vehicles[1].r)

#         for i in 2:num_vehicles
#             old_state = vehicles[i].state
#             control = generate_tracking_control(target_radii[i], track_center, old_state)
#             new_state = evolve_state(old_state, control, timestep)
#             vehicles[i] = (; state=new_state, vehicles[i].r)
#         end
        
#         #foreach(i->vehicles[i] = (; state = evolve_state(vehicles[i].state, zeros(2), timestep), vehicles[i].r), 2:num_vehicles)
#     end
#     visualize_simulation(sim_records, track_radius, track_center, lane_width)
# end

# function generate_tracking_control(target_radius, track_center, state)
#     cur_radius = norm(state[1:2]-track_center)
#     err = cur_radius-target_radius
#     u = [0.0, err]
# end

# """
# Generate a random vehicle size and state on a circular track with radius track_radius,
# center track_center, and width lane_width.
# """
# function generate_random_vehicle(rng, lane_width, track_radius, track_center, min_r, max_r, max_vel)
#     r = rand(rng)*(max_r-min_r) + min_r
#     θ = rand(rng)*2π
#     rad = rand(rng)*(lane_width-2r) + track_radius-lane_width/2+r
#     p1 = cos(θ-π/2)*rad
#     p2 = sin(θ-π/2)*rad
#     v = rand(rng)*max_vel/2 + max_vel
#     (; state=[p1, p2, v, θ], r)
# end

function generate_trajectory(localization_state_channel, 
    perception_state_channel, 
    cur_seg, 
    ,  gt_channel)
    
    # ego, V2, V3, lane_width, track_radius, track_center, callbacks, timestep, trajectory_length)
    # X1 = ego.state
    # X2 = V2.state
    # X3 = V3.state
    # r1 = ego.r
    # r2 = V2.r
    # r3 = V3.r
    latest_localization_state = fetch(localization_state_channel)
    latest_perception_state = fetch(perception_state_channel)

    vehicle_state = latest_localization_state.x

    # TODO refine callbacks given current positions of vehicles, lane geometry,
    # etc.
    callbacks = create_callback_generator(vehicle_state, 
    latest_perception_state, 
    cur_seg)
    
    wrapper_f = function(z)
        callbacks.full_cost_fn(z, vehicle_state, latest_perception_state, cur_seg)
    end
    wrapper_grad_f = function(z, grad)
        callbacks.full_cost_grad_fn(grad, z, vehicle_state, latest_perception_state, cur_seg)
    end
    wrapper_con = function(z, con)
        callbacks.full_constraint_fn(con, z, vehicle_state, latest_perception_state, cur_seg)
    end
    wrapper_con_jac = function(z, rows, cols, vals)
        if isnothing(vals)
            rows .= callbacks.full_constraint_jac_triplet.jac_rows
            cols .= callbacks.full_constraint_jac_triplet.jac_cols
        else
            callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z, vehicle_state, latest_perception_state, cur_seg)
        end
        nothing
    end
    wrapper_lag_hess = function(z, rows, cols, cost_scaling, λ, vals)
        if isnothing(vals)
            rows .= callbacks.full_lag_hess_triplet.hess_rows
            cols .= callbacks.full_lag_hess_triplet.hess_cols
        else
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, vehicle_state, latest_perception_state, cur_seg, λ, cost_scaling)
        end
        nothing
    end

    n = Inf
    m = length(callbacks.constraints_lb)
    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(-Inf, n),
        fill(Inf, n),
        length(callbacks.constraints_lb),
        callbacks.constraints_lb,
        callbacks.constraints_ub,
        length(callbacks.full_constraint_jac_triplet.jac_rows),
        length(callbacks.full_lag_hess_triplet.hess_rows),
        wrapper_f,
        wrapper_con,
        wrapper_grad_f,
        wrapper_con_jac,
        wrapper_lag_hess
    )

    controls = repeat([zeros(2),], 200)
    #states = repeat([X1,], trajectory_length)
    #zinit = compose_trajectory(states, controls)
    #prob.x = zinit

    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    status = Ipopt.IpoptSolve(prob)

    if status != 0
        @warn "Problem not cleanly solved. IPOPT status is $(status)."
    end
    #states, controls = decompose_trajectory(prob.x)
    (; controls, status)
end

# function visualize_simulation(sim_results, track_radius, track_center, lane_width)
#     f = Figure()
#     ax = f[1,1] = Axis(f, aspect = DataAspect())
#     r_inside = track_radius - lane_width/2
#     r_outside = track_radius + lane_width/2
#     xlims!(ax, track_center[1]-r_outside, track_center[1]+r_outside)
#     ylims!(ax, track_center[2]-r_outside, track_center[2]+r_outside)

#     θ = 0:0.1:2pi+0.1
#     lines!(r_inside*cos.(θ), r_inside*sin.(θ), color=:black, linewidth=3)
#     lines!(r_outside*cos.(θ), r_outside*sin.(θ), color=:black, linewidth=3)

#     ps = [Observable(Point2f(v.state[1], v.state[2])) for v in sim_results[1].vehicles]
#     traj = [Observable(Point2f(state[1], state[2])) for state in sim_results[1].trajectory.states]
#     for t in traj
#         plot!(ax, t, color=:green)
#     end

#     circles = [@lift(Circle($p, v.r)) for (p,v) in zip(ps, sim_results[1].vehicles)]
#     for (e, circle) in enumerate(circles)
#         if e == 1
#             poly!(ax, circle, color = :blue)
#         else
#             poly!(ax, circle, color = :red)
#         end
#     end

#     record(f, "mpc_animation.mp4", sim_results;
#         framerate = 10) do sim_step
#             for (t,state) in zip(traj, sim_step.trajectory.states)
#                 t[] = Point2f(state[1], state[2])
#             end
#             for (p,v) in zip(ps, sim_step.vehicles)
#                 p[] = Point2f(v.state[1], v.state[2])
#             end
#             display(f)
#             sleep(0.25)
#         end
            
# end