include("perception.jl")

struct SimpleVehicleState
    p1::Float64
    p2::Float64
    θ::Float64
    v::Float64
    l::Float64
    w::Float64
    h::Float64
end

struct FullVehicleState
    position::SVector{3, Float64}
    velocity::SVector{3, Float64}
    orientation::SVector{4, Float64}
    angular_vel::SVector{3, Float64}
end

struct MyLocalizationType
    last_update::Float64
    x::FullVehicleState
end

struct MyPerceptionType
    last_update::Float64
    x::Vector{SimpleVehicleState}
end

function test_algorithms(gt_channel,
        localization_state_channel,
        perception_state_channel, 
        ego_vehicle_id)
    estimated_vehicle_states = Dict{Int, Tuple{Float64, Union{SimpleVehicleState, FullVehicleState}}}
    gt_vehicle_states = Dict{Int, GroundTruthMeasurement}

    t = time()
    while true

        while isready(gt_channel)
            meas = take!(gt_channel)
            id = meas.vehicle_id
            if meas.time > gt_vehicle_states[id].time
                gt_vehicle_states[id] = meas
            end
        end

        latest_estimated_ego_state = fetch(localization_state_channel)
        latest_true_ego_state = gt_vehicle_states[ego_vehicle_id]
        if latest_estimated_ego_state.last_update < latest_true_ego_state.time - 0.5
            @warn "Localization algorithm stale."
        else
            estimated_xyz = latest_estimated_ego_state.position
            true_xyz = latest_true_ego_state.position
            position_error = norm(estimated_xyz - true_xyz)
            t2 = time()
            if t2 - t > 5.0
                @info "Localization position error: $position_error"
                t = t2
            end
        end

        latest_perception_state = fetch(perception_state_channel)
        last_perception_update = latest_perception_state.last_update
        vehicles = last_perception_state.x

        for vehicle in vehicles
            xy_position = [vehicle.p1, vehicle.p2]
            closest_id = 0
            closest_dist = Inf
            for (id, gt_vehicle) in gt_vehicle_states
                if id == ego_vehicle_id
                    continue
                else
                    gt_xy_position = gt_vehicle_position[1:2]
                    dist = norm(gt_xy_position-xy_position)
                    if dist < closest_dist
                        closest_id = id
                        closest_dist = dist
                    end
                end
            end
            paired_gt_vehicle = gt_vehicle_states[closest_id]

            # compare estimated to GT

            if last_perception_update < paired_gt_vehicle.time - 0.5
                @info "Perception upate stale"
            else
                # compare estimated to true size
                estimated_size = [vehicle.l, vehicle.w, vehicle.h]
                actual_size = paired_gt_vehicle.size
                @info "Estimated size error: $(norm(actual_size-estimated_size))"
            end
        end
    end
end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end
        
        # process measurements

        localization_state = MyLocalizationType(0,0.0)
        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function fake_localize(gt_channel, localization_state_channel, ego_id)
    while true
        sleep(0.001)
        fresh_gt_meas = []
        while isready(gt_channel)
            meas = take!(gt_channel)
            push!(fresh_gt_meas, meas)
        end

        latest_meas_time = -Inf
        latest_meas = nothing
        for meas in fresh_gt_meas
            #@info "vehicle id: $(meas.vehicle_id), time: $(meas.time), ego id: $(ego_id)"

            if meas.time > latest_meas_time && meas.vehicle_id - 1 == ego_id
                latest_meas = meas
                latest_meas_time = meas.time
            end
            
        end
        # @info "$(latest_meas)"
        if isnothing(latest_meas)
            continue
        end
        # @info "test1"
        
        # Convert latest_meas to MyLocalizationType
        time = latest_meas.time
        # @info "test2"
        fvs = FullVehicleState(latest_meas.position, latest_meas.velocity, latest_meas.orientation, latest_meas.angular_velocity)
        # @info "test3"
        my_converted_gt_message = MyLocalizationType(time, fvs)
        # @info "test4"

        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        # @info "test5"
        
        put!(localization_state_channel, my_converted_gt_message)
        # @info "we have $(length(localization_state_channel.data))"
    end
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    
    # # set up stuff
    # while true
    #     # @info "1"

    #     # obtain camera measurements
    #     fresh_cam_meas = []
        
    #     while isready(cam_meas_channel)
            
    #         meas = take!(cam_meas_channel)
    #         push!(fresh_cam_meas, meas)
    #     end

    #     # Should we do this in the loop?
    #     # obtain localization information
    #     #latest_localization_state = fetch(localization_state_channel)
    #     latest_localization_state = nothing
    #     if isready(localization_state_channel)
    #         latest_localization_state = fetch(localization_state_channel)
    #     end
    #     if isnothing(latest_localization_state)
    #         continue
    #     end
    #     # @info "latest localization state: $(latest_localization_state)"
        
    #     loc_state = latest_localization_state.x
    #     loc_time = latest_localization_state.last_update
        
    #     # Initial values of μ and Σ for x0
    #     μ_init = [loc_state.position[1] loc_state.position[2] 0 0 13.2 5.7 5.3]
    #     Σ_init = Diagonal([5, 5, 0, 1, 0.01, 0.01, 0.01])

    #     # μₖ₋₁ and Σₖ₋₁. They are initialized as their initial value.
    #     μ_prev = μ_init
    #     Σ_prev = Σ_init

    #     # Σ for measurement model and process model. Probably need to finetune them.
    #     Σₘ = Diagonal([5, 5, 0, 1, 0.01, 0.01, 0.01])
    #     Σₚ = Diagonal([0.1, 0.1, 0.1, 0.1])

    #     # curr_time is the time at which the currently processing camera measurement is obtained.
    #     curr_time = -Inf

    #     # prev_time is the time at which the previously processed camera measurement is obtained.
    #     prev_time = 0

    #     # μ_prev_list is a list of predicted states of objects described by the bboxes in the previous 
    #     # camera measurements. Σ_prev_list is similar, but it's for Σ. They are updated after EKF has
    #     # processed each camera measurement.
    #     μ_prev_list = []
    #     Σ_prev_list = []

    #     μ_list = []
    #     Σ_list = []

    #     # @info "fresh camera meas: $(fresh_cam_meas)"

        
    #     # Process camera measurements.
    #     for i in fresh_cam_meas

    #         # if i.time < curr_time, we just discard this measurement.
    #         if i.time > curr_time
    #             curr_time = i.time
    #             if !isempty(i.bounding_boxes)
                    
    #                 # In case the localization information was not obtained at the same at which the
    #                 # camera measurement was obtained, we predict the localization information at the
    #                 # time the camera measurement was obtained, using the given localization information.
    #                 x_ego = rigid_body_dynamics(loc_state.position, loc_state.orientation, loc_state.velocity, 
    #                                             loc_state.angular_velocity, curr_time - loc_time)

    #                 # Δ is the time step.
    #                 Δ = curr_time - prev_time

    #                 # assign μ_prev to bounding boxes
    #                 μ_index = assign_bb(i.camera_id, μ_prev_list, i.bounding_boxes, x_ego, Δ)
    #                 # @info "μ index: $(μ_index)"

    #                 for j in eachindex(i.bounding_boxes)

    #                     if isempty(μ_prev_list)
    #                         μ_prev_list = [μ_init for i in length(i.bounding_boxes)]
    #                         Σ_prev_list = [Σ_init for i in length(i.bounding_boxes)]
    #                     end

    #                     # if bbox cannot be matched with a previous object, we perform EKF from start
    #                     # by giving it initial values for μ and Σ
    #                     if μ_index[j] != 0
    #                         μ_prev = μ_prev_list[μ_index[j]]
    #                         Σ_prev = Σ_prev_list[μ_index[j]]
    #                     else
    #                         μ_prev = μ_init
    #                         Σ_prev = Σ_init
    #                     end                        

    #                     # Extended Kalman Filter
    #                     A = jac_f(μ_prev, Δ)
    #                     Σ̂  = Σₘ + A * Σ_prev * A'
    #                     μ̂  = f(μ_prev, Δ)
    #                     h1 = h(i.camera_id, μ̂ , x_ego)
    #                     C = jac_h(μ̂ , h1)
    #                     Σ = inv(inv(Σ̂ )+ C' * inv(Σₚ) * C)
    #                     μ = Σ * (inv(Σ̂ ) * μ̂ + C' * inv(Σₚ) * (i.bounding_boxes[j]))
    #                     μ_prev = μ
    #                     push!(μ_list, μ_prev)
    #                     Σ_prev = Σ
    #                     push!(Σ_list, Σ_prev)
    #                 end

    #                 μ_prev_list = μ_list
    #                 Σ_prev_list = Σ_list
    #                 μ_list = []
    #                 Σ_list = []
    #                 # @info "μ prev list: $(μ_prev_list)"
    #             end
    #             prev_time = curr_time
    #         else
    #             continue
    #         end
    #     end

    #     # Output.
    #     @info "1"
        
    #     μ_prev_list_struct = []
    #     for i in μ_prev_list
    #         temp = SimpleVehicleState(i[1], i[2], i[3], i[4], i[5], i[6], i[7])
    #         push!(μ_prev_list_struct, temp)
    #     end
    #     perception_state = MyPerceptionType(curr_time, μ_prev_list_struct)
        
    #     if isready(perception_state_channel)
    #         take!(perception_state_channel)
    #     end
    #     @info "perception: $(perception_state)"
        
    #     put!(perception_state_channel, perception_state)
    # end

    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        # Should we do this in the loop?
        # obtain localization information
        
        latest_localization_state = fetch(localization_state_channel)
        # @info "localization state: $(latest_localization_state)"
        # latest_localization_state = nothing
        # if isready(localization_state_channel)
        #     latest_localization_state = fetch(localization_state_channel)
        # end
        # if isnothing(latest_localization_state)
        #     continue
        # end
        # @info "latest localization state: $(latest_localization_state)"
        
        loc_state = latest_localization_state.x
        loc_time = latest_localization_state.last_update
        # @info "loc_state: $(loc_state)"
        # @info "loc_time: $(loc_time)"
        
        # Initial values of μ and Σ for x0
        μ_init = [loc_state.position[1]+14 loc_state.position[2] 0 0.3 13.2 5.7 5.3]
        Σ_init = Diagonal([5, 5, 0.01, 1, 0.01, 0.01, 0.01])

        # μₖ₋₁ and Σₖ₋₁. They are initialized as their initial value.
        μ_prev = μ_init
        Σ_prev = Σ_init

        # Σ for measurement model and process model. Probably need to finetune them.
        Σₘ = Diagonal([3, 3, 0.001, 1, 0.001, 0.001, 0.001])
        Σₚ = Diagonal([1, 1, 1, 1])

        # curr_time is the time at which the currently processing camera measurement is obtained.
        curr_time = -Inf

        # prev_time is the time at which the previously processed camera measurement is obtained.
        prev_time = 0

        # μ_prev_list is a list of predicted states of objects described by the bboxes in the previous 
        # camera measurements. Σ_prev_list is similar, but it's for Σ. They are updated after EKF has
        # processed each camera measurement.
        μ_prev_list = []
        Σ_prev_list = []

        μ_list = []
        Σ_list = []

        

        # Process camera measurements.
        for i in fresh_cam_meas
            #@info "cam_meas: $(i)"
            

            # if i.time < curr_time, we just discard this measurement.
            if i.time >= curr_time
                curr_time = i.time
                #@info "cur time: $(curr_time)"
                
                if !isempty(i.bounding_boxes)
                                  
                    # In case the localization information was not obtained at the same at which the
                    # camera measurement was obtained, we predict the localization information at the
                    # time the camera measurement was obtained, using the given localization information.
                    # @info "position: $(loc_state.position)"
                    # @info "orientation: $(loc_state.orientation)"
                    # @info "velocity: $(loc_state.velocity)"
                    # @info "angular velocity: $(loc_state.angular_vel)"
                    # @info "Δ: $(curr_time - loc_time)"
                    x_ego = rigid_body_dynamics(loc_state.position, loc_state.orientation, loc_state.velocity, 
                                                loc_state.angular_vel, curr_time - loc_time)
                    #@info "x_ego: $(x_ego)"

                    # Δ is the time step.
                    Δ = curr_time - prev_time
                    #@info "Δ: $(Δ)"

                    # assign μ_prev to bounding boxes
                    if isempty(μ_prev_list)
                        μ_prev_list = [μ_init for i in length(i.bounding_boxes)]
                        Σ_prev_list = [Σ_init for i in length(i.bounding_boxes)]
                    end

                    #@info "μ_prev_list: $(μ_prev_list)"

                    μ_index_raw = assign_bb(i.camera_id, μ_prev_list, i.bounding_boxes, x_ego, Δ)
                    μ_index = [trunc(Int, i) for i in μ_index_raw]

                    for j in eachindex(i.bounding_boxes)

                        # if bbox cannot be matched with a previous object, we perform EKF from start
                        # by giving it initial values for μ and Σ
                        if μ_index[j] != 0
                            μ_prev = μ_prev_list[μ_index[j]]
                            Σ_prev = Σ_prev_list[μ_index[j]]
                        else
                            μ_prev = μ_init
                            Σ_prev = Σ_init
                        end 
                        #@info "μ_prev1: $(μ_prev)"    
                              

                        # Extended Kalman Filter
                        A = jac_f(μ_prev, Δ)
                       
                        Σ̂  = Σₘ + A * Σ_prev * A'
                        
                        μ̂  = f(μ_prev, Δ)
                        
                        h1 = h(i.camera_id, μ̂ , x_ego)
                        
                        C = jac_h(μ̂ , h1)
                        
                        # @info "A: $(A)"
                        # @info "Σ_prev: $(Σ_prev)"
                        # @info "Σₘ: $(Σₘ)"
                        # @info "Σ̂ : $(Σ̂ )"
                        # @info "inv(Σ̂ ): $(inv(Σ̂ ))"
                        # @info "inv(Σₚ): $(inv(Σₚ))"
                        Σ = inv(inv(Σ̂ )+ C' * inv(Σₚ) * C)
                        
                        # @info "9"
                        μ = Σ * (inv(Σ̂ ) * μ̂ + C' * inv(Σₚ) * (i.bounding_boxes[j]))
                        
                        # @info "10"
                        μ_prev = μ
                        #@info "μ_prev2: $(μ_prev)"
                        push!(μ_list, μ_prev)
                        Σ_prev = Σ
                        push!(Σ_list, Σ_prev)
                    end

                    μ_prev_list = μ_list
                    Σ_prev_list = Σ_list
                    μ_list = []
                    Σ_list = []
                    # @info "μ prev list: $(μ_prev_list)"
                end
                
                prev_time = curr_time
            else
                
                continue
            end
        end

        μ_prev_list_struct = []
        for i in μ_prev_list
            temp = SimpleVehicleState(i[1], i[2], i[3], i[4], i[5], i[6], i[7])
            push!(μ_prev_list_struct, temp)
        end
        perception_state = MyPerceptionType(curr_time, μ_prev_list_struct)
        #if perception_state.last_update != -Inf
        @info "perpection: $(perception_state)"
        #end

        #perception_state = MyPerceptionType(0.0, [])
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

function decision_making(localization_state_channel, 
        perception_state_channel, 
        map, 
        target_road_segment_id, 
        socket)
    # do some setup
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 0.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end


function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end

function debug(gt_channel, cam_meas_channel, ego_vehicle_id)
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        gt_x = []
        gt_ego = []
        while isready(gt_channel)
            meas = take!(gt_channel)
            if meas.vehicle_id != ego_vehicle_id
                push!(gt_x, meas)
            else
                push!(gt_ego, meas)
            end
        end
        



    end

end

function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment
        # @info "meas msg: $(measurement_msg)"
        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                # @info "cam: $(meas)"
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                if meas.vehicle_id != ego_vehicle_id
                    pos = meas.position[1:2]
                    temp = [pos[1], pos[2], 0, 0, 13.2, 5.7, 5.3]
                    time = meas.time
                    @info "gt_x: $(time), $(temp)"
                end
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    # @async localize(gps_channel, imu_channel, localization_state_channel)

    @async fake_localize(gt_channel, localization_state_channel, ego_vehicle_id)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, target_map_segment, socket)
end