include("perception.jl")
include("measurements.jl")

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
    orientation::SVector{3, Float64}
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

"""
time: the time at which the corresponding camera measurement was taken.
μ: an array of states of the objects being tracked.
Σ: an array of variances of the states of objects being tracked.
"""
# struct MyPerceptionType
#     time::Float64
#     μ::SVector{13, Float64}
#     Σ::Array{Float64, 2}
# end

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

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    
    while true

        # obtain camera measurements
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        # Should we do this in the loop?
        # obtain localization information
        latest_localization_state = fetch(localization_state_channel)
        loc_state = latest_localization_state.x
        loc_time = latest_localization_state.last_update
        
        # Initial values of μ and Σ for x0
        μ_init = [loc_state.position[1] loc_state.position[2] 0 0 13.2 5.7 5.3]
        Σ_init = Diagonal([5, 5, 0, 1, 0.01, 0.01, 0.01])

        # μₖ₋₁ and Σₖ₋₁. They are initialized as their initial value.
        μ_prev = μ_init
        Σ_prev = Σ_init

        # Σ for measurement model and process model. Probably need to finetune them.
        Σₘ = Diagonal([5, 5, 0, 1, 0.01, 0.01, 0.01])
        Σₚ = Diagonal([0.1, 0.1, 0.1, 0.1])

        # curr_time is the time at which the currently processing camera measurement is obtained.
        curr_time = -Inf

        # prev_time is the time at which the previously processed camera measurement is obtained.
        prev_time = 0

        # μ_prev_list is a list of predicted states of objects described by the bboxes in the previous 
        # camera measurements. Σ_prev_list is similar, but it's for Σ. They are updated after EKF has
        # processed each camera measurement.
        μ_prev_list = []
        Σ_prev_list = []

        # Process camera measurements.
        for i in fresh_cam_meas

            # clear these two lists.
            μ_prev_list = []
            Σ_prev_list = []

            # if i.time < curr_time, we just discard this measurement.
            if i.time > curr_time
                curr_time = i.time
                if !isempty(i.bounding_boxes)
                    
                    # In case the localization information was not obtained at the same at which the
                    # camera measurement was obtained, we predict the localization information at the
                    # time the camera measurement was obtained, using the given localization information.
                    x_ego = rigid_body_dynamics(loc_state.position, loc_state.orientation, loc_state.velocity, 
                                                loc_state.angular_velocity, curr_time - loc_time)

                    # Δ is the time step.
                    Δ = curr_time - prev_time

                    # assign μ_prev to bounding boxes
                    μ_index = assign_bb(i.camera_id, μ_prev_list, i.bounding_boxes, x_ego, Δ)

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

                        # Extended Kalman Filter
                        A = jac_f(μ_prev, Δ)
                        Σ̂  = Σₘ + A * Σ_prev * A'
                        μ̂  = f(μ_prev, Δ)
                        h1 = h(i.camera_id, μ̂ , x_ego)
                        C = jac_h(μ̂ , h1)
                        Σ = inv(inv(Σ̂ )+ C' * inv(Σₚ) * C)
                        μ = Σ * (inv(Σ̂ ) * μ̂ + C' * inv(Σₚ) * (i.bounding_boxes[j]))
                        μ_prev = μ
                        push!(μ_prev_list, μ_prev)
                        Σ_prev = Σ
                        push!(Σ_prev_list, Σ_prev)
                    end
                end
                prev_time = curr_time
            else
                continue
            end
        end

        # Output.
        μ_prev_list_struct = []
        for i in μ_prev_list
            temp = SimpleVehicleState(i[1], i[2], i[3], i[4], i[5], i[6], i[7])
            push!(μ_prev_list_struct, temp)
        end
        perception_state = MyPerceptionType(curr_time, μ_prev_list_struct)
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


function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    #localization_state_channel = Channel{MyLocalizationType}(1)
    #perception_state_channel = Channel{MyPerceptionType}(1)

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
        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    @async localize(gps_channel, imu_channel, localization_state_channel)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, socket)
end
