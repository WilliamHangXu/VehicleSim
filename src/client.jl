struct VehicleCommand
    steering_angle::Float64
    velocity::Float64
    controlled::Bool
end

function get_c()
    ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, true)
    ret == 0 || error("unable to switch to raw mode")
    c = read(stdin, Char)
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, false)
    c
end

function keyboard_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10)
    socket = Sockets.connect(host, port)
    (peer_host, peer_port) = getpeername(socket)
    msg = deserialize(socket) # Visualization info
    @info msg

    @async while isopen(socket)
        sleep(0.001)
        state_msg = deserialize(socket)
        measurements = state_msg.measurements
        num_cam = 0
        num_imu = 0
        num_gps = 0
        num_gt = 0
        for meas in measurements
            if meas isa GroundTruthMeasurement
                num_gt += 1
            elseif meas isa CameraMeasurement
                num_cam += 1
            elseif meas isa IMUMeasurement
                num_imu += 1
            elseif meas isa GPSMeasurement
                num_gps += 1
            end
        end
  #      @info "Measurements received: $num_gt gt; $num_cam cam; $num_imu imu; $num_gps gps"
    end
    
    target_velocity = 0.0
    steering_angle = 0.0
    controlled = true
    @info "Press 'q' at any time to terminate vehicle."
    while controlled && isopen(socket)
        key = get_c()
        if key == 'q'
            # terminate vehicle
            controlled = false
            target_velocity = 0.0
            steering_angle = 0.0
            @info "Terminating Keyboard Client."
        elseif key == 'i'
            # increase target velocity
            target_velocity += v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'k'
            # decrease forward force
            target_velocity -= v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'j'
            # increase steering angle
            steering_angle += s_step
            @info "Target steering angle: $steering_angle"
        elseif key == 'l'
            # decrease steering angle
            steering_angle -= s_step
            @info "Target steering angle: $steering_angle"
        end
        cmd = VehicleCommand(steering_angle, target_velocity, controlled)
        serialize(socket, cmd)
    end
end

# function example_client(host::IPAddr=IPv4(0), port=4444)
#     socket = Sockets.connect(host, port)
#     map_segments = training_map()
#     (; chevy_base) = load_mechanism()

#     @async while isopen(socket)
#         state_msg = deserialize(socket)
#     end
   
#     shutdown = false
#     persist = true
#     while isopen(socket)
#         position = state_msg.q[5:7]
#         @info position
#         if norm(position) >= 100
#             shutdown = true
#             persist = false
#         end
#         cmd = VehicleCommand(0.0, 2.5, persist, shutdown)
#         serialize(socket, cmd) 
#     end

# end

function example_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    (peer_host, peer_port) = getpeername(socket)
    msg = deserialize(socket) # Visualization info
    @info msg

    @async for j in 1:500
        println(j)
        state_msg = deserialize(socket)
        measurements = state_msg.measurements
        @info state_msg
    end

    for i in 500:1500
        println(i)
        #latest_localization_state = fetch(localization_state_channel)
        #latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        if i == 1500
            steering_angle = 0.0
            target_vel = 0.0
            cmd = VehicleCommand(steering_angle, target_vel, false)
            serialize(socket, cmd)
            close(socket)
        else
            steering_angle = 0.0
            target_vel = 0.5
            cmd = VehicleCommand(steering_angle, target_vel, true)
            serialize(socket, cmd)
        end
    end

end
