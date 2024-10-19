# x is [p1 p2 v theata]
# u is [a angular velocity]
# Δ time step

function euler_to_quaternion(yaw, pitch, roll)

    qx = sin(roll/2) * cos(pitch/2) * cos(yaw/2) - cos(roll/2) * sin(pitch/2) * sin(yaw/2)
    qy = cos(roll/2) * sin(pitch/2) * cos(yaw/2) + sin(roll/2) * cos(pitch/2) * sin(yaw/2)
    qz = cos(roll/2) * cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * cos(yaw/2)
    qw = cos(roll/2) * cos(pitch/2) * cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)

    return [qx, qy, qz, qw]
end


# function extract_yaw_from_quaternion(q)
#     atan(2(q[1]*q[4]+q[2]*q[3]), 1-2*(q[3]^2+q[4]^2))
# end

# function J_Tbody(x)
#     J_Tbody_xyz = (zeros(3,4), zeros(3,4), zeros(3,4))
#     for i = 1:3
#        J_Tbody_xyz[i][i,4] = 1.0
#     end
#     return (J_Tbody_xyz, [[dR zeros(3)] for dR in J_R_q(x[4:7])])
# end  


"""
Unicycle model
"""
function f(position, quaternion, velocity, angular_vel, Δt)
    r = angular_vel
    mag = norm(r)

    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * (r / mag)
    end

    sₙ = quaternion[1]
    vₙ = quaternion[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    R = Rot_from_quat(quaternion)  
    new_position = position + Δt * R * velocity
    new_quaternion = [s; v]
    new_velocity = velocity
    new_angular_vel = angular_vel
    return [new_position; new_quaternion; new_velocity; new_angular_vel]


    # r = angular_vel
    # mag = norm(r)

    # if mag < 1e-5
    #     sᵣ = 1.0
    #     vᵣ = zeros(3)
    # else
    #     sᵣ = cos(mag*Δt / 2.0)
    #     vᵣ = sin(mag*Δt / 2.0) * axis
    # end

    # sₙ = quaternion[1]
    # vₙ = quaternion[2:4]

    # s = sₙ*sᵣ - vₙ'*vᵣ
    # v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    # new_position = position + Δt * velocity
    # new_quaternion = [s; v]
    # new_velocity = velocity
    # new_angular_vel = angular_vel
    # return [new_position; new_quaternion; new_velocity; new_angular_vel]
end

# function J_R_q(q)
#     qw = q[1]
#     qx = q[2]
#     qy = q[3]
#     qz = q[4]

#     dRdq1 = 2*[qw -qz qy;
#              qz qw -qx;
#              -qy qx qw]
#     dRdq2 = 2*[qx qy qz;
#                qy -qx -qw;
#                qz qw -qx]
#     dRdq3 = 2*[-qy qx qw;
#                qx qy qz;
#                -qw qz -qy]
#     dRdq4 = 2*[-qz -qw qx;
#                qw -qz qy;
#                qx qy qz]
#     (dRdq1, dRdq2, dRdq3, dRdq4)
# end

"""
Jacobian of f_localization with respect to x, evaluated at x,Δ.
"""
function jac_fx(x, Δt)
    J = zeros(13, 13)

    r = x[11:13]
    mag = norm(r)
    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * (r / mag)
    end
    sₙ = x[4]
    vₙ = x[5:7]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    R = Rot_from_quat([sₙ; vₙ])  
    (J_R_q1, J_R_q2, J_R_q3, J_R_q4) = J_R_q([sₙ; vₙ])
    velocity = x[8:10]

    J[1:3, 1:3] = I(3)
    J[1:3, 4] = Δt * J_R_q1*velocity
    J[1:3, 5] = Δt * J_R_q2*velocity
    J[1:3, 6] = Δt * J_R_q3*velocity
    J[1:3, 7] = Δt * J_R_q4*velocity
    J[1:3, 8:10] = Δt * R
    J[4, 4] = sᵣ
    J[4, 5:7] = -vᵣ'
    J[5:7, 4] = vᵣ
    J[5:7, 5:7] = [sᵣ vᵣ[3] -vᵣ[2];
                   -vᵣ[3] sᵣ vᵣ[1];
                   vᵣ[2] -vᵣ[1] sᵣ]
 
    Jsv_srvr = [sₙ -vₙ'
                vₙ [sₙ -vₙ[3] vₙ[2];
                    vₙ[3] sₙ -vₙ[1];
                    -vₙ[2] vₙ[1] sₙ]]
    Jsrvr_mag = [-sin(mag*Δt / 2.0) * Δt / 2; sin(mag*Δt/2.0) * (-r / mag^2) + cos(mag*Δt/2)*Δt/2 * r/mag]
    Jsrvr_r = [zeros(1,3); sin(mag*Δt / 2) / mag * I(3)]
    Jmag_r = 1/mag * r'
    J[4:7, 11:13] = Jsv_srvr * (Jsrvr_mag*Jmag_r + Jsrvr_r)
    J[8:10, 8:10] = I(3)
    J[11:13, 11:13] = I(3)
    J
 
end





"""
Non-standard measurement model. Can we extract state estimate from just this?
"""
# x is position, quaternion, velocity, angular vel
# function h_gps(x)
    
#     T = get_gps_transform()
#     gps_loc_body = T*[zeros(3); 1.0]
#     xyz_body = x[1:3] # position
#     q_body = x[4:7] # quaternion
#     Tbody = get_body_transform(q_body, xyz_body)
#     xyz_gps = Tbody * [gps_loc_body; 1]
#     yaw = extract_yaw_from_quaternion(q_body)
#     meas = [xyz_gps[1:2]; yaw]
# end

function h_imu(x)
    velocity = x[8: 10]
    angular_vel = x[11:13]
    T_body_imu = get_imu_transform()
    T_imu_body = invert_transform(T_body_imu)
    R_imu = T_imu_body[1:3,1:3]
    p_imu = T_imu_body[1:3,end]
    w_imu = R_imu * angular_vel
    v_imu = R_imu * velocity + p_imu × w_imu


    return [v_imu; w_imu]
end

"""
Jacobian of h with respect to x, evaluated at x.
"""
function jac_h_gps(x) # 3 * 13


    T = get_gps_transform()
    gps_loc_body = T*[zeros(3); 1.0]
    xyz_body = x[1:3] # position
    q_body = x[4:7] # quaternion
    Tbody = get_body_transform(q_body, xyz_body)
    xyz_gps = Tbody * [gps_loc_body; 1]
    yaw = extract_yaw_from_quaternion(q_body)
    J = zeros(3, 13)
    (J_Tbody_xyz, J_Tbody_q) = J_Tbody(x)
    for i = 1:3
        J[1:2,i] = (J_Tbody_xyz[i]*[gps_loc_body; 1])[1:2]
    end
    for i = 1:4
	J[1:2,3+i] = (J_Tbody_q[i]*[gps_loc_body; 1])[1:2]
    end
    w = q_body[1]
    x = q_body[2]
    y = q_body[3]
    z = q_body[4]
    J[3,4] = -(2 * z * (-1 + 2 * (y^2 + z^2)))/(4 * (x * y + w * z)^2 + (1 - 2 * (y^2 + z^2))^2)
    J[3,5] = -(2 * y * (-1 + 2 * (y^2 + z^2)))/(4 * (x * y + w * z)^2 + (1 - 2 * (y^2 + z^2))^2)
    J[3,6] = (2 * (x + 2 * x * y^2 + 4 * w * y * z - 2 * x * z^2))/(1 + 4 * y^4 + 8 * w * x * y * z + 4 * (-1 + w^2) * z^2 + 4 * z^4 + 4 * y^2 * (-1 + x^2 + 2 * z^2))
    J[3,7] = (2 * (w - 2 * w * y^2 + 4 * x * y * z + 2 * w * z^2))/(1 + 4 * y^4 + 8 * w * x * y * z + 4 * (-1 + w^2) * z^2 + 4 * z^4 + 4 * y^2 * (-1 + x^2 + 2 * z^2))
    J
end


    

function jac_h_imu(x)
   [0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.9998  0.0     -0.0199 0.0     0.7     0.0;
    0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     1.0     0.0     -0.7    0.0     0.0;
    0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0199  0.0     0.9998  0.0     0.014   0.0;
    0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     0.0     0.0     0.9998  0.0     -0.0199;
    0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     0.0     0.0     0.0     1.0     0.0;
    0.0 0.0 0.0    0.0                             0.0                             0.0                             0.0                             0.0     0.0     0.0     0.0199  0.0     0.9998]
end



"""
Extended kalman filter implementation.

Assume that the 'true' physical update in the world is given by 

xₖ = f(xₖ₋₁, uₖ, Δ), where Δ is the time difference between times k and k-1.

Here, uₖ is the 'true' controls applied to the system. These controls can be assumed to be a random variable,
with probability distribution given by 𝒩 (mₖ, proc_cov) where mₖ is some IMU-like measurement, and proc_cov is a constant covariance matrix.

The process model distribution is then approximated as:

P(xₖ | xₖ₋₁, uₖ) ≈ 𝒩 ( Axₖ₋₁  + c, Σ̂ )

where 
A = ∇ₓf(μₖ₋₁,  Δ),

c = f(μₖ₋₁, Δ) - Aμₖ₋₁ 

μ̂ = Aμₖ₋₁  + c
  = f(μₖ₋₁, mₖ, Δ)
Σ̂ = A Σₖ₋₁ A' + proc_cov(how much noisy is for the process model), 


Further, assume that the 'true' measurement generation in the world is given by

zₖ = h(xₖ) + wₖ,

where wₖ is some additive gaussian noise with probability density function given by

𝒩 (0, meas_var).

The measurement model is then approximated as 

P(zₖ | xₖ) ≈ 𝒩 ( C xₖ + d , meas_var )


where 
C = ∇ₓ h(μ̂), 
d = h(μ̂) - Cμ̂

The extended Kalman filter update equations can be implemented as the following:

Σₖ = (Σ̂⁻¹ + C' (meas_var)⁻¹ C)⁻¹
μₖ = Σₖ ( Σ̂⁻¹ μ̂ + C' (meas_var)⁻¹ (zₖ - d) )  | z is 6 by 1

"""
function filter(meas, Δ,  μ , Σ, localization_state_channel)
  #  # # @info " f1"
  meas_var = Diagonal([1 ;1; 0.05;0.1;0.1;0.1;0.1;0.05;0.05;0.05;0.05;0.05;0.05]) 
  meas_var_gps = Diagonal([0.05,0.05, 0.05])
  meas_var_imu = Diagonal([0.05,0.05,0.05,0.05,0.05,0.05]) 

  proc_cov = Diagonal([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
  x_prev = μ
  Σ_prev = Σ
  xₖ = f(x_prev[1:3], x_prev[4:7], x_prev[8:10], x_prev[11:13], Δ) #Current x
  z = meas
  A = jac_fx(xₖ, Δ)  
  μ̂ = f(x_prev[1:3], x_prev[4:7], x_prev[8:10], x_prev[11:13],  Δ)

#  @info "f2"
  Σ̂ = A * Σ_prev * A' + proc_cov

  if meas isa GPSMeasurement
      @info " GPS"
      tmp_z = []
      
      push!(tmp_z, z.lat)
      push!(tmp_z, z.long)
      push!(tmp_z, z.heading)
      C = jac_h_gps(μ̂) # gps version
      d = h_gps(μ̂) - C*μ̂ # gps version
      Σ = inv(inv(Σ̂) + C'*inv(meas_var_gps)*C)
      μ = Σ * (inv(Σ̂) * μ̂ + C' * inv(meas_var_gps) * (tmp_z - d))
      full_state = FullVehicleState(μ[1:3], μ[8:10], μ[11:13], μ[4:7])
      local_state = MyLocalizationType( meas.time, full_state)
      if isready(localization_state_channel)
          take!(localization_state_channel)
      end
      @info "$(local_state)"
      put!(localization_state_channel, local_state)
    #  @info "finished"
  else
      @info " IMU"
      tmp_z = []
      push!(tmp_z, z.linear_vel[1])
      push!(tmp_z, z.linear_vel[2])
      push!(tmp_z, z.linear_vel[3])
      push!(tmp_z, z.angular_vel[1])
      push!(tmp_z, z.angular_vel[2])
      push!(tmp_z, z.angular_vel[3])

   #   @info " imu 1"
      C = jac_h_imu(μ̂) # imu version
     # @info " imu 4"
      d = h_imu(μ̂) - C*μ̂ # imu version
    #  @info " imu 2"
      Σ = inv(inv(Σ̂) + C'*inv(meas_var_imu)*C)
      μ = Σ * (inv(Σ̂) * μ̂ + C'* (meas_var_imu) * (tmp_z - d))
    #  @info " imu 3"
      full_state = FullVehicleState(μ[1:3],μ[8:10], μ[11:13], μ[4:7])

      local_state = MyLocalizationType(meas.time, full_state)
      if isready(localization_state_channel)
          take!(localization_state_channel)
      end

      @info "local: state$(local_state)"
      put!(localization_state_channel, local_state)
  end

end