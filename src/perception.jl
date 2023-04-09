"""
This is the perception module of our autonomous vehicle.
We use Extended Kalman Filter (EKF) to estimate the state of objects (other cars) detected by the 
cameras.

Outline of EKF:
    a. Previous measurement: P(xₖ | xₖ₋₁) = 𝒩(μₖ₋₁, Σₖ₋₁)
    b. Process model: P(xₖ | xₖ₋₁) = 𝒩(f(xₖ₋₁), Σₚ)
    c. Measurement model: P(zₖ | xₖ) = 𝒩(h(xₖ, xegoₖ), Σₘ)
    where x = [p1, p2, θ, v, l, w, h] describes the state of an object, 
    and z = [y1, y2, y3, y4, y5, y6, y7, y8] describes the measurement (bb) collected by the two 
    cameras.

"""

"""
# Modified from get_body_transform in measurement.jl by Gavin
This function calculates a matrix that expresses a point in a loc-centered frame into world frame.

@param
loc: the coordinate of a point.
@output
a matrix.
"""
function get_body_transform_perception(loc)
    R = one(RotMatrix{3, Float64})
    [R loc]
end

"""
# Modified from get_3d_bbox_corners in measurement.jl by Gavin
This function calculates the coordinates of the 8 corners of the 3D bounding box describing the
object.

@param
position: position of the object ([x,y,z]) in world frame
box_size: size of bounding box

@output
An array of 8 points ([x,y,z]). These are the 8 corners of the 3D bounding box describing an object.
"""
function get_3d_bbox_corners_perception(position, box_size)
    T = get_body_transform_perception(position)
    corners = []
    for dx in [-box_size[1]/2, box_size[1]/2]
        for dy in [-box_size[2]/2, box_size[2]/2]
            for dz in [-box_size[3]/2, box_size[3]/2]
                push!(corners, T*[dx, dy, dz, 1])
            end
        end
    end
    corners
end

"""
# Copied from measurement.jl by William
This function converts image frame coordinates into pixel.
"""
function convert_to_pixel(num_pixels, pixel_len, px)
    min_val = -pixel_len*num_pixels/2
    pix_id = cld(px - min_val, pixel_len)+1 |> Int
    return pix_id
end

"""
This function predicts current state based on previous state.

@param
x: previous state [p1, p2, θ, v, l, w, h]
Δ: time step

@output
current state
"""
function f(x, Δ)
    [
        x[1] + Δ * x[4] * cos(θ)
        x[2] + Δ * x[4] * sin(θ)
        x[3]
        x[4]
        x[5]
        x[6]
        x[7]
    ]
end

"""
# Modified from cameras in measurement.jl by William Xu
This function predicts bounding box measurements based on the state of the object being tracked.

@param
x: the state of the object being tracked
x_ego: the state of ego vehicle

@output
bounding box measurement of the object being tracked
"""
function h(x, x_ego, focal_len=0.01, pixel_len=0.001, image_width=640, image_height=480)
    # the position of the vehicle in world frame
    x_ego_pos_world = x_ego[1:3]      # p1, p2, p3
    # the position of the object in world frame
    x_pos_world = x[1:3]              # p1, p2, p3
    # the angle of the vehicle
    x_ego_angles = x_ego[4:6]   # r, p, y
    x_size = x[6:8]             # l, w, h
    
    # two cameras
    z1 = []
    z2 = []

    # corners_world is an array that contains coordinates of 3d bbox of the object in world frame
    corners_world = get_3d_bbox_corners_perception(x_pos_world, x_size)
    corners_img_1 = []
    corners_img_2 = []

    left = image_width/2
    right = -image_width/2
    top = image_height/2
    bot = -image_height/2

    # This for-loop convert coordinates in corners_world into coordinates expressed in two 
    # cameras' image frames.
    for world in corners_world

        # x_pos_ego is the coordinate of a point (one of the corners of bbox) expressed in ego frame     
        x_pos_ego = world - x_ego_pos_world

        # x_ego_rot is a matrix that can correct ego's rpy angles.
        # Now rpy should be 0 for the ego car.
        x_ego_rot = inv(RotXYZ(x_ego_angles))

        # Since we corrected all the rpy angles, we must correct the position of the object as well, 
        # so its relatve position to the camera doesn't change.
        # After applying x_ego_rot, the new x_pos_body is now the corrected coordinate of a corner.
        x_pos_ego = x_ego_rot * x_pos_ego

        # express the corner in camera frame
        x_pos_camera_1 = x_pos_ego - [1.35, 1.7, 2.4]
        x_pos_camera_2 = x_pos_ego - [1.35, -1.7, 2.4]

        # correct the camera angle
        cam_rot = inv(RotY(0.02))
        x_pos_camera_1 = cam_rot * x_pos_camera_1
        x_pos_camera_2 = cam_rot * x_pos_camera_2

        # switch to rotated camera frame
        R = [0 0 1.;
            -1 0 0;
            0 -1 0]
        x_pos_camera_1 = R * x_pos_camera_1
        x_pos_camera_2 = R * x_pos_camera_2

        if (x_pos_camera_1[3] < focal_len || x_pos_camera_2[3] < focal_len)
            break
        end

        # switch to image frame (y1, y2)
        x_pos_img_1 = [x_pos_camera_1[1]/x_pos_camera_1[3] * focal_len, x_pos_camera_1[2]/x_pos_camera_1[3] * focal_len]
        x_pos_img_2 = [x_pos_camera_2[1]/x_pos_camera_2[3] * focal_len, x_pos_camera_2[2]/x_pos_camera_2[3] * focal_len]
        
        # store the result into two arrays
        push!(corners_img_1, x_pos_img_1)
        push!(corners_img_2, x_pos_img_2)
    end

    # convert 3d bounding box (now in image frame) into 2d bounding box.
    for (corners, cam) in zip((corners_img_1, corners_img_2), (z1, z2))
        # find the top, left, bottom, right points.
        for corner in corners
            if corner[3] < focal_len
                break
            end
            px = focal_len*corner[1]/corner[3]
            py = focal_len*corner[2]/corner[3]
            left = min(left, px)
            right = max(right, px)
            top = min(top, py)
            bot = max(bot, py)
        end
        # convert these four points into pixel position.
        if top ≈ bot || left ≈ right || top > bot || left > right
            # out of frame
            continue
        else 
            top = convert_to_pixel(image_height, pixel_len, top)
            bot = convert_to_pixel(image_height, pixel_len, bot)
            left = convert_to_pixel(image_width, pixel_len, left)
            top = convert_to_pixel(image_width, pixel_len, right)
            push!(cam, SVector(top, left, bot, right))
        end
    end
    # z1 and z2 contain 2d bbox for each camera respectively.
    [z1 z2]
end

