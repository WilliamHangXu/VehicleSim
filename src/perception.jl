"""
This is the perception module of our autonomous vehicle.
We use Extended Kalman Filter (EKF) to estimate the state of objects (other cars) detected by the 
cameras.

Outline of EKF:
    a. Previous measurement: P(xₖ | xₖ₋₁) = 𝒩(μₖ₋₁, Σₖ₋₁)
    b. Process model: P(xₖ | xₖ₋₁) = 𝒩(f(xₖ₋₁), Σₚ)
    c. Measurement model: P(zₖ | xₖ) = 𝒩(h(xₖ, x_egoₖ), Σₘ)
    where x = [p1, p2, θ, v, l, w, h] describes the state of an object, 
    and z = [y1, y2, y3, y4, y5, y6, y7, y8] describes the measurement (bb) collected by the two 
    cameras.

"""

"""
# Modified from get_body_transform in measurement.jl by Gavin and William
This function calculates a matrix that expresses a point in a loc-centered frame into world frame.

@param
loc: the coordinate of a point.
@output
a matrix.
"""
function get_body_transform_perception(loc, R=one(RotMatrix{3, Float64}))
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
This function predicts current state based on previous state. Created by William

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
# Modified from cameras in measurement.jl by Gavin and William
This function predicts bounding box measurements based on the state of the object x being tracked.

@param
x: the state of the object being tracked
x_ego: the state of ego vehicle

@output
bounding box measurement of the object being tracked
"""
function h(x, x_ego, focal_len=0.01, pixel_len=0.001, image_width=640, image_height=480)
    # the position of the ego vehicle in world frame
    x_ego_pos_world = x_ego[1:3]      # p1, p2, p3
    # the position of the object in world frame
    x_pos_world = x[1:3]              # p1, p2, p3
    # the angle of the ego vehicle
    x_ego_angles = x_ego[4:6]         # r, p, y
    # the size of object
    x_size = x[6:8]                   # l, w, h
    # this stores the 2d bounding box coordinates
    z = []

    # corners_world is an array that contains coordinates of 3d bbox of the object in world frame
    corners_world = get_3d_bbox_corners_perception(x_pos_world, x_size)

    # initialize the boundaries of the 2d bbox
    left = image_width/2
    right = -image_width/2
    top = image_height/2
    bot = -image_height/2

    # This part takes care of the camera's angle and its relative position w.r.t. ego.
    T_body_cam1 = get_cam_transform(1)
    T_body_cam2 = get_cam_transform(2)

    # This part changes the camera frame into the rotated camera frame, where z points forward.
    T_cam_camrot = get_rotated_camera_transform()

    # This part combines the two transformations together.
    T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
    T_body_camrot2 = multiply_transforms(T_body_cam2, T_cam_camrot)

    # This part takes care of ego's rpy angles and the object's relative position w.r.t. ego.
    T_world_body = get_body_transform_perception(x_ego_pos_world, RotXYZ(x_ego_angles))

    # Combines all the transformations together (order matters)
    T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1) 
    T_world_camrot2 = multiply_transforms(T_world_body, T_body_camrot2)

    # We need to invert the transformations above to get the correct order.
    # change to ego frame -> adjust rpy angle -> change to camera frame -> adjust cam angle -> change to rotated camera frame
    T_camrot1_world = invert_transform(T_world_camrot1)
    T_camrot2_world = invert_transform(T_world_camrot2)

    for transform in (T_camrot1_world, T_camrot2_world)
        # apply the transformation to points in corners_world to convert them into rotated camera frame
        vehicle_corners = [transform * [pt;1] for pt in corners_world]

        # convert the points from rotated camera frame into image frame (y1, y2)
        for corner in vehicle_corners
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

        # convert image frame coordinates into pixel numbers
        top = convert_to_pixel(image_height, pixel_len, top) # top 0.00924121388699952 => 251
        bot = convert_to_pixel(image_height, pixel_len, bot)
        left = convert_to_pixel(image_width, pixel_len, left)
        top = convert_to_pixel(image_width, pixel_len, right)

        push!(z, SVector(top, left, bot, right))
    end
    z
end

