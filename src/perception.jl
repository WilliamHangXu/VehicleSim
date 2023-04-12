include("measurements.jl")

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
R: rotation matrix
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
This function predicts current state based on previous state. Created by William
@param
x: previous state [p1, p2, θ, v, l, w, h]
Δ: time step
@output
current state
"""
function f(x, Δ)
    [
        x[1] + Δ * x[4] * cos(x[3])
        x[2] + Δ * x[4] * sin(x[3])
        x[3]
        x[4]
        x[5]
        x[6]
        x[7]
    ]
end

"""
This function calculates the jacobian of the f function above.
@param
x: previous state
Δ: time step
@output
The jacobian
"""
function jac_f(x, Δ)
    [
        1 0 -sin(x[3])*Δ*x[4] Δ*cos(x[3]) 0 0 0
        0 1 cos(x[3])*Δ*x[4] Δ*sin(x[3]) 0 0 0
        0 0 1 0 0 0 0
        0 0 0 1 0 0 0
        0 0 0 0 1 0 0
        0 0 0 0 0 1 0
        0 0 0 0 0 0 1
    ]
end

"""
# Modified from cameras in measurement.jl by Gavin and William
This function predicts bounding box measurements based on the state of the object x being tracked.
@param
x: the state of the object being tracked [p1, p2, θ, v, l, w, h]
x_ego: the state of ego vehicle # TODO figure out what this looks like
@output
z: bounding box measurement of the object being tracked [y1, y2, y3, y4, y5, y6, y7, y8]
indices: [[index_1][index_2]], where index_n is an array of size 4 that contains the index of 3d bbox 
         corners that contributes to the corresponding value of y1, y2, y3, y4 for camera n.
rot: This is an array of two matrices. They change a point from world frame into rotated camera frame
     for each of the cameras respectively.
rot_cam: [[rot_coord_1][rot_coord_2]], where rot_coord_n is and array of the coordinate of the 3d bbox 
         corner expressed in rotated camera frame for camera n. These corners are the corners stored 
         in indices.
"""
function h(x, x_ego, focal_len=0.01, pixel_len=0.001, image_width=640, image_height=480)

    # the position of the ego vehicle in world frame
    x_ego_pos_world = x_ego[1:3]      # p1, p2, p3
    # the position of the object in world frame
    x_pos_world = [x[1], x[2], 2.645] # p1, p2, p3
    # the angle of the ego vehicle
    x_ego_angles = x_ego[4:6]         # r, p, y
    # the size of object
    x_size = x[5:7]                   # l, w, h
    # this stores the 2d bounding box coordinates
    z = []
    # this array stores two index arrays. See below.
    indices = []
    # this array stores two rot_coord arrays. See below.
    rot_cam = []

    # corners_world is an array that contains coordinates of 3d bbox of the object in world frame
    corners_world = get_3d_bbox_corners_perception(x_pos_world, x_size)

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
    # change to ego frame -> adjust rpy angle -> change to camera frame -> adjust cam angle -> 
    # change to rotated camera frame
    T_camrot1_world = invert_transform(T_world_camrot1)
    T_camrot2_world = invert_transform(T_world_camrot2)

    # This is the rot that is part of the return values.
    rot = [T_camrot1_world, T_camrot2_world]

    for transform in (T_camrot1_world, T_camrot2_world)

        # initialize the boundaries of the 2d bbox
        left = image_width/2
        right = -image_width/2
        top = image_height/2
        bot = -image_height/2
        
        # apply the transformation to points in corners_world to convert them into rotated camera frame
        vehicle_corners = [transform * [pt;1] for pt in corners_world]

        # Keep track of the index of bbox corner in the loop below.
        num = 1

        # This array tells us which corners contribute to the values of top, left, bottom, right, respectively
        index = [1,1,1,1]

        # This array contains the coordinates of the points stored in index, in rotated camera frame
        rot_coord = []

        for corner in vehicle_corners
            if corner[3] < focal_len
                break
            end
            px = focal_len*corner[1]/corner[3]
            py = focal_len*corner[2]/corner[3]
            left_temp = left
            right_temp = right
            top_temp = top
            bot_temp = bot
            left = min(left, px)
            right = max(right, px)
            top = min(top, py)
            bot = max(bot, py)

            # Update index. The code above finds the min/max value for x/y. When we find a new min/max,
            # the value (left, right, top, bot) changes. If it changes, we update index so that we
            # know which corner contributes to this new value.
            if top != top_temp
                index[1] = num
            end
            if left != left_temp
                index[2] = num
            end
            if bot != bot_temp
                index[3] = num
            end
            if right != right_temp
                index[4] = num
            end
            num += 1
        end

        # Add the corresponding coordinates (rotated camera frame) into rot_coord.
        for j in index
            push!(rot_coord, vehicle_corners[j])
        end

        # convert image frame coordinates into pixel numbers
        top = convert_to_pixel(image_height, pixel_len, top) # top 0.00924121388699952 => 251
        bot = convert_to_pixel(image_height, pixel_len, bot)
        left = convert_to_pixel(image_width, pixel_len, left)
        top = convert_to_pixel(image_width, pixel_len, right)

        push!(z, SVector(top, left, bot, right))
        push!(indices, index)
        push!(rot_cam, rot_coord)
    end
    [z, indices, rot, rot_cam]
end

"""
This function calculates the jacobian of h function.
@param
h: The output of h
focal_len: focal length of the camera
pixel_len: size of a pixel
@output
The jacobian of h. Should be a 8*7 matrix.
"""
function jac_h(h, focal_len=0.01, pixel_len=0.001)
    jac = []

    # j4 is the jacobian of the matrix that converts image frame into pixel values.
    j4 = [1/pixel_len 0
    0 1/pixel_len]

    # two cameras, each camera's 2d bbox has 4 values.
    for i = 1:2
        for j = 1:4

            # j1 is the jacobian of the matrix that calculates world-frame coordinates of corners
            # of 3d bbox.
            j1 = jac_h_j1(h[2][i][j])

            # j2 is the jacobian of the matrix that converts world-frame coordinates into rotated
            # camera frame.
            j2 = h[3][i]

            # These are the xyz coordinates of a corner
            c1 = h[4][i][j][1]
            c2 = h[4][i][j][2]
            c3 = h[4][i][j][3]

            # j3 is the jacobian of the matrix that converts coordinates from rotate camera frame
            # into image frame (y1, y2)
            j3 = [
                focal_len/c3 0 -focal_len*c1*c3^(-2)
                0 focal_len/c3 -focal_len*c2*c3^(-2)
            ]

            # Use chain rule to get the overall jacobian of the entire function. This should be 2*7
            j = j4 * j3 * j2 * j1

            # If we are dealing with top or bottom, we only need the second row of the jacobian, 
            # because only the y value of the point contributes to top/bottom. Similarly, if we are
            # dealing with left or right, we only need the first row of the jacobian.
            if j == 1 || j == 3
                push!(jac, j[2,:]')
            else
                push!(jac, j[1,:]')
            end            
        end
    end
    jac
end

"""
The h function above can be broken down into 4 matrix operations. This function calculates the jacobian
of the first matrix operation (generating bbox corners).
@param 
index: the index of the corner that we want. (We are looking down to the 1357 plane)
     7------5
    /|     /|
   / |    / |
  /  8---/--6            z  x
 /  /   /  /             | /
3------1  /          y___|/
| /    | /
|/     |/
4------2
@output
The jacobian.
"""
function jac_h_j1(index)
    # This function turns index-1 into a 3-digit binary number. (i.e. 7->111)
    exp = digits(index-1, base=2, pad=3)
    exp += [1 1 1]
    [
    1 0 0 0 0.5*(-1)^exp[3] 0 0
    0 1 0 0 0 0.5*(-1)^exp[2] 0
    0 0 0 0 0 0 0.5*(-1)^exp[1]        
    ]
end
