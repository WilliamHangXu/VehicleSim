using Hungarian

"""
This is the perception module of our autonomous vehicle.
We use Extended Kalman Filter (EKF) to estimate the state of objects (other cars) detected by the 
cameras.

Outline of EKF:
    a. Previous measurement: P(xₖ₋₁ | Z₁:ₖ₋₁) = 𝒩(μₖ₋₁, Σₖ₋₁)
    b. Process model: P(xₖ | xₖ₋₁) = 𝒩(f(xₖ₋₁), Σₚ)
    c. Measurement model: P(zₖ | xₖ) = 𝒩(h(xₖ, x_egoₖ), Σₘ)
    where x = [p1, p2, θ, v, l, w, h] describes the state of an object, 
    and z = [y1, y2, y3, y4] describes the measurement (bb) collected by the camera.
"""

"""
# Modified from get_body_transform in measurement.jl by William
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
# Modified from get_3d_bbox_corners in measurement.jl by William
This function calculates the coordinates of the 8 corners of the 3D bounding box describing the
object.
@param
position: position of the object ([x,y,z]) in world frame
θ: heading of the object
box_size: size of bounding box
@output
An array of 8 points ([x,y,z]). These are the 8 corners of the 3D bounding box describing an object.
"""
function get_3d_bbox_corners_perception(position, θ, box_size)
    d1 = cos(θ) * box_size[1] - sin(θ) * box_size[2]
    d2 = sin(θ) * box_size[1] + cos(θ) * box_size[2]
    d3 = box_size[3]
    T = get_body_transform_perception(position)
    corners = []
    for dx in [-d1/2, d1/2]
        for dy in [-d2/2, d2/2]
            for dz in [-d3/2, d3/2]
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
This function calculates the jacobian of the f function above. Created by William
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
id: camera id (1 or 2)
x: the state of the object being tracked [p1, p2, θ, v, l, w, h]
x_ego: the state of ego vehicle [position; quaternion; velocity; angular_vel]
@output
z: bounding box measurement of the object being tracked [y1, y2, y3, y4]
index: an array of size 4 that contains the index of 3d bbox corners that contributes to the 
       corresponding value of y1, y2, y3, y4 for camera id.
rot: A matrix that changes a point from world frame into rotated camera frame.
rot_coord: an array of the coordinate of the 3d bbox corner expressed in rotated camera frame for 
           camera id. These corners are the corners stored in index.
"""
function h(id, x, x_ego, focal_len=0.01, pixel_len=0.001, image_width=640, image_height=480)

    # the position of the ego vehicle in world frame
    x_ego_pos_world = x_ego[1:3]      # p1, p2, p3
    
    # the position of the object in world frame
    x_pos_world = [x[1], x[2], 2.645] # p1, p2, p3
    # the angle of the ego vehicle
    x_ego_angles = x_ego[4:7]         # quaternion
    # the size of object
    x_size = x[5:7]                   # l, w, h
    # this stores the 2d bounding box coordinates
    z = []   

    # corners_world is an array that contains coordinates of 3d bbox of the object in world frame
    corners_world = get_3d_bbox_corners_perception(x_pos_world, x[3], x_size)

    # This part takes care of the camera's angle and its relative position w.r.t. ego.
    T_body_cam = get_cam_transform(id)

    # This part changes the camera frame into the rotated camera frame, where z points forward.
    T_cam_camrot = get_rotated_camera_transform()

    # This part combines the two transformations together.
    T_body_camrot = multiply_transforms(T_body_cam, T_cam_camrot)

    # This part takes care of ego's rpy angles and the object's relative position w.r.t. ego.
    T_world_body = get_body_transform(x_ego_angles, x_ego_pos_world)                

    # Combines all the transformations together (order matters)
    T_world_camrot = multiply_transforms(T_world_body, T_body_camrot) 

    # We need to invert the transformations above to get the correct order.
    # change to ego frame -> adjust rpy angle -> change to camera frame -> adjust cam angle -> 
    # change to rotated camera frame
    T_camrot_world = invert_transform(T_world_camrot)

    # initialize the boundaries of the 2d bbox
    left = image_width/2
    right = -image_width/2
    top = image_height/2
    bot = -image_height/2
    
    # apply the transformation to points in corners_world to convert them into rotated camera frame
    vehicle_corners = [T_camrot_world * [pt;1] for pt in corners_world]

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
    right = convert_to_pixel(image_width, pixel_len, right)

    # 2d bbox information
    z = SVector(top, left, bot, right)
    
    [z, index, T_camrot_world, rot_coord]
end

"""
This function calculates the jacobian of h function. Created by William
@param
x: the input of h.
h: The output of h
focal_len: focal length of the camera
pixel_len: size of a pixel
@output
The jacobian of h. Should be a 4*7 matrix.
"""
function jac_h(x, h, focal_len=0.01, pixel_len=0.001)
    jac = []

    # j4 is the jacobian of the matrix that converts image frame into pixel values.
    j4 = [1/pixel_len 0
    0 1/pixel_len]
    
    for j = 1:4

        # j1 is the jacobian of the matrix that calculates world-frame coordinates of corners
        # of 3d bbox.
        j1 = jac_h_j1(x, h[2][j])

        # j2 is the jacobian of the matrix that converts world-frame coordinates into rotated
        # camera frame.
        j2 = h[3][:, 1:3]

        # These are the xyz coordinates of a corner
        c1 = h[4][j][1]
        c2 = h[4][j][2]
        c3 = h[4][j][3]

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
    jac
end

"""
The h function above can be broken down into 4 matrix operations. This function calculates the jacobian
of the first matrix operation (generating bbox corners). Created by William
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
function jac_h_j1(x, index)
   
    θ = x[3]
    l = x[5]
    w = x[6]
    
    # This function turns index-1 into a 3-digit binary number. (i.e. 7->111)
    # These numbers indicate the positive and negative signs for the calculation of the 8 corners.
    ex = digits(index-1, base=2, pad=3)
    
    ex += [1;1;1]
    
    [
    1 0 0.5*(-1)^(ex[3]+1)*(sin(θ)*l+cos(θ)*w) 0 0.5*(-1)^ex[3]*cos(θ) 0.5*(-1)^(ex[3]+1)*sin(θ) 0
    0 1 0.5*(-1)^ex[2]*(cos(θ)*l-sin(θ)*w) 0 0.5*(-1)^ex[2]*sin(θ) 0.5*(-1)^(ex[2])*cos(θ) 0
    0 0 0 0 0 0 0.5+0.5*(-1)^ex[1]
    ]
end

"""
Given the coordinates of two 2d bboxes, calculate the ratio of the area of intersection and union.
This is know as IOU. IOU < 1. Large IOU value indicates that the two bboxes overlap a lot.
EXTREMELY stupid implementation. Created by William
(xa1, ya1)-------
|               |
|               |
|               |
|               |
|               |
---------(xa2, ya2)
"""
function iou_bb(bb_a, bb_b)
    
    ya1, xa1, ya2, xa2 = bb_a
    yb1, xb1, yb2, xb2 = bb_b
    
    intersection = -1
    if xa1 <= xb1 && ya1 <= yb1
        if xa2 >= xb1 && xa2 < xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xa2-xb1)*(ya2-yb1)
        elseif xa2 >= xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xb2-xb1)*(ya2-yb1)
        elseif xa2 < xb2 && xa2 >= xb1 && ya2 >= yb2
            intersection = (xa2-xb1)*(yb2-yb1)
        elseif xa2 >= xb2 && ya2 >= yb2
            intersection = (xb2-xb1)*(yb2-yb1)
        else
            intersection = -1
        end
    elseif xa1 <= xb2 && xa1 > xb1 && ya1 <= yb1
        if xa2 >= xb1 && xa2 < xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xa2-xa1)*(ya2-yb1)
        elseif xa2 >= xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xb2-xa1)*(ya2-yb1)
        elseif xa2 < xb2 && xa2 >= xb1 && ya2 >= yb2
            intersection = (xa2-xa1)*(yb2-yb1)
        elseif xa2 >= xb2 && ya2 >= yb2
            intersection = (xb2-xa1)*(yb2-yb1)
        else
            intersection = -1
        end
    elseif xa1 <= xb1 && ya1 <= yb2 && ya1 < yb1
        if xa2 >= xb1 && xa2 < xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xa2-xb1)*(ya2-ya1)
        elseif xa2 >= xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xb2-xb1)*(ya2-ya1)
        elseif xa2 < xb2 && xa2 >= xb1 && ya2 >= yb2
            intersection = (xa2-xb1)*(yb2-ya1)
        elseif xa2 >= xb2 && ya2 >= yb2
            intersection = (xb2-xb1)*(yb2-ya1)
        else
            intersection = -1
        end
    elseif xa1 > xb1 && xa1 <= xb2 && ya1 > yb1 && ya1 <= yb2
        if xa2 >= xb1 && xa2 < xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xa2-xa1)*(ya2-ya1)
        elseif xa2 >= xb2 && ya2 >= yb1 && ya2 < yb2
            intersection = (xb2-xa1)*(ya2-ya1)
        elseif xa2 < xb2 && xa2 >= xb1 && ya2 >= yb2
            intersection = (xa2-xa1)*(yb2-ya1)
        elseif xa2 >= xb2 && ya2 >= yb2
            intersection = (xb2-xa1)*(yb2-ya1)
        else
            intersection = -1
        end
    else
        intersection = -1
    end
    
    union = (xa2-xa1)*(ya1-ya2) + (xb2-xb1)*(yb1-yb2) - intersection
    return intersection/union
end

"""
The purpose of this function is to match bounding boxes in camera measurement k with those in camera
measurement k-1. An external package, Hungarian, is used. Created by William.
https://github.com/Gnimuc/Hungarian.jl
@param
camera_id: camera id
prev_state: an array of states. These are the output of the previous run of EKF (the states of objects
            described by bounding boxes in camera measurement k-1)
bb: bounding boxes in camera measurement k
x_ego: the state of our vehicle when camera measurement k-1 was collected
iou_thr: threashold of iou value. Notice that small iou value indicates poor relation. If the iou value
         is less than the threshold, we do not consider two bounding boxes to be related at all.
Δ: time difference between camera measurement k-1 and camera measurement k.
@output
An array. array[i] is the index of bbox in camera measurement k-1 that can be matched with the ith 
bbox in camera measurement k. If array[i] == 0, then it is likely that the ith bbox describes a vehicle
that has just entered the vision of our vehicle, and we need to start a new EKF for it.
"""
function assign_bb(camera_id, prev_state, bb, x_ego, Δ, iou_thr=0.5)

    # From the previous result of EKF (the estimated state of the object in the bboxes from camera 
    # measurement k-1), we use our f function and h function to predict the bbox coordinates of these 
    # object in camera measurement k. Then, we try to pair these predicted bboxes with the real bboxes
    # obtained in camera measurement k.
    # This array contains the predicted bboxes.
    bb_p = []
    for i in prev_state
        state_p = f(i, Δ)       
        h_p = h(camera_id, state_p, x_ego)
        push!(bb_p, h_p[1])
        
    end

    # This is the cost matrix of the Hungarian Algorithm.
    cost = []
    for i in bb_p
        temp = []
        for j in bb
            iou  = iou_bb(i, j)
            if iou < iou_thr  
                push!(temp, missing)
            else          
                push!(temp, 1-iou)
            end
        end
        push!(cost, temp)
    end

    # Convert the cost matrix into a format that the Hungarian package can use.
    cost_mat = hcat([i for i in cost]...)
    
    if eltype(cost_mat) == Missing
        return zero(length(bb))
    else

        # Runs the Hungarian Algorithm
        assignment, cost = Hungarian.hungarian(cost_mat)
        
        # index is the output array.
        index = zeros(length(bb))
        for i in eachindex(assignment)
            if assignment[i] != 0
                index[assignment[i]] = i
            end
        end
        return index
    end    
end