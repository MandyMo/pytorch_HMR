
import h5py
import torch
import numpy as np
from config import args
import json
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

def load_mean_theta():
    mean = np.zeros(args.total_theta_count, dtype = np.float)

    mean_values = h5py.File(args.smpl_mean_theta_path)
    mean_pose = mean_values['pose']
    mean_pose[:3] = 0
    mean_shape = mean_values['shape']
    mean_pose[0]=np.pi

    #init sacle is 0.9
    mean[0] = 0.9

    mean[3:75] = mean_pose[:]
    mean[75:] = mean_shape[:]

    return mean

def batch_rodrigues(theta):
    #theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim = 1)
        return torch.cat([R_homo, t_homo], 2)
    
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).cuda()], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A


def batch_lrotmin(theta):
    theta = theta[:,3:].contiguous()
    Rs = batch_rodrigues(theta.view(-1, 3))
    print(Rs.shape)
    e = Variable(torch.eye(3).float())
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)

def batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
    '''
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    return (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)

def calc_aabb(ptSets):
    if not ptSets or len(ptSets) == 0:
        return False, False, False
    
    ptLeftTop     = np.array([ptSets[0][0], ptSets[0][1]])
    ptRightBottom = ptLeftTop.copy()
    for pt in ptSets:
        ptLeftTop[0]     = min(ptLeftTop[0], pt[0])
        ptLeftTop[1]     = min(ptLeftTop[1], pt[1])
        ptRightBottom[0] = max(ptRightBottom[0], pt[0])
        ptRightBottom[1] = max(ptRightBottom[1], pt[1])

    return ptLeftTop, ptRightBottom, len(ptSets) >= 5

def get_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center = None):
    if Center == None:
        Center = (leftTop + rightBottom) // 2

    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    cx = int(cx * ExpandsRatio)
    cy = int(cy * ExpandsRatio)
    
    r = max(cx, cy)

    cx = r
    cy = r
    
    x = int(Center[0])
    y = int(Center[1])

    return [x - cx, y - cy], [x + cx, y + cy]

def shrink(leftTop, rightBottom, width, height):
    xl = -leftTop[0]
    xr = rightBottom[0] - width

    yt = -leftTop[1]
    yb = rightBottom[1] - height

    cx = (leftTop[0] + rightBottom[0]) / 2
    cy = (leftTop[1] + rightBottom[1]) / 2

    r = (rightBottom[0] - leftTop[0]) / 2

    sx = max(xl, 0) + max(xr, 0)
    sy = max(yt, 0) + max(yb, 0)

    if (xl <= 0 and xr <= 0) or (yt <= 0 and yb <=0):
        return leftTop, rightBottom
    elif leftTop[0] >= 0 and leftTop[1] >= 0 : # left top corner is in box
        l = min(yb, xr)
        r = r - l / 2
        cx = cx - l / 2
        cy = cy - l / 2
    elif rightBottom[0] <= width and rightBottom[1] <= height : # right bottom corner is in box
        l = min(yt, xl)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy + l / 2
    elif leftTop[0] >= 0 and rightBottom[1] <= height : #left bottom corner is in box
        l = min(xr, yt)
        r = r - l  / 2
        cx = cx - l / 2
        cy = cy + l / 2
    elif rightBottom[0] <= width and leftTop[1] >= 0 : #right top corner is in box
        l = min(xl, yb)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy - l / 2
    elif xl < 0 or xr < 0 or yb < 0 or yt < 0:
        return leftTop, rightBottom
    elif sx >= sy:
        sx = max(xl, 0) + max(0, xr)
        sy = max(yt, 0) + max(0, yb)
        # cy = height / 2
        if yt >= 0 and yb >= 0:
            cy = height / 2
        elif yt >= 0:
            cy = cy + sy / 2
        else:
            cy = cy - sy / 2
        r = r - sy / 2
        
        if xl >= sy / 2 and xr >= sy / 2:
            pass
        elif xl < sy / 2:
            cx = cx - (sy / 2 - xl)
        else:
            cx = cx + (sy / 2 - xr)
    elif sx < sy:
        cx = width / 2
        r = r - sx / 2
        if yt >= sx / 2 and yb >= sx / 2:
            pass
        elif yt < sx / 2:
            cy = cy - (sx / 2 - yt)
        else:
            cy = cy + (sx / 2 - yb)
        

    return [cx - r, cy - r], [cx + r, cy + r]

'''
    offset the keypoint by leftTop
'''
def off_set_pts(keyPoints, leftTop):
    result = keyPoints.copy()
    result[:, 0] -= leftTop[0]
    result[:, 1] -= leftTop[1]
    return result

'''
    cut the image, by expanding a bounding box
'''
def cut_image(filePath, kps, expand_ratio, leftTop, rightBottom):
    originImage = cv2.imread(filePath)
    height       = originImage.shape[0]
    width        = originImage.shape[1]
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1

    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, expand_ratio)
    leftTop, rightBottom = shrink(leftTop, rightBottom, width, height)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop      = [int(leftTop[0]), int(leftTop[1])]
    rightBottom  = [int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)]

    dstImage = np.zeros(shape = [rightBottom[1] - leftTop[1], rightBottom[0] - leftTop[0], channels], dtype = np.uint8)
    dstImage[:,:,:] = 30

    offset = [lt[0] - leftTop[0], lt[1] - leftTop[1]]
    size   = [rb[0] - lt[0], rb[1] - lt[1]]

    dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    return dstImage, off_set_pts(kps, leftTop)

'''
    purpose:
        flip a image given by src_image and the 2d keypoints
    flip_mode: 
        0: horizontal flip
        >0: vertical flip
        <0: horizontal & vertical flip
'''

def flip_image(src_image, kps, flip_mode):
    h, w = src_image.shape[0], src_image.shape[1]

    if flip_mode == 0:
        src_image = cv2.flip(src_image, 0)
        if kps is not None:
            kps[:, 1] = h - 1 - kps[:, 1]
            kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
            kps[:, :] = kps[kp_map]
    elif flip_mode > 0:
        src_image = cv2.flip(src_image, 1)
        if kps is not None:
            kps[:, 0] = w - 1 - kps[:, 0]
            kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
            kps[:, :] = kps[kp_map]
    else:
        src_image = cv2.flip(src_image, -1)
        if kps is not None:
            kps[:, 0] = w - 1 - kps[:, 0]
            kps[:, 1] = h - 1 - kps[:, 1]
    return src_image

'''
    src_image: h x w x c
    pts: 14 x 3
'''
def draw_lsp_14kp__bone(src_image, pts):
        bones = [
            [0, 1, 255, 0, 0],
            [1, 2, 255, 0, 0],
            [2, 12, 255, 0, 0],
            [3, 12, 0, 0, 255],
            [3, 4, 0, 0, 255],
            [4, 5, 0, 0, 255],
            [12, 9, 0, 0, 255],
            [9,10, 0, 0, 255],
            [10,11, 0, 0, 255],
            [12, 8, 255, 0, 0],
            [8,7, 255, 0, 0],
            [7,6, 255, 0, 0],
            [12, 13, 0, 255, 0]
        ]

        for pt in pts:
            if pt[2] > 0.2:
                cv2.circle(src_image,(int(pt[0]), int(pt[1])),2,(0,255,255),-1)
        
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
            if pa[2] > 0.2 and pb[2] > 0.2:
                cv2.line(src_image,(xa,ya),(xb,yb),(line[2], line[3], line[4]),2)  

'''
    return whether two segment intersect
'''

def line_intersect(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True

'''
    return whether two rectangle intersect
    ra, rb left_top point, right_bottom point
'''
def rectangle_intersect(ra, rb):
    ax = [ra[0][0], ra[1][0]]
    ay = [ra[0][1], ra[1][1]]

    bx = [rb[0][0], rb[1][0]]
    by = [rb[0][1], rb[1][1]]

    return line_intersect(ax, bx) and line_intersect(ay, by)

def get_intersected_rectangle(lt0, rb0, lt1, rb1):
    if not rectangle_intersect([lt0, rb0], [lt1, rb1]):
        return None, None

    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = max(lt[0], lt1[0])
    lt[1] = max(lt[1], lt1[1])

    rb[0] = min(rb[0], rb1[0])
    rb[1] = min(rb[1], rb1[1])
    return lt, rb

def get_union_rectangle(lt0, rb0, lt1, rb1):
    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = min(lt[0], lt1[0])
    lt[1] = min(lt[1], lt1[1])

    rb[0] = max(rb[0], rb1[0])
    rb[1] = max(rb[1], rb1[1])
    return lt, rb

def get_rectangle_area(lt, rb):
    return (rb[0] - lt[0]) * (rb[1] - lt[1])

def get_rectangle_intersect_ratio(lt0, rb0, lt1, rb1):
    (lt0, rb0), (lt1, rb1) = get_intersected_rectangle(lt0, rb0, lt1, rb1), get_union_rectangle(lt0, rb0, lt1, rb1)

    if lt0 is None:
        return 0.0
    else:
        return 1.0 * get_rectangle_area(lt0, rb0) / get_rectangle_area(lt1, rb1)

def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))
    
    if normalize:
        src_image = src_image.astype(np.float) / 255
    
    return src_image

'''
    align ty pelvis
    joints: n x 14 x 3, by lsp order
'''
def align_by_pelvis(joints):
    left_id = 3
    right_id = 2
    pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
    return joints - torch.unsqueeze(pelvis, dim=1)


if __name__ == '__main__':
    na = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype = np.float)
    va = Variable(torch.from_numpy(na.reshape(-1, 2, 3)))
    
    nb =  np.array([3, 2, 1, 1, 2, 3], dtype = np.float)
    vb = Variable(torch.from_numpy(nb.reshape(2, 1, 3)))

    print(va)
    print(vb)
    print(va - vb)
