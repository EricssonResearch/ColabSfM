import numpy as np
import os
from glob import glob
from PIL import Image
import pykitti

from colabsfm.utils import to_homogeneous, from_homogeneous

# Change this to the directory where you store KITTI data

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.
    return depth


class DepthScene:
    def __init__(self, basedir = 'data/kitti', split = "train", date = '2011_09_26', drive = '0001') -> None:
        self.depth_folder = os.path.join(basedir,split,f"{date}_drive_{drive}_sync", "proj_depth", "groundtruth", "image_02")
        self.depth_paths = sorted(glob(f"{self.depth_folder}/*"))
    def __getitem__(self, idx):
        print(self.depth_paths[idx])
        return depth_read(self.depth_paths[idx])
    def __len__(self):
        return len(self.depth_paths)


class KittiScene:
    def __init__(self, basedir = 'data/kitti', date = '2011_09_26', drive = '0001') -> None:
        self.depths = DepthScene(basedir=basedir, date = date, drive = drive, split = "train")
        self.dataset = pykitti.raw(os.path.join(basedir,"raw"), date, drive)
        self.lidar_to_cam2 = self.dataset.calib.T_cam2_velo
        self.intrinsics = self.dataset.calib.K_cam2
        self.inv_intrinsics = np.linalg.inv(self.intrinsics)
    
    def __len__(self):
        return len(self.depths)
    
    def depth_to_pointcloud(self, depth):
        H, W = depth.shape
        grid = np.meshgrid(np.arange(W), np.arange(H))
        grid = np.stack((grid[0], grid[1]), axis = -1)
        bearing_vectors = depth[...,None] * (self.inv_intrinsics @ to_homogeneous(grid)[...,None])[...,0] 
        return bearing_vectors[depth > 0]
    
    def __getitem__(self, idx):
        OFFSET = 5 # depths start at 5
        depth = self.depths[idx]
        print(self.dataset.cam2_files[idx+OFFSET])
        lidar = self.dataset.get_velo(idx + OFFSET)
        rgb = self.dataset.get_cam2(idx + OFFSET)
        depth_xyz = self.depth_to_pointcloud(depth)
        lidar = lidar[::10]
        lidar_cam_coords = from_homogeneous((self.lidar_to_cam2 @ to_homogeneous(lidar[...,:3])[...,None])[...,0])
        depth_xyz = depth_xyz[::10]
        return {"lidar": lidar, "depth-xyz": depth_xyz, "lidar-cam": lidar_cam_coords, "rigid_transform": self.lidar_to_cam2}


def build_kitti_dataset(basedir):
    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0001'
    return KittiScene(basedir = basedir, date = date, drive = drive)


if __name__ == "__main__":
    kitti_data = build_kitti_dataset("data/kitti")
    batch = kitti_data[0]
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    lidar = batch["lidar"][...,:3]
    depth = batch["depth-xyz"]
    np.save("vis/lidar",lidar)
    np.save("vis/depth",depth)
    
    f = plt.figure()
    ax = f.add_subplot(111, projection = '3d')
    ax.scatter(lidar[::20, 0],
            lidar[::20, 2],
            lidar[::20, 1],
            c=lidar[::20, 2],
            s = 0.5,
            cmap='jet')
    plt.savefig("lidar")
    plt.close()
    f = plt.figure()
    ax = f.add_subplot(111, projection = '3d')
    ax.scatter(depth[:, 0],
            depth[:, 2],
            depth[:, 1],
            c=depth[:, 2],
            s = 0.5,
            cmap='jet')
    plt.savefig("depth")
