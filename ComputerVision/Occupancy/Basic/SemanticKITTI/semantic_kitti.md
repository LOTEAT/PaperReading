<!--
 * @Author: LOTEAT
 * @Date: 2024-09-06 13:37:20
-->
## Semantic KITTI

### 1.组织格式
Semantic KITTI数据格式如下：
```shell
dataset
├── sequences
│   ├── 00
│   │   ├── xxx.bin
│   │   ├── xxx.invalid
│   │   ├── xxx.label
│   │   ├── xxx.occluded
```
### 2.文件格式
- bin：voxel文件，类型为np.uint8。读取后，需要进一步解压。解压代码如下：
```python
def unpack(compressed):
    """given a bit encoded voxel grid, make a normal voxel grid out of it."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed
```
这是因为bin将8个voxel保存为一个8bit的数值，而voxel刚好是用0-1表示占用和未占用的。在Semantic KITTI数据中，体素空间大小为(256, 256, 32)，所以读取后还需要reshape。
- label：label文件夹，类型为np.uint16。与体素空间一一对应，也需要reshape为(256, 256, 32)。
- invalid：用于标记lidar无论从什么角度都无法检测出的voxel，其实也就是物体内部。类型与处理方式和bin文件一致。
- occluded：用于标记当前lidar检测不到，但是实际被占据的voxel。这些voxel只要lidar换角度就可以检测到。处理方式和类型与bin文件一致。
### 3.可视化
由于semantic kitti api我的环境无法配置成功，所以我可视化了一张voxel的俯视图：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="semantic_kitti.assets/voxel.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图1：俯视图
    </div>
</center>