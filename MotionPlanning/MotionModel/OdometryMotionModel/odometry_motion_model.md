<!--
 * @Author: LOTEAT
 * @Date: 2024-07-31 11:08:37
-->
## Odometry Motion Model
- 前置知识: 无

### 推导
Odometry Motion Model是通过$\delta_{rot1}$，$\delta_{rot2}$和$\delta_{trans}$控制robot进行移动，那么在这三个就是控制的$u_t$，通常这三个值是有robot自带的传感器所测量出来的。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="odometry_motion_model.assets/page1.jpg" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
  	</div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="odometry_motion_model.assets/page2.jpg" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
  	</div>
</center>
这里再额外补充一下车辆简单模型。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="odometry_motion_model.assets/page3.jpg" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
  	</div>
</center>
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="odometry_motion_model.assets/page4.jpg" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
  	</div>
</center>

### Demo实现
[Odometry Motion Model Demo](https://github.com/LOTEAT/PaperReading/blob/main/MotionPlanning/MotionModel/OdometryMotionModel/OdometryModelDemo).