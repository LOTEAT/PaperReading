<!--
 * @Author: LOTEAT
 * @Date: 2024-09-11 19:54:00
-->
## Dubins Curve
- 前置知识：<a href="../../MotionPlanning/MotionModel/OdometryMotionModel/odometry_motion_model.md">OdometryModel</a>
- 论文推荐：[Classification of the Dubins set](https://bpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/e/700/files/2014/04/Dubins_Set_Robotics_2001.pdf)

### 1. 推导过程
Dubins曲线是Dubins在1957年提出的，后来被Reeds和Shepp证明。但是原始的Dubins曲线过于古早，所以这里更关注[这篇论文](https://bpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/e/700/files/2014/04/Dubins_Set_Robotics_2001.pdf)。


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="bézier_curve.assets/bezier_curve_1order.gif" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图1：1阶贝塞尔曲线
    </div>
</center>
