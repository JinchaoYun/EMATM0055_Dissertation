function eul = euler_zyx_from_R(R)
% Z-Y-X 欧拉角（yaw-pitch-roll），单位：deg
% 与论文公式一致：roll = atan2(r32, r33), pitch = atan2(-r31, sqrt(r11^2+r21^2)), yaw = atan2(r21, r11)
r11 = R(1,1); r21 = R(2,1); r31 = R(3,1);
r32 = R(3,2); r33 = R(3,3);
roll  = atan2(r32, r33);
pitch = atan2(-r31, sqrt(r11^2 + r21^2));
yaw   = atan2(r21, r11);
eul = [roll, pitch, yaw] * 180/pi;   % [roll, pitch, yaw] (deg)
end
