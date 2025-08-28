function C = config()
% 全局配置（可在实验脚本内覆盖个别字段）

% ========== 运动学 / 轨迹 ==========
C.deg = pi/180;                         % 度转弧度
C.joint_start_deg = [  0, -90,  90,   0,  0,  0];
C.joint_end_deg   = [ 30, -120, 130, -30, 40, 30];

% ========== 故障注入 ==========
C.static_bias_deg = 0.5;                % 固定静态偏差
C.use_dynamic     = false;              % 是否叠加正弦扰动
C.A_dyn_deg       = 0.0;                % 动态扰动幅值（deg）
C.f_dyn_hz        = 1.0;                % 动态扰动频率（Hz）

% ========== 采样 & 数据 ==========
C.T_sec        = 5;                     % 轨迹时长
C.sample_hz    = 200;                   % 采样频率（可在脚本中覆盖）
C.train_val_test = [0.70, 0.15, 0.15]; % 划分比例
C.feature_type = 'pos+eul';             % 双特征融合（位置 + 欧拉姿态）
C.class_names  = {'normal','J1','J2','J3','J4','J5','J6'}; % 7 类

% ========== 网络结构 ==========
C.hidden_sizes    = [13, 13];           % 2 层隐藏层
C.trainFcn        = 'traingdx';         % 训练算法（Levenberg-Marquardt 如显存足可用 'trainlm'）
C.performFcn      = 'crossentropy';     % 交叉熵损失
C.max_epoch       = 1000;
C.max_fail        = 50;
C.minibatch       = 0;                  % patternnet 不用手动 batch（内部自适应）
C.showWindow      = true;               % 训练窗口可视化
C.showCommandLine = true;

% ========== 输出/保存 ==========
C.save_tag = 'default';

end
