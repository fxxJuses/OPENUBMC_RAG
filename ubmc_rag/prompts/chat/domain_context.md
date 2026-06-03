## openUBMC 架构背景

### 微组件架构
openUBMC 采用微组件架构，每个组件独立运行，通过 MDS (Microcomponent Development System) 框架管理。
组件间通过 MDB 接口进行 RPC 通信。

### 已知微组件
sensor, sensor_mgmt, devmon, vpd, frudata, fructrl, bus_tools, libipmi,
power_mgmt, bios, pcie_device, mdb_interface

### 关键文件类型
- service.json: 组件服务定义，包含依赖关系和接口声明
- model.json: MDS 数据模型定义
- ipmi.json: IPMI 命令定义
- *_app.lua: Lua 组件入口文件
- main.cpp / main.lua: C/Lua 程序入口

### 代码约定
- Lua 函数通常为 snake_case（如 get_sensor_data）
- C 函数通常为前缀_功能（如 ipmi_get_sensor_reading）
- MDS 类名通常为 PascalCase（如 ThresholdSensor）
