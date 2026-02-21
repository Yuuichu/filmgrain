# iOS 快捷指令配置指南

## 1. 部署 API

### 步骤 1: 推送到 GitHub
```bash
cd G:\Repository\Filmgrain
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/filmgrain.git
git push -u origin main
```

### 步骤 2: 部署到 Vercel
1. 访问 [vercel.com](https://vercel.com) 并用 GitHub 登录
2. 点击 "New Project"
3. 选择你的 `filmgrain` 仓库
4. 点击 "Deploy"
5. 等待部署完成，获得 URL（如 `https://filmgrain-xxx.vercel.app`）

---

## 2. 创建 iOS 快捷指令

打开 iPhone/iPad 的「快捷指令」App，创建新快捷指令：

### 动作列表（按顺序添加）

#### 动作 1: 选择照片
```
从相簿选择照片
  - 选择多张: 关闭
```

#### 动作 2: 选择 ISO
```
从菜单中选择
  - 提示: "选择 ISO 感光度"
  - 选项:
    - ISO 100 (细腻)
    - ISO 400 (标准)
    - ISO 800 (明显颗粒)
    - ISO 1600 (粗颗粒)
```

#### 动作 3: 设置 ISO 值
```
如果 菜单结果 是 "ISO 100 (细腻)"
  设置变量 iso 为 100
否则如果 菜单结果 是 "ISO 400 (标准)"
  设置变量 iso 为 400
否则如果 菜单结果 是 "ISO 800 (明显颗粒)"
  设置变量 iso 为 800
否则
  设置变量 iso 为 1600
结束如果
```

#### 动作 4: 选择胶片类型
```
从菜单中选择
  - 提示: "选择胶片类型"
  - 选项:
    - 🎨 彩色胶片
    - ⬛ 黑白胶片
```

#### 动作 5: 设置模式
```
如果 菜单结果 包含 "彩色"
  设置变量 mode 为 "color"
否则
  设置变量 mode 为 "bw"
结束如果
```

#### 动作 6: 图片转 Base64
```
将 选择的照片 进行 Base64 编码
设置变量 imageBase64 为 Base64 编码结果
```

#### 动作 7: 构建请求体
```
获取内容
  - URL: https://你的域名.vercel.app/api/grain
  - 方法: POST
  - 请求体: JSON
    {
      "image": imageBase64,
      "iso": iso,
      "mode": mode
    }
```

#### 动作 8: 解析响应
```
获取 JSON 中 "image" 的值
设置变量 resultBase64 为 JSON值
```

#### 动作 9: 解码并保存
```
将 resultBase64 进行 Base64 解码
存储到相册
```

#### 动作 10: 显示结果
```
显示通知 "✅ 胶片颗粒效果已应用"
快速查看 解码后的图片
```

---

## 3. 简化版快捷指令

如果觉得太复杂，可以用这个简化版（固定 ISO 400 彩色）:

### 简化版动作列表

1. **选择照片** - 从相簿选择照片
2. **Base64 编码** - 将照片进行 Base64 编码
3. **获取 URL 内容**
   - URL: `https://你的域名.vercel.app/api/grain`
   - 方法: POST
   - 请求体类型: JSON
   - 请求体: `{"image": "编码结果", "iso": 400, "mode": "color"}`
4. **获取 JSON 值** - 获取 `image` 字段
5. **Base64 解码**
6. **存储到相册**
7. **快速查看**

---

## 4. 测试 API

部署后可以先用 curl 测试:

```bash
# 测试 API 是否正常
curl https://你的域名.vercel.app/api/grain

# 测试图片处理 (需要先将图片转为 base64)
base64 -i test.jpg | curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(cat -)\", \"iso\": 400}" \
  https://你的域名.vercel.app/api/grain
```

---

## 常见问题

### Q: 处理很慢？
A: Vercel 免费版冷启动需要几秒，首次请求会慢一些

### Q: 图片太大报错？
A: API 限制最大 2048px，超大图片会自动缩小

### Q: 快捷指令超时？
A: 在「获取 URL 内容」动作中增加超时时间到 60 秒
