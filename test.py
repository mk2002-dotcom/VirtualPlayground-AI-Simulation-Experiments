import pybullet as p
import pybullet_data  # サンプルURDFを使う
import time

# 1. PyBulletをGUIモードで起動
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)  # 重力設定

# 2. サンプルURDFフォルダを検索パスに追加
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. 平面とロボットをロード
planeId = p.loadURDF("plane.urdf")                  # 床
r2d2StartPos = [0, 0, 1]                            # 初期位置
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2Id = p.loadURDF("r2d2.urdf", r2d2StartPos, r2d2StartOrientation)

# 4. シミュレーションを動かす
for i in range(240):   # 240ステップ = 1秒（240Hzの場合）
    p.stepSimulation()
    time.sleep(1./240.)  # スローダウンして見やすく

# 5. 終了処理
p.disconnect()

