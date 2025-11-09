# face
import bpy
import os

# === 設定 ===
# 👇あなたのOBJパスと出力先を指定
OBJ_PATH = r"C:\Users\あなた\Documents\face.obj"
OUT_PATH = r"C:\Users\あなた\Documents\rendered_face.png"

# === シーン初期化 ===
bpy.ops.wm.read_factory_settings(use_empty=True)

# === モデル読み込み ===
bpy.ops.import_scene.obj(filepath=OBJ_PATH)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# === カメラ ===
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0.0, -1.2, 0.6)
cam_obj.rotation_euler = (1.1, 0.0, 0.0)
bpy.context.scene.camera = cam_obj

# === 光源 ===
light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
light_data.energy = 1500
light_data.size = 1.0
light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
light_obj.location = (0.6, -0.8, 1.0)
bpy.context.collection.objects.link(light_obj)

# === スキンマテリアル ===
mat = bpy.data.materials.new(name="SkinMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
for n in nodes: nodes.remove(n)

# ノード構築
output = nodes.new(type='ShaderNodeOutputMaterial')
output.location = (400, 0)

bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)
bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1)
bsdf.inputs['Subsurface'].default_value = 0.15
bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.8, 0.6)
bsdf.inputs['Subsurface Color'].default_value = (0.9, 0.7, 0.6, 1)
bsdf.inputs['Roughness'].default_value = 0.5
bsdf.inputs['Specular'].default_value = 0.4

links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# マテリアルを適用
if len(obj.data.materials):
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)

# === Cycles設定 ===
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128
scene.cycles.device = 'GPU' if bpy.context.preferences.addons.get("cycles") else 'CPU'
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.filepath = OUT_PATH

# === レンダリング ===
bpy.ops.render.render(write_still=True)
print(f"✅ レンダリング完了: {OUT_PATH}")
