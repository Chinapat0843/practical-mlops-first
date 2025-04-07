import onnx

# โหลดไฟล์ .onnx
model = onnx.load("distilroberta-sequence-classification.onnx")

# ตรวจสอบว่ามี graph หรือไม่
if model.graph.node:
    print("✅ มี graph: ไฟล์ ONNX ใช้ได้")
else:
    print("❌ ไม่มี graph: ไฟล์ ONNX เสียหรือ export ไม่สมบูรณ์")

print("ชื่อ input:", [inp.name for inp in model.graph.input])
print("ชื่อ output:", [out.name for out in model.graph.output])
print("จำนวน nodes:", len(model.graph.node))
