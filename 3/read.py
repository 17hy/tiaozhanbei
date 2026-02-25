from ultralytics import YOLO

# 加载你的模型
model = YOLO("C:/Users/Guohu/Desktop/1/1/best.pt")

# 打印模型里自带的类别字典
print(model.names)