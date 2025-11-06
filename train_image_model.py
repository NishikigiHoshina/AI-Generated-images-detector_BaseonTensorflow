import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# 配置GPU内存增长（避免OOM）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 数据路径配置（请根据实际情况修改）
train_dir = 'data/images/train'
val_dir = 'data/images/validation'
test_dir = 'data/images/test'
model_save_path = 'models/image_AIgenerated_model.h5'

# 确保模型保存目录存在
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 图像尺寸和批量大小
img_height, img_width = 224, 224
batch_size = 32

# 数据增强配置
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 加载训练数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# 加载验证数据
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# 加载测试数据
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 打印类别映射（用于调试）
print("类别映射:", train_generator.class_indices)

# 构建模型 - 使用迁移学习
base_model = MobileNetV2(weights='imagenet', include_top=False, 
                        input_shape=(img_height, img_width, 3))

# 冻结基础模型层（前100层可训练，提高迁移学习效果）
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

# 构建自定义模型
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)  # 2个类别：Fake(0)和Real(1)

model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型（先使用低学习率微调）
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 定义回调函数
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# 训练模型
print("开始训练图片模型...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, early_stopping]
)

# 评估模型
print("评估模型性能...")
evaluation = model.evaluate(test_generator)
print(f"测试集损失: {evaluation[0]:.4f}, 测试集准确率: {evaluation[1]*100:.2f}%")

# 保存训练历史（可选，用于可视化）
np.save('models/training_history.npy', history.history)